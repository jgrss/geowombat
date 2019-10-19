import os

from ..core.windows import get_window_offsets
from ..errors import logger
from ..config import config
from .rasterio_ import get_ref_image_meta, union, warp, get_file_bounds

import xarray as xr
from xarray.ufuncs import maximum as xr_maximum
from xarray.ufuncs import minimum as xr_mininum


def _update_kwarg(ref_obj, ref_kwargs, key):

    """
    Updates keyword arguments for global config parameters

    Args:
        ref_obj (str or object)
        ref_kwargs (dict)
        key (str)

    Returns:
        ``dict``
    """

    if isinstance(ref_obj, str) and os.path.isfile(ref_obj):

        # Get the metadata from the reference image
        ref_meta = get_ref_image_meta(ref_obj)
        ref_kwargs[key] = getattr(ref_meta, key)

    else:

        if ref_obj:
            ref_kwargs[key] = ref_obj

    return ref_kwargs


def warp_open(filename,
              band_names=None,
              resampling='nearest',
              return_windows=False,
              **kwargs):

    """
    Warps and opens a file

    Args:
        filename (str): The file to open.
        band_names (Optional[int, str, or list]): The band names.
        resampling (Optional[str]): The resampling method.
        return_windows (Optional[bool]): Whether to return block windows.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    ref_kwargs = {'bounds': None, 'crs': None, 'res': None}

    # Check if there is a reference image
    if 'ref_image' in config:

        ref_image = config['ref_image']

        if isinstance(ref_image, str) and os.path.isfile(ref_image):

            # Get the metadata from the reference image
            ref_meta = get_ref_image_meta(ref_image)

            ref_kwargs = {'bounds': ref_meta.bounds,
                          'crs': ref_meta.crs,
                          'res': ref_meta.res}

    if 'ref_bounds' in config:
        ref_kwargs = _update_kwarg(config['ref_bounds'], ref_kwargs, 'bounds')

    if 'ref_crs' in config:
        ref_kwargs = _update_kwarg(config['ref_crs'], ref_kwargs, 'crs')

    if 'ref_res' in config:
        ref_kwargs = _update_kwarg(config['ref_res'], ref_kwargs, 'res')

    with xr.open_rasterio(warp(filename,
                               resampling=resampling,
                               **ref_kwargs),
                          **kwargs) as src:

        if band_names:
            src.coords['band'] = band_names
        else:

            if src.gw.sensor:
                src.coords['band'] = list(src.gw.wavelengths[src.gw.sensor]._fields)

        if return_windows:

            if isinstance(kwargs['chunks'], tuple):
                chunksize = kwargs['chunks'][-1]
            else:
                chunksize = kwargs['chunks']

            src.attrs['block_windows'] = get_window_offsets(src.shape[-2],
                                                            src.shape[-1],
                                                            chunksize,
                                                            chunksize,
                                                            return_as='list')

        return src


def mosaic(filenames,
           overlap='max',
           resampling='nearest',
           **kwargs):

    """
    Mosaics a list of images

    Args:
        filenames (list): A list of file names to mosaic.
        overlap (Optional[str]): The keyword that determines how to handle overlapping data.
            Choices are ['min', 'max', 'mean'].
        resampling (Optional[str]): The resampling method.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    if overlap not in ['min', 'max', 'mean']:
        logger.exception("  The overlap argument must be one of ['min', 'max', 'mean'].")

    ref_kwargs = {'crs': None, 'res': None}

    # Check if there is a reference image
    if 'ref_image' in config:

        ref_image = config['ref_image']

        if isinstance(ref_image, str) and os.path.isfile(ref_image):

            # Get the metadata from the reference image
            ref_meta = get_ref_image_meta(ref_image)

            ref_kwargs = {'crs': ref_meta.crs,
                          'res': ref_meta.res}

    if 'ref_crs' in config:
        ref_kwargs = _update_kwarg(config['ref_crs'], ref_kwargs, 'crs')

    if 'ref_res' in config:
        ref_kwargs = _update_kwarg(config['ref_res'], ref_kwargs, 'res')

    # Get the union of all images
    union_grids = union(filenames,
                        resampling=resampling,
                        **ref_kwargs)

    ds = xr.open_rasterio(union_grids[0], **kwargs)

    attrs = ds.attrs

    for fn in union_grids[1:]:

        with xr.open_rasterio(fn, **kwargs) as dsb:

            if overlap == 'min':
                ds = xr_mininum(ds, dsb)
            elif overlap == 'max':
                ds = xr_maximum(ds, dsb)
            elif overlap == 'mean':
                ds = (ds + dsb) / 2.0

            # ds = ds.combine_first(dsb)

    return ds.assign_attrs(**attrs)


def concat(filenames,
           how='reference',
           resampling='nearest',
           time_names=None,
           overlap='max',
           **kwargs):

    """
    Concatenates a list of images

    Args:
        filenames (list): A list of file names to concatenate.
        how (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union', 'reference'].

            * reference: Use the bounds of the reference image
            * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
            * union: Use the union (i.e., maximum extent) of all the image bounds

        resampling (Optional[str]): The resampling method.
        time_names (Optional[1d array-like]): A list of names to give the time dimension if ``bounds`` is given.
        overlap (Optional[str]): The keyword that determines how to handle overlapping data.
            Choices are ['min', 'max', 'mean'].
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    if how not in ['intersection', 'union', 'reference']:
        logger.exception("  Only 'intersection', 'union', and 'reference' are supported.")

    ref_kwargs = {'bounds': None, 'crs': None, 'res': None}

    # Check if there is a reference image
    if 'ref_image' in config:

        ref_image = config['ref_image']

        if isinstance(ref_image, str) and os.path.isfile(ref_image):

            # Get the metadata from the reference image
            ref_meta = get_ref_image_meta(ref_image)

            ref_kwargs = {'bounds': ref_meta.bounds,
                          'crs': ref_meta.crs,
                          'res': ref_meta.res}

    if 'ref_bounds' in config:
        ref_kwargs = _update_kwarg(config['ref_bounds'], ref_kwargs, 'bounds')

    if 'ref_crs' in config:
        ref_kwargs = _update_kwarg(config['ref_crs'], ref_kwargs, 'crs')

    if 'ref_res' in config:
        ref_kwargs = _update_kwarg(config['ref_res'], ref_kwargs, 'res')

    # Replace the bounds keyword, if needed
    if how == 'intersection':

        # Get the intersecting bounds of all images
        ref_kwargs['bounds'] = get_file_bounds(filenames,
                                               how='intersection',
                                               crs=ref_kwargs['crs'],
                                               res=ref_kwargs['res'],
                                               return_bounds=True)

    elif how == 'union':

        # Get the union bounds of all images
        ref_kwargs['bounds'] = get_file_bounds(filenames,
                                               how='union',
                                               crs=ref_kwargs['crs'],
                                               res=ref_kwargs['res'],
                                               return_bounds=True)

    if time_names:

        concat_list = list()
        new_time_names = list()

        # Check the time names for duplicates
        for tidx in range(0, len(time_names)):

            if list(time_names).count(time_names[tidx]) > 1:

                if time_names[tidx] not in new_time_names:

                    # Get the file names to mosaic
                    filenames_mosaic = [filenames[i] for i in range(0, len(time_names))
                                        if time_names[i] == time_names[tidx]]

                    # Mosaic the images into a single-date array
                    concat_list.append(mosaic(filenames_mosaic,
                                              overlap=overlap,
                                              resampling=resampling,
                                              **kwargs))

                    new_time_names.append(time_names[tidx])

            else:

                new_time_names.append(time_names[tidx])

                # Warp the date
                concat_list.append(xr.open_rasterio(warp(filenames[tidx],
                                                         resampling=resampling,
                                                         **ref_kwargs),
                                                    **kwargs))

        # Warp all images and concatenate along the 'time' axis into a DataArray
        output = xr.concat(concat_list, dim='time')

        # Assign the new time band names
        return output.assign_coords(time=new_time_names)

    else:

        # Warp all images and concatenate along the 'time' axis into a DataArray
        return xr.concat([xr.open_rasterio(warp(fn,
                                                resampling=resampling,
                                                **ref_kwargs), **kwargs)
                          for fn in filenames], dim='time')
