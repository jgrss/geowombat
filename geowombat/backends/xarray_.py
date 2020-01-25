import os

from ..core.windows import get_window_offsets
from ..core.util import parse_filename_dates
from ..errors import logger
from ..config import config
from .rasterio_ import get_ref_image_meta, warp, warp_images, get_file_bounds, transform_crs

import dask.array as da
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


def _check_config_globals(filenames, bounds_by, ref_kwargs):

    """
    Checks global configuration parameters

    Args:
        filenames (str or str list)
        bounds_by (str)
        ref_kwargs (dict)
    """

    # Check if there is a reference image
    if config['ref_image']:

        if isinstance(config['ref_image'], str) and os.path.isfile(config['ref_image']):

            # Get the metadata from the reference image
            ref_meta = get_ref_image_meta(config['ref_image'])

            ref_kwargs['bounds'] = ref_meta.bounds
            ref_kwargs['crs'] = ref_meta.crs
            ref_kwargs['res'] = ref_meta.res

    else:

        if config['ref_bounds']:
            ref_kwargs = _update_kwarg(config['ref_bounds'], ref_kwargs, 'bounds')
        else:

            if isinstance(filenames, str):

                # Use the bounds of the image
                ref_kwargs['bounds'] = get_file_bounds([filenames],
                                                       bounds_by='reference',
                                                       crs=ref_kwargs['crs'],
                                                       res=ref_kwargs['res'],
                                                       return_bounds=True)

            else:

                # Replace the bounds keyword, if needed
                if bounds_by.lower() == 'intersection':

                    # Get the intersecting bounds of all images
                    ref_kwargs['bounds'] = get_file_bounds(filenames,
                                                           bounds_by='intersection',
                                                           crs=ref_kwargs['crs'],
                                                           res=ref_kwargs['res'],
                                                           return_bounds=True)

                elif bounds_by.lower() == 'union':

                    # Get the union bounds of all images
                    ref_kwargs['bounds'] = get_file_bounds(filenames,
                                                           bounds_by='union',
                                                           crs=ref_kwargs['crs'],
                                                           res=ref_kwargs['res'],
                                                           return_bounds=True)

                elif bounds_by.lower() == 'reference':

                    # Use the bounds of the first image
                    ref_kwargs['bounds'] = get_file_bounds(filenames,
                                                           bounds_by='reference',
                                                           crs=ref_kwargs['crs'],
                                                           res=ref_kwargs['res'],
                                                           return_bounds=True)

                else:
                    logger.exception("  Choose from 'intersection', 'union', or 'reference'.")

                config['ref_bounds'] = ref_kwargs['bounds']

        if config['ref_crs']:
            ref_kwargs = _update_kwarg(config['ref_crs'], ref_kwargs, 'crs')

        if config['ref_res']:
            ref_kwargs = _update_kwarg(config['ref_res'], ref_kwargs, 'res')

    return ref_kwargs


def warp_open(filename,
              band_names=None,
              resampling='nearest',
              dtype=None,
              return_windows=False,
              warp_mem_limit=512,
              num_threads=1,
              tap=False,
              **kwargs):

    """
    Warps and opens a file

    Args:
        filename (str): The file to open.
        band_names (Optional[int, str, or list]): The band names.
        resampling (Optional[str]): The resampling method.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        return_windows (Optional[bool]): Whether to return block windows.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        tap (Optional[bool]): Whether to target align pixels.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    ref_kwargs = {'bounds': None,
                  'crs': None,
                  'res': None,
                  'warp_mem_limit': warp_mem_limit,
                  'num_threads': num_threads,
                  'tap': tap}

    ref_kwargs = _check_config_globals(filename, 'reference', ref_kwargs)

    with xr.open_rasterio(warp(filename,
                               resampling=resampling,
                               **ref_kwargs),
                          **kwargs) as src:

        if band_names:
            src.coords['band'] = band_names
        else:

            if src.gw.sensor:

                if src.gw.sensor not in src.gw.avail_sensors:

                    logger.warning('  The {} sensor is not currently supported.\nChoose from [{}].'.format(src.gw.sensor,
                                                                                                           ', '.join(src.gw.avail_sensors)))

                else:

                    new_band_names = list(src.gw.wavelengths[src.gw.sensor]._fields)

                    # Avoid nested opens within a `config` context
                    if len(new_band_names) != len(src.band.values.tolist()):

                        logger.warning('  The new bands, {}, do not match the sensor bands, {}.'.format(new_band_names,
                                                                                                        src.band.values.tolist()))

                    else:

                        src.coords['band'] = new_band_names
                        src.attrs['sensor'] = src.gw.sensor_names[src.gw.sensor]

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

        src.attrs['filename'] = filename
        src.attrs['resampling'] = resampling

        if dtype:

            attrs = src.attrs.copy()
            return src.astype(dtype).assign_attrs(**attrs)

        else:
            return src


def mosaic(filenames,
           overlap='max',
           bounds_by='reference',
           resampling='nearest',
           band_names=None,
           dtype=None,
           warp_mem_limit=512,
           num_threads=1,
           **kwargs):

    """
    Mosaics a list of images

    Args:
        filenames (list): A list of file names to mosaic.
        overlap (Optional[str]): The keyword that determines how to handle overlapping data.
            Choices are ['min', 'max', 'mean'].
        bounds_by (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union', 'reference'].

            * reference: Use the bounds of the reference image
            * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
            * union: Use the union (i.e., maximum extent) of all the image bounds

        resampling (Optional[str]): The resampling method.
        band_names (Optional[1d array-like]): A list of names to give the band dimension.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    if overlap not in ['min', 'max', 'mean']:
        logger.exception("  The overlap argument must be one of ['min', 'max', 'mean'].")

    ref_kwargs = {'bounds': None,
                  'crs': None,
                  'res': None,
                  'warp_mem_limit': warp_mem_limit,
                  'num_threads': num_threads}

    ref_kwargs = _check_config_globals(filenames, bounds_by, ref_kwargs)

    # Warp all images to the same grid.
    warped_objects = warp_images(filenames,
                                 resampling=resampling,
                                 **ref_kwargs)

    # Combine the data
    with xr.open_rasterio(warped_objects[0], **kwargs) as ds:

        attrs = ds.attrs.copy()

        for fn in warped_objects[1:]:

            with xr.open_rasterio(fn, **kwargs) as dsb:

                if overlap == 'min':
                    ds = xr_mininum(ds, dsb)
                elif overlap == 'max':
                    ds = xr_maximum(ds, dsb)
                elif overlap == 'mean':
                    ds = (ds + dsb) / 2.0

                # ds = ds.combine_first(dsb)

        ds = ds.assign_attrs(**attrs)

        if band_names:
            ds.coords['band'] = band_names
        else:

            if ds.gw.sensor:

                if ds.gw.sensor not in ds.gw.avail_sensors:

                    logger.warning('  The {} sensor is not currently supported.\nChoose from [{}].'.format(ds.gw.sensor,
                                                                                                           ', '.join(ds.gw.avail_sensors)))

                else:

                    new_band_names = list(ds.gw.wavelengths[ds.gw.sensor]._fields)

                    if len(new_band_names) != len(ds.band.values.tolist()):
                        logger.warning('  The band list length does not match the sensor bands.')
                    else:

                        ds.coords['band'] = new_band_names
                        ds.attrs['sensor'] = ds.gw.sensor_names[ds.gw.sensor]

        ds.attrs['resampling'] = resampling

        if dtype:

            attrs = ds.attrs.copy()
            return ds.astype(dtype).assign_attrs(**attrs)

        else:
            return ds


def concat(filenames,
           stack_dim='time',
           bounds_by='reference',
           resampling='nearest',
           time_names=None,
           band_names=None,
           dtype=None,
           overlap='max',
           warp_mem_limit=512,
           num_threads=1,
           **kwargs):

    """
    Concatenates a list of images

    Args:
        filenames (list): A list of file names to concatenate.
        stack_dim (Optional[str]): The stack dimension. Choices are ['time', 'band'].
        bounds_by (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union', 'reference'].

            * reference: Use the bounds of the reference image
            * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
            * union: Use the union (i.e., maximum extent) of all the image bounds

        resampling (Optional[str]): The resampling method.
        time_names (Optional[1d array-like]): A list of names to give the time dimension.
        band_names (Optional[1d array-like]): A list of names to give the band dimension.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        overlap (Optional[str]): The keyword that determines how to handle overlapping data.
            Choices are ['min', 'max', 'mean']. Only used when mosaicking arrays from the same timeframe.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    if stack_dim.lower() not in ['band', 'time']:
        logger.exception("  The stack dimension should be 'band' or 'time'")

    ref_kwargs = {'bounds': None,
                  'crs': None,
                  'res': None,
                  'warp_mem_limit': warp_mem_limit,
                  'num_threads': num_threads}

    ref_kwargs = _check_config_globals(filenames, bounds_by, ref_kwargs)

    # Keep a copy of the transformed attributes.
    with xr.open_rasterio(warp(filenames[0],
                               resampling=resampling,
                               **ref_kwargs), **kwargs) as ds_:

        attrs = ds_.attrs.copy()

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
                                              bounds_by=bounds_by,
                                              resampling=resampling,
                                              band_names=band_names,
                                              warp_mem_limit=warp_mem_limit,
                                              num_threads=num_threads,
                                              **kwargs))

                    new_time_names.append(time_names[tidx])

            else:

                new_time_names.append(time_names[tidx])

                # Warp the date
                concat_list.append(warp_open(filenames[tidx],
                                             resampling=resampling,
                                             band_names=band_names,
                                             warp_mem_limit=warp_mem_limit,
                                             num_threads=num_threads,
                                             **kwargs))

        # Warp all images and concatenate along the 'time' axis into a DataArray
        output = xr.concat(concat_list, dim=stack_dim.lower())

        # Assign the new time band names
        ds = output.assign_coords(time=new_time_names)

    else:

        # Warp all images and concatenate along
        #   the 'time' axis into a DataArray.
        ds = xr.concat([xr.open_rasterio(warp(fn,
                                              resampling=resampling,
                                              **ref_kwargs), **kwargs)
                        for fn in filenames], dim=stack_dim.lower())

    ds = ds.assign_attrs(**attrs)

    if not time_names and (stack_dim == 'time'):
        ds.coords['time'] = parse_filename_dates(filenames)

    if band_names:
        ds.coords['band'] = band_names
    else:

        if ds.gw.sensor:

            if ds.gw.sensor not in ds.gw.avail_sensors:

                logger.warning('  The {} sensor is not currently supported.\nChoose from [{}].'.format(ds.gw.sensor,
                                                                                                       ', '.join(ds.gw.avail_sensors)))

            else:

                new_band_names = list(ds.gw.wavelengths[ds.gw.sensor]._fields)

                if len(new_band_names) != len(ds.band.values.tolist()):

                    logger.warning('  The new bands, {}, do not match the sensor bands, {}.'.format(new_band_names,
                                                                                                    ds.band.values.tolist()))

                else:

                    ds.coords['band'] = new_band_names
                    ds.attrs['sensor'] = ds.gw.sensor_names[ds.gw.sensor]

    if dtype:
        
        attrs = ds.attrs.copy()
        return ds.astype(dtype).assign_attrs(**attrs)

    else:
        return ds


def to_crs(data_src,
           dst_crs,
           dst_res=None,
           resampling='nearest',
           warp_mem_limit=512,
           num_threads=1):

    """
    Transforms a DataArray to a new coordinate reference system

    Args:
        data_src (DataArray): The data to transform.
        dst_crs (``CRS`` | int | dict | str): The destination CRS.
        dst_res (Optional[tuple]): The destination resolution.
        resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        warp_mem_limit (Optional[int]): The warp memory limit.
        num_threads (Optional[int]): The number of parallel threads.

    Returns:
        ``xarray.DataArray``
    """

    data_dst = xr.DataArray(data=da.from_array(transform_crs(data_src,
                                                             dst_crs,
                                                             dst_res=dst_res,
                                                             resampling=resampling,
                                                             warp_mem_limit=warp_mem_limit,
                                                             num_threads=num_threads),
                                               chunks=data_src.data.chunksize),
                            coords={'band': data_src.band.values.tolist(),
                                    'y': data_src.y.values,
                                    'x': data_src.x.values},
                            dims=('band', 'y', 'x'),
                            attrs=data_src.attrs)

    data_dst.attrs['resampling'] = resampling

    if 'sensor' in data_src.attrs:
        data_dst.attrs['sensor'] = data_src.attrs['sensor']

    if 'filename' in data_src.attrs:
        data_dst.attrs['filename'] = data_src.attrs['filename']

    return data_dst

    # with xr.open_rasterio(transform_crs(data_src,
    #                                     dst_crs,
    #                                     dst_res=dst_res,
    #                                     resampling=resampling,
    #                                     warp_mem_limit=warp_mem_limit,
    #                                     num_threads=num_threads),
    #                       chunks=data_src.data.chunksize) as dst_:
    #
    #     dst_.coords['band'] = data_src.band.values.tolist()
    #
    #     dst_.attrs['resampling'] = resampling
    #
    #     if 'sensor' in data_src.attrs:
    #         dst_.attrs['sensor'] = data_src.attrs['sensor']
    #
    #     if 'filename' in data_src.attrs:
    #         dst_.attrs['filename'] = data_src.attrs['filename']
    #
    #     attrs = data_src.attrs.copy()
    #
    #     return dst_.astype(dtype).assign_attrs(**attrs)
