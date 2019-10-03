import os

from ..errors import logger
from ..config import config
from .rasterio_ import get_ref_image_meta, union, warp_to_vrt, get_file_bounds

import xarray as xr
from xarray.ufuncs import maximum as xr_maximum
from xarray.ufuncs import minimum as xr_mininum


def mosaic(filenames, overlap='max', resampling='nearest', **kwargs):

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

    ref_image = None
    ref_kwargs = {'crs': None}

    # Check if there is a reference image
    if 'ref_image' in config:
        ref_image = config['ref_image']

    if isinstance(ref_image, str) and os.path.isfile(ref_image):

        # Get the metadata from the reference image
        ref_meta = get_ref_image_meta(ref_image)

        ref_kwargs = {'crs': ref_meta.crs}

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


def concat(filenames, how='reference', resampling='nearest', **kwargs):

    """
    Concatenates a list of images

    Args:
        filenames (list): A list of file names to concatenate.
        how (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union', 'reference'].

            * reference: Use the bounds of the reference image
            * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
            * union: Use the union (i.e., maximum extent) of all the image bounds

        resampling (Optional[str]): The resampling method.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

    if how not in ['intersection', 'union', 'reference']:
        logger.exception("  Only 'intersection', 'union', and 'reference' are supported.")

    ref_image = None
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

    if how == 'intersection':

        # Get the intersecting bounds of all images
        ref_kwargs['bounds'] = get_file_bounds(filenames,
                                               how='intersection',
                                               crs=ref_kwargs['crs'],
                                               return_bounds=True)

    elif how == 'union':

        # Get the union bounds of all images
        ref_kwargs['bounds'] = get_file_bounds(filenames,
                                               how='union',
                                               crs=ref_kwargs['crs'],
                                               return_bounds=True)

    # Warp all images and concatenate into a DataArray
    return xr.concat([xr.open_rasterio(warp_to_vrt(fn,
                                                   resampling=resampling,
                                                   **ref_kwargs), **kwargs)
                      for fn in filenames], dim='time')
