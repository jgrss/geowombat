import os

from ..config import config
from .rasterio_ import get_ref_image_meta, union, warp_to_vrt

import xarray as xr
from xarray.ufuncs import maximum as xr_maximum


def mosaic(filenames, resampling='nearest', **kwargs):

    """
    Mosaics a list of images

    Args:
        filenames (list): A list of file names to mosaic.
        resampling (Optional[str]): The resampling method.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

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

            ds = xr_maximum(ds, dsb)

            # ds = ds.combine_first(dsb)

    return ds.assign_attrs(**attrs)


def concat(filenames, resampling='nearest', **kwargs):

    """
    Concatenates a list of images

    Args:
        filenames (list): A list of file names to concatenate.
        resampling (Optional[str]): The resampling method.
        kwargs (Optional[dict]): Keyword arguments passed to ``xarray.open_rasterio``.

    Returns:
        ``xarray.DataArray``
    """

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

    # Warp all images and concatenate into a DataArray
    return xr.concat([xr.open_rasterio(warp_to_vrt(fn, resampling=resampling, **ref_kwargs), **kwargs)
                      for fn in filenames], dim='time')
