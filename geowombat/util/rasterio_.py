from collections import namedtuple

from ..errors import logger

import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import aligned_target, transform_bounds
from rasterio.transform import array_bounds
from affine import Affine


def align_bounds(minx, miny, maxx, maxy, res):

    """
    Aligns bounds to resolution

    Args:
        minx (float)
        miny (float)
        maxx (float)
        maxy (float)
        res (tuple)

    Returns:
        ``affine.Affine``, ``int``, ``int``
    """

    xres, yres = res

    new_height = (maxy - miny) / yres
    new_width = (maxx - minx) / xres

    new_transform = Affine(xres, 0.0, minx, 0.0, -yres, maxy)

    return aligned_target(new_transform, new_width, new_height, res)


def get_file_bounds(filenames,
                    how='intersection',
                    crs=None,
                    res=None,
                    return_bounds=False):

    """
    Gets the union of all files

    Args:
        filenames (list): The file names to mosaic.
        how (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union'].
        crs (Optional[crs]): The CRS to warp to.
        res (Optional[tuple]): The cell resolution to warp to.
        return_bounds (Optional[bool]): Whether to return the bounds tuple.

    Returns:
        transform, width, height
    """

    if how not in ['intersection', 'union']:
        logger.exception("  Only 'intersection' and 'union' are supported.")

    with rio.open(filenames[0]) as src:

        if not crs:
            crs = src.crs

        if not res:
            res = src.res

        # Transform the extent to the reference CRS
        bounds_left, bounds_bottom, bounds_right, bounds_top = transform_bounds(src.crs,
                                                                                crs,
                                                                                src.bounds.left,
                                                                                src.bounds.bottom,
                                                                                src.bounds.right,
                                                                                src.bounds.top,
                                                                                densify_pts=21)

    for fn in filenames[1:]:

        with rio.open(fn) as src:

            # Transform the extent to the reference CRS
            left, bottom, right, top = transform_bounds(src.crs,
                                                        crs,
                                                        src.bounds.left,
                                                        src.bounds.bottom,
                                                        src.bounds.right,
                                                        src.bounds.top,
                                                        densify_pts=21)

        # Update the mosaic bounds
        if how == 'union':

            bounds_left = min(bounds_left, left)
            bounds_bottom = min(bounds_bottom, bottom)
            bounds_right = max(bounds_right, right)
            bounds_top = max(bounds_top, top)

        elif how == 'intersection':

            bounds_left = max(bounds_left, left)
            bounds_bottom = max(bounds_bottom, bottom)
            bounds_right = min(bounds_right, right)
            bounds_top = min(bounds_top, top)

    # Align the cells
    bounds_transform, bounds_width, bounds_height = align_bounds(bounds_left,
                                                                 bounds_bottom,
                                                                 bounds_right,
                                                                 bounds_top,
                                                                 res)

    if return_bounds:
        return array_bounds(bounds_height, bounds_width, bounds_transform)
    else:
        return bounds_transform, bounds_width, bounds_height


def union(filenames, crs=None, res=None, resampling='nearest'):

    """
    Transforms a list of images to the union of all the files

    Args:
        filenames (list): The file names to mosaic.
        crs (Optional[object]): The CRS to warp to.
        res (Optional[tuple]): The cell resolution to warp to.
        resampling (Optional[str]): The resampling method. Choices are ['average', 'bilinear', 'cubic',
            'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].

    Returns:
        ``list`` of ``rasterio.vrt.WarpedVRT`` objects
    """

    if resampling not in ['average', 'bilinear', 'cubic', 'cubic_spline',
                          'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest']:

        logger.exception('  The resampling method is not supported by rasterio.')

    # Get the union bounds of all images
    dst_transform, dst_width, dst_height = get_file_bounds(filenames,
                                                           how='union',
                                                           crs=crs,
                                                           res=res)

    vrt_options = {'resampling': getattr(Resampling, resampling),
                   'crs': crs,
                   'transform': dst_transform,
                   'height': dst_height,
                   'width': dst_width,
                   'nodata': 0,
                   'warp_mem_limit': 512,
                   'warp_extras': {'multi': True}}

    vrt_list = list()

    # Warp each image to a common grid
    for fn in filenames:

        with rio.open(fn) as src:

            with WarpedVRT(src, **vrt_options) as vrt:
                vrt_list.append(vrt)

    return vrt_list


def get_ref_image_meta(filename):

    """
    Gets warping information from a reference image

    Args:
        filename (str): The file name to get information from.

    Returns:
        ``collections.namedtuple``
    """

    WarpInfo = namedtuple('WarpInfo', 'bounds crs res')

    with rio.open(filename) as src:

        bounds = src.bounds
        crs = src.crs
        res = src.res

    return WarpInfo(bounds=bounds, crs=crs, res=res)


def warp_to_vrt(filename,
                resampling='nearest',
                bounds=None,
                crs=None,
                res=None,
                warp_mem_limit=512):

    """
    Warps an image to a VRT object

    Args:
        filename (str): The input file name.
        resampling (Optional[str]): The resampling method. Choices are ['average', 'bilinear', 'cubic',
            'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        bounds (Optional[tuple]): The extent bounds to warp to.
        crs (Optional[object]): The CRS to warp to.
        res (Optional[tuple]): The cell resolution to warp to.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.

    Returns:
        ``rasterio.vrt.WarpedVRT``
    """

    with rio.open(filename) as src:

        if not bounds:
            bounds = src.bounds

        if not crs:
            crs = src.crs

        if not res:
            res = src.res

        xres, yres = res

        left, bottom, right, top = bounds

        dst_height = (top - bottom) / yres
        dst_width = (right - left) / xres

        # Do not warp if all the key metadata match the reference information
        if (src.bounds == bounds) and (src.res == res) and (src.crs == crs) and (src.width == dst_width) and (src.height == dst_height):
            return filename
        else:

            # Output image transform
            dst_transform = Affine(xres, 0.0, left, 0.0, -yres, top)

            vrt_options = {'resampling': getattr(Resampling, resampling),
                           'crs': crs,
                           'transform': dst_transform,
                           'height': dst_height,
                           'width': dst_width,
                           'nodata': 0,
                           'warp_mem_limit': warp_mem_limit,
                           'warp_extras': {'multi': True}}

            # 'warp_extras': {'warp_option': 'NUM_THREADS=ALL_CPUS'}

            with WarpedVRT(src, **vrt_options) as vrt:
                return vrt
