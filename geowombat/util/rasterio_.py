from collections import namedtuple

import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import aligned_target, transform_bounds
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


def union(filenames, crs=None, resampling='nearest'):

    """
    Transforms a list of images to the union of all the files

    Args:
        filenames (list): The file names to mosaic.
        crs (Optional[crs]): The CRS to warp to.
        resampling (Optional[str]): The resampling method. Choices are ['average', 'bilinear', 'cubic',
            'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].

    Returns:
        ``list``
    """

    with rio.open(filenames[0]) as src:

        if not crs:
            crs = src.crs

        res = src.res

        # Transform the extent to the reference CRS
        min_left, min_bottom, max_right, max_top = transform_bounds(src.crs,
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
        min_left = min(min_left, left)
        min_bottom = min(min_bottom, bottom)
        max_right = max(max_right, right)
        max_top = max(max_top, top)

    # Align the cells
    dst_transform, dst_width, dst_height = align_bounds(min_left,
                                                        min_bottom,
                                                        max_right,
                                                        max_top,
                                                        res)

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


def warp_to_vrt(filename, bounds=None, crs=None, res=None, resampling='nearest'):

    """
    Warps an image to a VRT object

    Args:
        filename (str): The input file name.
        bounds (Optional[tuple]): The extent bounds to warp to.
        crs (Optional[crs]): The CRS to warp to.
        res (Optional[tuple]): The cell resolution to warp to.
        resampling (Optional[str]): The resampling method. Choices are ['average', 'bilinear', 'cubic',
            'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].

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
                           'warp_mem_limit': 512,
                           'warp_extras': {'multi': True}}

            # 'warp_extras': {'warp_option': 'NUM_THREADS=ALL_CPUS'}

            with WarpedVRT(src, **vrt_options) as vrt:
                return vrt
