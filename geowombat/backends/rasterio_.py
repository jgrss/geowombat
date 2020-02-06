import os
import shutil
from collections import namedtuple

from ..errors import logger

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import aligned_target, calculate_default_transform, transform_bounds, reproject
from rasterio.transform import array_bounds, from_bounds
from rasterio.windows import Window
from rasterio.coords import BoundingBox

from affine import Affine
import zarr
import numcodecs


class WriteDaskArray(object):

    """
    ``Rasterio`` wrapper to allow ``dask.array.store`` to save chunks as windows.

    Args:
        filename (str): The file to write to.
        overwrite (Optional[bool]): Whether to overwrite an existing output file.
        separate (Optional[bool]): Whether to write blocks as separate files. Otherwise, write to the same file.
        out_block_type (Optional[str]): The output block type. Choices are ['GTiff', 'zarr'].
            *Only used if ``separate`` = ``True``.
        keep_blocks (Optional[bool]): Whether to keep the blocks stored on disk.
            *Only used if ``separate`` = ``True``.
        gdal_cache (Optional[int]): The GDAL cache size (in MB).
        kwargs (Optional[dict]): Other keyword arguments passed to ``rasterio``.

    Reference:
        Code modified from https://github.com/dymaxionlabs/dask-rasterio

    Returns:
        None
    """

    def __init__(self,
                 filename,
                 overwrite=False,
                 separate=False,
                 out_block_type='zarr',
                 keep_blocks=False,
                 gdal_cache=512,
                 **kwargs):

        self.filename = filename
        self.overwrite = overwrite
        self.separate = separate
        self.out_block_type = out_block_type
        self.keep_blocks = keep_blocks
        self.gdal_cache = gdal_cache
        self.kwargs = kwargs

        self.d_name, f_name = os.path.split(self.filename)
        self.f_base, self.f_ext = os.path.splitext(f_name)

        self.root = None
        self.compressor = None
        self.sub_dir = None
        self.zarr_file = None

        if self.separate:

            if self.out_block_type.lower() not in ['gtiff', 'zarr']:

                logger.warning('  The output block type is not recognized. Save blocks as zarr files.')
                self.out_block_type = 'zarr'

            self.sub_dir = os.path.join(self.d_name, 'sub_tmp_')
            self.zarr_file = os.path.join(self.sub_dir, 'data.zarr')

            self.compressor = numcodecs.Blosc(cname='zstd',
                                              clevel=3,
                                              shuffle=numcodecs.Blosc.BITSHUFFLE)

            if os.path.isdir(self.sub_dir):
                shutil.rmtree(self.sub_dir)

            os.makedirs(self.sub_dir)

    def __setitem__(self, key, item):

        if len(key) == 3:

            index_range, y, x = key
            indexes = list(range(index_range.start + 1, index_range.stop + 1, index_range.step or 1))

        else:

            indexes = 1
            y, x = key

        if self.separate:

            if self.out_block_type.lower() == 'zarr':

                group_name = '{BASE}_y{Y:09d}_x{X:09d}_h{H:09d}_w{W:09d}'.format(BASE=self.f_base,
                                                                                 Y=y.start,
                                                                                 X=x.start,
                                                                                 H=y.stop - y.start,
                                                                                 W=x.stop - x.start)

                group = self.root.create_group(group_name)

                z = group.array('data',
                                item,
                                compressor=self.compressor,
                                dtype=item.dtype.name,
                                chunks=(self.kwargs['blockysize'], self.kwargs['blockxsize']))

                group.attrs['row_off'] = y.start
                group.attrs['col_off'] = x.start
                group.attrs['height'] = y.stop - y.start
                group.attrs['width'] = x.stop - x.start

            else:

                out_filename = os.path.join(self.sub_dir,
                                            '{BASE}_y{Y:09d}_x{X:09d}_h{H:09d}_w{W:09d}{EXT}'.format(BASE=self.f_base,
                                                                                                     Y=y.start,
                                                                                                     X=x.start,
                                                                                                     H=y.stop - y.start,
                                                                                                     W=x.stop - x.start,
                                                                                                     EXT=self.f_ext))

                if self.overwrite:

                    if os.path.isfile(out_filename):
                        os.remove(out_filename)

                io_mode = 'w'

                # Every block starts at (0, 0) for the output
                w = Window(col_off=0,
                           row_off=0,
                           width=x.stop - x.start,
                           height=y.stop - y.start)

                # xres, 0, minx, 0, yres, maxy
                # TODO: hardcoded driver type
                kwargs = dict(driver=self.kwargs['driver'],
                              width=w.width,
                              height=w.height,
                              count=self.kwargs['count'],
                              dtype=self.kwargs['dtype'],
                              nodata=self.kwargs['nodata'],
                              blockxsize=self.kwargs['blockxsize'],
                              blockysize=self.kwargs['blockysize'],
                              crs=self.kwargs['crs'],
                              transform=Affine(self.kwargs['transform'][0],
                                               0.0,
                                               self.kwargs['transform'][2] + (x.start * self.kwargs['transform'][0]),
                                               0.0,
                                               self.kwargs['transform'][4],
                                               self.kwargs['transform'][5] - (y.start * -self.kwargs['transform'][4])),
                              compress=self.kwargs['compress'] if 'compress' in self.kwargs else 'none',
                              tiled=True)

        else:

            out_filename = self.filename

            w = Window(col_off=x.start,
                       row_off=y.start,
                       width=x.stop - x.start,
                       height=y.stop - y.start)

            io_mode = 'r+'
            kwargs = {}

        if not self.separate or (self.separate and self.out_block_type.lower() == 'gtiff'):

            with rio.open(out_filename,
                          mode=io_mode,
                          sharing=False,
                          **kwargs) as dst_:

                dst_.write(item,
                           window=w,
                           indexes=indexes)

    def __enter__(self):

        if self.separate:

            if self.out_block_type.lower() == 'zarr':
                self.root = zarr.open(self.zarr_file, mode='w')

        else:

            if 'compress' in self.kwargs:
                logger.warning('\nCannot write concurrently to a compressed raster when using a combination of processes and threads.\nTherefore, compression will be applied after the initial write.')
                del self.kwargs['compress']

            # An alternative here is to leave the writeable object open as self.
            # However, this does not seem to work when used within a Dask
            #   client environment because the `self.dst_` object cannot be pickled.

            # Create the output file
            with rio.open(self.filename, mode='w', **self.kwargs) as dst_:
                pass

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def check_crs(crs):

    """
    Checks a CRS instance

    Args:
        crs (``CRS`` | int | dict | str): The CRS instance.

    Returns:
        ``rasterio.crs.CRS``
    """

    if isinstance(crs, CRS):
        dst_crs = crs
    elif isinstance(crs, int):
        dst_crs = CRS.from_epsg(crs)
    elif isinstance(crs, dict):
        dst_crs = CRS.from_dict(crs)
    elif isinstance(crs, str):

        if crs.startswith('+proj'):
            dst_crs = CRS.from_proj4(crs)
        else:
            dst_crs = CRS.from_string(crs)

    else:
        logger.exception('  The CRS was not understood.')
        raise TypeError

    return dst_crs


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
                    bounds_by='intersection',
                    crs=None,
                    res=None,
                    return_bounds=False):

    """
    Gets the union of all files

    Args:
        filenames (list): The file names to mosaic.
        bounds_by (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union'].
        crs (Optional[crs]): The CRS to warp to.
        res (Optional[tuple]): The cell resolution to warp to.
        return_bounds (Optional[bool]): Whether to return the bounds tuple.

    Returns:
        transform, width, height
    """

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

    if bounds_by.lower() in ['union', 'intersection']:

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
            if bounds_by.lower() == 'union':

                bounds_left = min(bounds_left, left)
                bounds_bottom = min(bounds_bottom, bottom)
                bounds_right = max(bounds_right, right)
                bounds_top = max(bounds_top, top)

            elif bounds_by.lower() == 'intersection':

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

    else:

        bounds_width = int((bounds_right - bounds_left) / abs(res[0]))
        bounds_height = int((bounds_top - bounds_bottom) / abs(res[1]))

        bounds_transform = from_bounds(bounds_left,
                                       bounds_bottom,
                                       bounds_right,
                                       bounds_top,
                                       bounds_width,
                                       bounds_height)

    if return_bounds:
        return array_bounds(bounds_height, bounds_width, bounds_transform)
    else:
        return bounds_transform, bounds_width, bounds_height


def warp_images(filenames,
                bounds=None,
                crs=None,
                res=None,
                nodata=0,
                resampling='nearest',
                warp_mem_limit=512,
                num_threads=1):

    """
    Transforms a list of images to a common grid

    Args:
        filenames (list): The file names to mosaic.
        bounds (Optional[tuple]): The extent bounds to warp to. If not give, the union of all images is used.
        crs (Optional[object]): The CRS to warp to.
        res (Optional[tuple]): The cell resolution to warp to.
        nodata (Optional[int or float]): The 'no data' value.
        resampling (Optional[str]): The resampling method. Choices are ['average', 'bilinear', 'cubic',
            'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.

    Returns:
        ``list`` of ``rasterio.vrt.WarpedVRT`` objects
    """

    if resampling not in ['average', 'bilinear', 'cubic', 'cubic_spline',
                          'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest']:

        logger.warning("  The resampling method is not supported by rasterio. Setting to 'nearest'")

        resampling = 'nearest'

    warp_kwargs = {'resampling': resampling,
                   'crs': crs,
                   'res': res,
                   'nodata': nodata,
                   'warp_mem_limit': warp_mem_limit,
                   'num_threads': num_threads}

    if bounds:
        warp_kwargs['bounds'] = bounds
    else:

        # Get the union bounds of all images.
        #   *Target-aligned-pixels are returned.
        warp_kwargs['bounds'] = get_file_bounds(filenames,
                                                bounds_by='union',
                                                crs=crs,
                                                res=res,
                                                return_bounds=True)

    return [warp(fn, **warp_kwargs) for fn in filenames]

    # vrt_options = {'resampling': getattr(Resampling, resampling),
    #                'crs': crs,
    #                'transform': dst_transform,
    #                'height': dst_height,
    #                'width': dst_width,
    #                'nodata': nodata,
    #                'warp_mem_limit': warp_mem_limit,
    #                'warp_extras': {'multi': True,
    #                                'warp_option': 'NUM_THREADS={:d}'.format(num_threads)}}
    #
    # vrt_list = list()
    #
    # # Warp each image to a common grid
    # for fn in filenames:
    #
    #     with rio.open(fn) as src:
    #
    #         with WarpedVRT(src, **vrt_options) as vrt:
    #             vrt_list.append(vrt)
    #
    # return vrt_list


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


def warp(filename,
         resampling='nearest',
         bounds=None,
         crs=None,
         res=None,
         nodata=0,
         warp_mem_limit=512,
         num_threads=1,
         tap=False,
         tac=None):

    """
    Warps an image to a VRT object

    Args:
        filename (str): The input file name.
        resampling (Optional[str]): The resampling method. Choices are ['average', 'bilinear', 'cubic',
            'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        bounds (Optional[tuple]): The extent bounds to warp to.
        crs (Optional[``CRS`` | int | dict | str]): The CRS to warp to.
        res (Optional[tuple]): The cell resolution to warp to.
        nodata (Optional[int or float]): The 'no data' value.
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        tap (Optional[bool]): Whether to target align pixels.
        tac (Optional[tuple]): Target aligned raster coordinates (x, y).

    Returns:
        ``rasterio.vrt.WarpedVRT``
    """

    with rio.open(filename) as src:

        if res:
            dst_res = res
        else:
            dst_res = src.res

        if crs:
            dst_crs = check_crs(crs)
        else:
            dst_crs = src.crs

        # Check if the data need to be subset
        if bounds and (bounds != src.bounds):

            if isinstance(bounds, str):

                if bounds.startswith('BoundingBox'):

                    bounds_str = bounds.replace('BoundingBox(', '').split(',')

                    for str_ in bounds_str:

                        if str_.strip().startswith('left='):
                            left_coord = float(str_.strip().split('=')[1].replace(')', ''))
                        elif str_.strip().startswith('bottom='):
                            bottom_coord = float(str_.strip().split('=')[1].replace(')', ''))
                        elif str_.strip().startswith('right='):
                            right_coord = float(str_.strip().split('=')[1].replace(')', ''))
                        elif str_.strip().startswith('top='):
                            top_coord = float(str_.strip().split('=')[1].replace(')', ''))

                    dst_bounds = BoundingBox(left=left_coord,
                                             bottom=bottom_coord,
                                             right=right_coord,
                                             top=top_coord)

                else:
                    logger.exception('  The bounds were not accepted.')

            else:

                dst_bounds = BoundingBox(left=bounds[0],
                                         bottom=bounds[1],
                                         right=bounds[2],
                                         top=bounds[3])

        else:
            dst_bounds = src.bounds

        dst_width = int((dst_bounds.right - dst_bounds.left) / dst_res[0])
        dst_height = int((dst_bounds.top - dst_bounds.bottom) / dst_res[1])

        # Do not warp if all the key metadata match the reference information
        if (src.bounds == bounds) and \
                (src.res == dst_res) and \
                (src.crs == dst_crs) and \
                (src.width == dst_width) and \
                (src.height == dst_height):

            output = filename

        else:

            dst_transform = Affine(dst_res[0], 0.0, dst_bounds.left, 0.0, -dst_res[1], dst_bounds.top)

            if tac:

                # Align the cells to target coordinates
                import ipdb
                ipdb.set_trace()

            if tap:

                # Align the cells to the resolution
                dst_transform, dst_width, dst_height = aligned_target(dst_transform,
                                                                      dst_width,
                                                                      dst_height,
                                                                      dst_res)

            vrt_options = {'resampling': getattr(Resampling, resampling),
                           'crs': dst_crs,
                           'transform': dst_transform,
                           'height': dst_height,
                           'width': dst_width,
                           'nodata': nodata,
                           'warp_mem_limit': warp_mem_limit,
                           'warp_extras': {'multi': True,
                                           'warp_option': 'NUM_THREADS={:d}'.format(num_threads)}}

            with WarpedVRT(src, **vrt_options) as vrt:
                output = vrt

    return output


def transform_crs(data_src,
                  dst_crs,
                  dst_res=None,
                  dst_width=None,
                  dst_height=None,
                  dst_bounds=None,
                  resampling='nearest',
                  warp_mem_limit=512,
                  num_threads=1):

    """
    Transforms a DataArray to a new coordinate reference system

    Args:
        data_src (DataArray): The data to transform.
        dst_crs (CRS | int | dict | str): The destination CRS.
        dst_res (Optional[tuple]): The destination resolution.
        dst_width (Optional[int]): The destination width. Cannot be used with ``dst_res``.
        dst_height (Optional[int]): The destination height. Cannot be used with ``dst_res``.
        dst_bounds (Optional[BoundingBox | tuple]): The destination bounds, as a ``rasterio.coords.BoundingBox``
            or as a tuple of (left, bottom, right, top).
        resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        warp_mem_limit (Optional[int]): The warp memory limit.
        num_threads (Optional[int]): The number of parallel threads.

    Returns:
        ``numpy.ndarray`` ``tuple`` ``CRS``
    """

    dst_crs = check_crs(dst_crs)

    if not dst_bounds:
        dst_bounds = data_src.gw.bounds

    if not isinstance(dst_bounds, BoundingBox):

        dst_bounds = BoundingBox(left=dst_bounds[0],
                                 bottom=dst_bounds[1],
                                 right=dst_bounds[2],
                                 top=dst_bounds[3])

    if not dst_res and not dst_width:

        # Transform to the same dimensions as the input
        dst_transform, dst_width, dst_height = calculate_default_transform(data_src.crs,
                                                                           dst_crs,
                                                                           data_src.gw.ncols,
                                                                           data_src.gw.nrows,
                                                                           left=dst_bounds.left,
                                                                           bottom=dst_bounds.bottom,
                                                                           right=dst_bounds.right,
                                                                           top=dst_bounds.top,
                                                                           dst_width=data_src.gw.ncols,
                                                                           dst_height=data_src.gw.nrows)

    elif dst_res:

        # Transform by cell resolution
        dst_transform, dst_width, dst_height = calculate_default_transform(data_src.crs,
                                                                           dst_crs,
                                                                           data_src.gw.ncols,
                                                                           data_src.gw.nrows,
                                                                           left=dst_bounds.left,
                                                                           bottom=dst_bounds.bottom,
                                                                           right=dst_bounds.right,
                                                                           top=dst_bounds.top,
                                                                           resolution=dst_res)

    else:

        # Transform by destination dimensions
        dst_transform, dst_width, dst_height = calculate_default_transform(data_src.crs,
                                                                           dst_crs,
                                                                           data_src.gw.ncols,
                                                                           data_src.gw.nrows,
                                                                           left=dst_bounds.left,
                                                                           bottom=dst_bounds.bottom,
                                                                           right=dst_bounds.right,
                                                                           top=dst_bounds.top,
                                                                           dst_width=dst_width,
                                                                           dst_height=dst_height)

    # vrt_options = {'resampling': getattr(Resampling, resampling),
    #                'crs': dst_crs,
    #                'transform': dst_transform,
    #                'height': dst_height,
    #                'width': dst_width,
    #                'nodata': nodata,
    #                'warp_mem_limit': warp_mem_limit,
    #                'warp_extras': {'multi': True,
    #                                'warp_option': 'NUM_THREADS={:d}'.format(num_threads)}}

    destination = np.zeros((data_src.gw.nbands,
                            dst_height,
                            dst_width), dtype=data_src.dtype)

    data_dst, dst_transform = reproject(data_src.data.compute(),
                                        destination,
                                        src_transform=data_src.transform,
                                        src_crs=data_src.crs,
                                        dst_transform=dst_transform,
                                        dst_crs=dst_crs,
                                        resampling=getattr(Resampling, resampling),
                                        dst_resolution=dst_res,
                                        warp_mem_limit=warp_mem_limit,
                                        num_threads=num_threads)

    return data_dst, dst_transform, dst_crs

    # with rio.open(data_src.attrs['filename']) as src_:
    #     with WarpedVRT(src_, **vrt_options) as vrt_:
    #         return vrt_
