import os
import shutil
from pathlib import Path
from collections import namedtuple
import threading
import logging
import warnings
import typing as T
from contextlib import contextmanager

import affine

import geowombat as gw

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import aligned_target, calculate_default_transform, transform_bounds, reproject
from rasterio.transform import array_bounds, from_bounds
from rasterio.windows import Window
from rasterio.coords import BoundingBox
import dask.array as da
import xarray as xr
from pyproj import CRS
from affine import Affine


try:
    import zarr
    import numcodecs

    ZARR_INSTALLED = True
except:
    ZARR_INSTALLED = False


logger = logging.getLogger(__name__)


def get_file_info(src_obj):

    src_bounds = src_obj.bounds
    src_res = src_obj.res
    src_width = src_obj.width
    src_height = src_obj.height

    FileInfo = namedtuple('FileInfo', 'src_bounds src_res src_width src_height')

    return FileInfo(src_bounds=src_bounds,
                    src_res=src_res,
                    src_width=src_width,
                    src_height=src_height)


def to_gtiff(filename, data, window, indexes, transform, n_workers, separate, tags, kwargs):

    """
    Writes data to a GeoTiff file.

    Args:
        filename (str): The output file name. The file must already exist.
        data (ndarray): The data to write.
        window (namedtuple): A ``rasterio.window.Window`` object.
        indexes (int | 1d array-like): The output ``data`` indices.
        transform (tuple): The original raster transform.
        n_workers (int): The number of parallel workers being used.
        tags (Optional[dict]): Image tags to write to file.
        kwargs (Optional[dict]): Additional keyword arguments to pass to ``rasterio.write``.

    Returns:
        ``None``
    """

    p = Path(filename)

    # Strip the file ending
    f_base = p.name.split('.')[0]

    if separate:

        # Create a sub-directory
        pout = p.parent / f_base
        pout.mkdir(exist_ok=True, parents=True)

        group_name = 'y{Y:09d}_x{X:09d}_h{H:09d}_w{W:09d}.tif'.format(Y=window.row_off,
                                                                      X=window.col_off,
                                                                      H=window.height,
                                                                      W=window.width)

        group_path = str(pout / group_name)

        kwargs_copy = kwargs.copy()

        kwargs_copy['width'] = window.width
        kwargs_copy['height'] = window.height
        kwargs_copy['transform'] = Affine(*transform)

        group_window = Window(col_off=0,
                              row_off=0,
                              width=window.width,
                              height=window.height)

        for item in [
            'with_config', 'ignore_warnings', 'sensor', 'scale_factor', 'ref_image', 'ref_bounds', 'ref_crs',
            'ref_res', 'ref_tar', 'l57_angles_path', 'l8_angles_path'
        ]:
            if item in kwargs_copy:
                del kwargs_copy[item]

        with rio.open(group_path, mode='w', **kwargs_copy) as dst:

            if tags:
                dst.update_tags(**tags)

    else:

        group_path = str(filename)
        group_window = window

    if separate or (n_workers == 1):

        with rio.open(group_path,
                      mode='r+') as dst_:

            dst_.write(data,
                       window=group_window,
                       indexes=indexes)

    else:

        with threading.Lock():

            with rio.open(group_path,
                          mode='r+',
                          sharing=False) as dst_:

                dst_.write(data,
                           window=group_window,
                           indexes=indexes)


class RasterioStore(object):
    def __init__(
        self,
        filename: T.Union[str, Path],
        data: xr.DataArray,
        tags: dict = None,
        **kwargs
    ):
        self.filename = filename
        self.data = data
        self.tags = tags
        self.kwargs = kwargs
        self.dst = None

    def __setitem__(self, key, item):
        if len(key) == 3:
            index_range, y, x = key
            indexes = list(range(index_range.start+1, index_range.stop+1, index_range.step or 1))
        else:
            indexes = 1
            y, x = key

        w = Window(
            col_off=x.start,
            row_off=y.start,
            width=x.stop - x.start,
            height=y.stop - y.start
        )

        self.dst.write(
            item,
            window=w,
            indexes=indexes
        )

    def __enter__(self) -> 'RasterioStore':
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> 'RasterioStore':
        self.dst = rio.open(self.filename, mode='w', **self.kwargs)
        if self.tags is not None:
            self.dst.update_tags(**self.tags)

        return self

    def write(self):
        return da.store(
            da.squeeze(self.data.data),
            self,
            lock=True,
            compute=False
        )

    def close(self):
        self.dst.close()


class WriteDaskArray(object):
    """``Rasterio`` wrapper to allow ``dask.array.store`` to save chunks as windows.

    Args:
        filename (str): The file to write to.
        overwrite (Optional[bool]): Whether to overwrite an existing output file.
        separate (Optional[bool]): Whether to write blocks as separate files. Otherwise, write to the same file.
        out_block_type (Optional[str]): The output block type. Choices are ['gtiff', 'zarr'].
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
    def __init__(
        self,
        filename: T.Union[str, Path],
        overwrite: bool = False,
        separate: bool = False,
        out_block_type: str = 'gtiff',
        keep_blocks: bool = False,
        gdal_cache: int = 512,
        **kwargs
    ):
        if out_block_type == 'zarr':
            if not ZARR_INSTALLED:
                logger.exception('Zarr and numcodecs must be installed.')

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

            # Create the output file
            with rio.open(self.filename, mode='w', **self.kwargs) as dst_:
                pass

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def check_res(res):

    """
    Checks a resolution

    Args:
        res (int | float | tuple): The resolution.

    Returns:
        ``tuple``
    """

    if isinstance(res, tuple):
        dst_res = res
    elif isinstance(res, int) or isinstance(res, float):
        dst_res = (res, res)
    else:
        logger.exception('  The resolution should be given as an integer, float, or tuple.')
        raise TypeError

    return dst_res


def check_src_crs(src):
    """Checks a rasterio open() instance

    Args:
        src (object): A `rasterio.open` instance.

    Returns:
        ``rasterio.crs.CRS``
    """
    return src.crs if src.crs else src.gcps[1]


def check_crs(crs):
    """Checks a CRS instance

    Args:
        crs (``CRS`` | int | dict | str): The CRS instance.

    Returns:
        ``pyproj.CRS``
    """
    with rio.Env():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            if isinstance(crs, (CRS, rio.CRS)):
                dst_crs = CRS.from_proj4(crs.to_proj4())
            elif isinstance(crs, CRS):
                dst_crs = crs
            elif isinstance(crs, int):
                dst_crs = CRS.from_epsg(crs)
            elif isinstance(crs, dict):
                dst_crs = CRS.from_dict(crs)
            elif isinstance(crs, str):
                if crs.startswith('+proj'):
                    dst_crs = CRS.from_proj4(crs)
                else:
                    try:
                        dst_crs = CRS.from_user_input(crs.replace('+init=', ''))
                    except ValueError:
                        dst_crs = CRS.from_string(crs.replace('+init=', ''))

            else:
                logger.exception('  The CRS was not understood.')
                raise TypeError

    return dst_crs


def check_file_crs(filename):
    """Checks a file CRS

    Args:
        filename (Path | str): The file to open.

    Returns:
        ``object`` or ``string``
    """
    # rasterio does not read the CRS from a .nc file
    if '.nc' in str(filename).lower():
        # rasterio does not open and read metadata from NetCDF files
        if str(filename).lower().startswith('netcdf:'):
            with xr.open_dataset(filename.split(':')[1], chunks=256) as src:
                src_crs = src.crs
        else:
            with xr.open_dataset(filename, chunks=256) as src:
                src_crs = src.crs

    else:
        with rio.open(filename) as src:
            src_crs = check_src_crs(src)

    return src_crs


def unpack_bounding_box(bounds):
    """Unpacks a BoundBox() string

    Args:
        bounds (object)

    Returns:
        ``tuple``
    """

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

    return left_coord, bottom_coord, right_coord, top_coord


def unpack_window(bounds):
    """Unpacks a Window() string

    Args:
        bounds (object)

    Returns:
        ``object``
    """
    bounds_str = bounds.replace('Window(', '').split(',')

    for str_ in bounds_str:

        if str_.strip().startswith('col_off='):
            col_off = int(str_.strip().split('=')[1].replace(')', ''))
        elif str_.strip().startswith('row_off='):
            row_off = int(str_.strip().split('=')[1].replace(')', ''))
        elif str_.strip().startswith('height='):
            height = int(str_.strip().split('=')[1].replace(')', ''))
        elif str_.strip().startswith('width='):
            width = int(str_.strip().split('=')[1].replace(')', ''))

    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def window_to_bounds(filenames, w):
    """Transforms a rasterio Window() object to image bounds

    Args:
        filenames (str or str list)
        w (object)

    Returns:
        ``tuple``
    """
    if isinstance(filenames, str):
        src = rio.open(filenames)
    else:
        src = rio.open(filenames[0])

    left, top = src.transform * (w.col_off, w.row_off)

    right = left + w.width * abs(src.res[0])
    bottom = top - w.height * abs(src.res[1])

    src.close()

    return left, bottom, right, top


def align_bounds(minx, miny, maxx, maxy, res):
    """Aligns bounds to resolution

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


def get_file_bounds(
    filenames,
    bounds_by='union',
    crs=None,
    res=None,
    return_bounds=False
):
    """Gets the union of all files

    Args:
        filenames (list): The file names to mosaic.
        bounds_by (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union'].
        crs (Optional[crs]): The CRS to warp to.
        res (Optional[tuple]): The cell resolution to warp to.
        return_bounds (Optional[bool]): Whether to return the bounds tuple.

    Returns:
        transform, width, height
    """
    if filenames[0].lower().startswith('netcdf:'):

        with rio.open(filenames[0]) as src:
            file_bounds = src.bounds

        return file_bounds

    else:

        if crs is not None:
            dst_crs = check_crs(crs)
        else:
            dst_crs = check_file_crs(filenames[0])

        src_crs = check_file_crs(filenames[0])

        with rio.open(filenames[0]) as src:

            src_info = get_file_info(src)

            if res:
                dst_res = check_res(res)
            else:
                dst_res = src_info.src_res

            # Transform the extent to the reference CRS
            bounds_left, bounds_bottom, bounds_right, bounds_top = transform_bounds(
                src_crs,
                dst_crs,
                src_info.src_bounds.left,
                src_info.src_bounds.bottom,
                src_info.src_bounds.right,
                src_info.src_bounds.top,
                densify_pts=21
            )

        if bounds_by.lower() in ['union', 'intersection']:

            for fn in filenames[1:]:

                src_crs = check_file_crs(fn)

                with rio.open(fn) as src:

                    src_info = get_file_info(src)

                    # Transform the extent to the reference CRS
                    left, bottom, right, top = transform_bounds(
                        src_crs,
                        dst_crs,
                        src_info.src_bounds.left,
                        src_info.src_bounds.bottom,
                        src_info.src_bounds.right,
                        src_info.src_bounds.top,
                        densify_pts=21
                    )

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
            bounds_transform, bounds_width, bounds_height = align_bounds(
                bounds_left,
                bounds_bottom,
                bounds_right,
                bounds_top,
                dst_res
            )

        else:
            bounds_width = int((bounds_right - bounds_left) / abs(dst_res[0]))
            bounds_height = int((bounds_top - bounds_bottom) / abs(dst_res[1]))

            bounds_transform = from_bounds(
                bounds_left,
                bounds_bottom,
                 bounds_right,
                 bounds_top,
                 bounds_width,
                 bounds_height
            )

        if return_bounds:
            return array_bounds(bounds_height, bounds_width, bounds_transform)
        else:
            return bounds_transform, bounds_width, bounds_height


def warp_images(filenames,
                bounds_by='union',
                bounds=None,
                crs=None,
                res=None,
                nodata=0,
                resampling='nearest',
                warp_mem_limit=512,
                num_threads=1,
                tac=None):
    """Transforms a list of images to a common grid

    Args:
        filenames (list): The file names to mosaic.
        bounds_by (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union'].
        bounds (Optional[tuple]): The extent bounds to warp to. If not give, the union of all images is used.
        crs (Optional[object]): The CRS to warp to.
        res (Optional[tuple]): The cell resolution to warp to.
        nodata (Optional[int or float]): The 'no data' value.
        resampling (Optional[str]): The resampling method. Choices are ['average', 'bilinear', 'cubic',
            'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest', 'q1', 'q3'].
        warp_mem_limit (Optional[int]): The memory limit (in MB) for the ``rasterio.vrt.WarpedVRT`` function.
        num_threads (Optional[int]): The number of warp worker threads.
        tac (Optional[tuple]): Target aligned raster coordinates (x, y).

    Returns:
        ``list`` of ``rasterio.vrt.WarpedVRT`` objects
    """
    if resampling not in [rmethod for rmethod in dir(Resampling) if not rmethod.startswith('__')]:
        logger.warning("  The resampling method is not supported by rasterio. Setting to 'nearest'")
        resampling = 'nearest'

    warp_kwargs = {
        'resampling': resampling,
        'crs': crs,
        'res': res,
        'nodata': nodata,
        'warp_mem_limit': warp_mem_limit,
        'num_threads': num_threads,
        'tac': tac
    }

    if bounds:
        warp_kwargs['bounds'] = bounds
    else:

        # Get the union bounds of all images.
        #   *Target-aligned-pixels are returned.
        warp_kwargs['bounds'] = get_file_bounds(
            filenames,
            bounds_by=bounds_by,
            crs=crs,
            res=res,
            return_bounds=True
        )

    return [warp(fn, **warp_kwargs) for fn in filenames]


def get_ref_image_meta(filename):

    """
    Gets warping information from a reference image

    Args:
        filename (str): The file name to get information from.

    Returns:
        ``collections.namedtuple``
    """

    WarpInfo = namedtuple('WarpInfo', 'bounds crs res')

    src_crs = check_file_crs(filename)

    with rio.open(filename) as src:

        bounds = src.bounds
        res = src.res

    return WarpInfo(bounds=bounds, crs=src_crs, res=res)


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
    """Warps an image to a VRT object

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
    if crs is not None:
        dst_crs = check_crs(crs)
    else:
        dst_crs = check_file_crs(filename)

    src_crs = check_file_crs(filename)

    with rio.open(filename) as src:
        src_info = get_file_info(src)

        if res:
            dst_res = check_res(res)
        else:
            dst_res = src_info.src_res

        # Check if the data need to be subset
        if (bounds is None) or (tuple(bounds) == tuple(src_info.src_bounds)):

            if crs:

                left_coord, bottom_coord, right_coord, top_coord = transform_bounds(
                    src_crs,
                    dst_crs,
                    src_info.src_bounds.left,
                    src_info.src_bounds.bottom,
                    src_info.src_bounds.right,
                    src_info.src_bounds.top,
                    densify_pts=21
                )

                dst_bounds = BoundingBox(
                    left=left_coord,
                    bottom=bottom_coord,
                    right=right_coord,
                    top=top_coord
                )

            else:
                dst_bounds = src_info.src_bounds

        else:
            # Ensure that the user bounds object is a ``BoundingBox``
            if isinstance(bounds, BoundingBox):
                dst_bounds = bounds
            elif isinstance(bounds, str):

                if bounds.startswith('BoundingBox'):
                    left_coord, bottom_coord, right_coord, top_coord = unpack_bounding_box(bounds)
                else:
                    logger.exception('  The bounds were not accepted.')
                    raise TypeError

                dst_bounds = BoundingBox(
                    left=left_coord,
                    bottom=bottom_coord,
                    right=right_coord,
                    top=top_coord
                )

            elif isinstance(bounds, tuple) or isinstance(bounds, list) or isinstance(bounds, np.ndarray):
                dst_bounds = BoundingBox(
                    left=bounds[0],
                    bottom=bounds[1],
                    right=bounds[2],
                    top=bounds[3]
                )

            else:
                logger.exception(f'  The bounds type was not understood. Bounds should be given as a '
                                 f'rasterio.coords.BoundingBox, tuple, or ndarray, not a {type(bounds)}.')
                raise TypeError

        dst_width = int((dst_bounds.right - dst_bounds.left) / dst_res[0])
        dst_height = int((dst_bounds.top - dst_bounds.bottom) / dst_res[1])

        # Do not warp if all the key metadata match the reference information
        if (tuple(src_info.src_bounds) == tuple(bounds)) and \
                (src_info.src_res == dst_res) and \
                (src_crs == dst_crs) and \
                (src_info.src_width == dst_width) and \
                (src_info.src_height == dst_height) and \
                ('.nc' not in filename.lower()):

            output = filename

        else:

            src_transform = Affine(src_info.src_res[0], 0.0, src_info.src_bounds.left, 0.0, -src_info.src_res[1], src_info.src_bounds.top)
            dst_transform = Affine(dst_res[0], 0.0, dst_bounds.left, 0.0, -dst_res[1], dst_bounds.top)

            if tac:

                # Align the cells to target coordinates
                tap_left = tac[0][np.abs(tac[0] - dst_bounds.left).argmin()]
                tap_top = tac[1][np.abs(tac[1] - dst_bounds.top).argmin()]

                dst_transform = Affine(dst_res[0], 0.0, tap_left, 0.0, -dst_res[1], tap_top)

            if tap:

                # Align the cells to the resolution
                dst_transform, dst_width, dst_height = aligned_target(
                    dst_transform,
                    dst_width,
                    dst_height,
                    dst_res
                )

            vrt_options = {
                'resampling': getattr(Resampling, resampling),
                'src_crs': src_crs,
                 'crs': dst_crs,
                 'src_transform': src_transform,
                 'transform': dst_transform,
                 'height': dst_height,
                 'width': dst_width,
                 'nodata': nodata,
                 'warp_mem_limit': warp_mem_limit,
                 'warp_extras': {
                     'multi': True,
                     'warp_option': f'NUM_THREADS={num_threads}'
                 }
            }

            with WarpedVRT(src, **vrt_options) as vrt:
                output = vrt

    return output


def transform_crs(
    data_src,
    dst_crs=None,
    dst_res=None,
    dst_width=None,
    dst_height=None,
    dst_bounds=None,
    src_nodata=None,
    dst_nodata=None,
    coords_only=False,
    resampling='nearest',
    warp_mem_limit=512,
    num_threads=1
):
    """Transforms a DataArray to a new coordinate reference system.

    Args:
        data_src (DataArray): The data to transform.
        dst_crs (Optional[CRS | int | dict | str]): The destination CRS.
        dst_res (Optional[float | int | tuple]): The destination resolution.
        dst_width (Optional[int]): The destination width. Cannot be used with ``dst_res``.
        dst_height (Optional[int]): The destination height. Cannot be used with ``dst_res``.
        dst_bounds (Optional[BoundingBox | tuple]): The destination bounds, as a ``rasterio.coords.BoundingBox``
            or as a tuple of (left, bottom, right, top).
        src_nodata (Optional[int | float]): The source nodata value. Pixels with this value will not be used for
            interpolation. If not set, it will default to the nodata value of the source image if a masked ndarray
            or rasterio band, if available.
        dst_nodata (Optional[int | float]): The nodata value used to initialize the destination; it will remain in
            all areas not covered by the reprojected source. Defaults to the nodata value of the destination
            image (if set), the value of src_nodata, or 0 (GDAL default).
        coords_only (Optional[bool]): Whether to return transformed coordinates. If ``coords_only`` = ``True`` then
            the array is not warped and the size is unchanged. It also avoids in-memory computations.
        resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        warp_mem_limit (Optional[int]): The warp memory limit.
        num_threads (Optional[int]): The number of parallel threads.

    Returns:

        If ``coords_only`` = ``True``:
            ``tuple`` ``tuple`` ``CRS``

        If ``coords_only`` = ``False``:
            ``numpy.ndarray`` ``tuple`` ``CRS``
    """
    if dst_crs:
        dst_crs = check_crs(dst_crs)
    else:
        dst_crs = data_src.crs

    if isinstance(dst_res, int) or isinstance(dst_res, float) or isinstance(dst_res, tuple):
        dst_res = check_res(dst_res)

        dst_width = None
        dst_height = None

    else:
        if not isinstance(dst_width, int) and not isinstance(dst_height, int):
            dst_res = data_src.res

    if not dst_res:

        if not dst_width:
            dst_width = data_src.gw.ncols

        if not dst_height:
            dst_height = data_src.gw.nrows

    dst_transform, dst_width_, dst_height_ = calculate_default_transform(
        data_src.gw.crs_to_pyproj,
        dst_crs,
        data_src.gw.ncols,
        data_src.gw.nrows,
        left=data_src.gw.left,
        bottom=data_src.gw.bottom,
        right=data_src.gw.right,
        top=data_src.gw.top,
        dst_width=dst_width,
        dst_height=dst_height,
        resolution=dst_res
    )

    if coords_only:
        if isinstance(dst_width, int) and isinstance(dst_height, int):
            xs = (dst_transform * (np.arange(0, dst_width) + 0.5, np.arange(0, dst_width) + 0.5))[0]
            ys = (dst_transform * (np.arange(0, dst_height) + 0.5, np.arange(0, dst_height) + 0.5))[1]
        else:
            xs = (dst_transform * (np.arange(0, dst_width_) + 0.5, np.arange(0, dst_width_) + 0.5))[0]
            ys = (dst_transform * (np.arange(0, dst_height_) + 0.5, np.arange(0, dst_height_) + 0.5))[1]

        XYCoords = namedtuple('XYCoords', 'xs ys')
        xy_coords = XYCoords(xs=xs, ys=ys)

        return xy_coords, dst_transform, dst_crs

    if not isinstance(dst_width, int):
        dst_width = data_src.gw.ncols

    if not isinstance(dst_height, int):
        dst_height = data_src.gw.nrows

    if not dst_bounds:
        dst_left = dst_transform[2]
        dst_top = dst_transform[5]
        dst_right = dst_left + (dst_width*dst_transform[0])
        dst_bottom = dst_top - (dst_height*dst_transform[4])

        dst_bounds = (dst_left, dst_bottom, dst_right, dst_top)

    if not isinstance(dst_bounds, BoundingBox):
        dst_bounds = BoundingBox(
            left=dst_bounds[0],
            bottom=dst_bounds[1],
            right=dst_bounds[2],
            top=dst_bounds[3]
        )

    if not dst_res:
        cellx = (dst_bounds.right - dst_bounds.left) / dst_width
        celly = (dst_bounds.top - dst_bounds.bottom) / dst_height
        dst_res = (cellx, celly)

    # Ensure the final transform is set based on adjusted bounds
    dst_transform = affine.Affine(abs(dst_res[0]), 0.0, dst_bounds.left, 0.0, -abs(dst_res[1]), dst_bounds.top)

    transformed_array = []

    for band in range(0, data_src.gw.nbands):
        destination = np.zeros(
            (dst_height, dst_width), dtype=data_src.dtype
        )
        data_dst, dst_transform = reproject(
            data_src[band, :, :].data.compute(num_workers=num_threads),
            destination,
            src_transform=data_src.gw.transform,
            src_crs=data_src.gw.crs_to_pyproj,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
            resampling=getattr(Resampling, resampling),
            dst_resolution=dst_res,
            warp_mem_limit=warp_mem_limit,
            num_threads=num_threads
        )

        transformed_array.append(data_dst)

    return np.array(transformed_array), dst_transform, dst_crs
