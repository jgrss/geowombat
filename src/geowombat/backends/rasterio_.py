import logging
import threading
import typing as T
import warnings
from collections import namedtuple
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import rasterio as rio
import xarray as xr
from affine import Affine
from dask.delayed import Delayed
from pyproj import CRS
from pyproj.exceptions import CRSError
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import (
    aligned_target,
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from rasterio.windows import Window

import geowombat as gw

try:
    import numcodecs
    import zarr

    ZARR_INSTALLED = True
except ImportError:
    ZARR_INSTALLED = False


logger = logging.getLogger(__name__)


def get_dims_from_bounds(
    bounds: BoundingBox, res: T.Tuple[float, float]
) -> T.Tuple[int, int]:
    width = int((bounds.right - bounds.left) / abs(res[0]))
    height = int((bounds.top - bounds.bottom) / abs(res[1]))

    return height, width


def get_file_info(
    src_obj: T.Union[rio.io.DatasetReader, rio.io.DatasetWriter]
) -> namedtuple:
    src_bounds = src_obj.bounds
    src_res = src_obj.res
    src_width = src_obj.width
    src_height = src_obj.height

    FileInfo = namedtuple(
        'FileInfo', 'src_bounds src_res src_width src_height'
    )

    return FileInfo(
        src_bounds=src_bounds,
        src_res=src_res,
        src_width=src_width,
        src_height=src_height,
    )


def to_gtiff(
    filename,
    data,
    window,
    indexes,
    transform,
    n_workers,
    separate,
    tags,
    kwargs,
):
    """Writes data to a GeoTiff file.

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

        group_name = 'y{Y:09d}_x{X:09d}_h{H:09d}_w{W:09d}.tif'.format(
            Y=window.row_off, X=window.col_off, H=window.height, W=window.width
        )

        group_path = str(pout / group_name)

        kwargs_copy = kwargs.copy()

        kwargs_copy['width'] = window.width
        kwargs_copy['height'] = window.height
        kwargs_copy['transform'] = Affine(*transform)

        group_window = Window(
            col_off=0, row_off=0, width=window.width, height=window.height
        )

        for item in [
            'with_config',
            'ignore_warnings',
            'sensor',
            'scale_factor',
            'ref_image',
            'ref_bounds',
            'ref_crs',
            'ref_res',
            'ref_tar',
            'l57_angles_path',
            'l8_angles_path',
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
        with rio.open(group_path, mode='r+') as dst_:
            dst_.write(data, window=group_window, indexes=indexes)

    else:
        with threading.Lock():
            with rio.open(group_path, mode='r+', sharing=False) as dst_:
                dst_.write(data, window=group_window, indexes=indexes)


class RasterioStore(object):
    """``Rasterio`` wrapper to allow ``dask.array.store`` to save chunks as
    windows.

    Reference:
        Code modified from https://github.com/dymaxionlabs/dask-rasterio
    """

    def __init__(
        self,
        filename: T.Union[str, Path],
        mode: str = 'w',
        tags: dict = None,
        **kwargs,
    ):
        self.filename = Path(filename)
        self.mode = mode
        self.tags = tags
        self.kwargs = kwargs
        self.dst = None

    def __setitem__(self, key, item):
        if len(key) == 3:
            index_range, y, x = key
            indexes = list(
                range(
                    index_range.start + 1,
                    index_range.stop + 1,
                    index_range.step or 1,
                )
            )
        else:
            indexes = 1
            y, x = key

        w = Window(
            col_off=x.start,
            row_off=y.start,
            width=x.stop - x.start,
            height=y.stop - y.start,
        )

        self.dst.write(item, window=w, indexes=indexes)

    def __enter__(self) -> 'RasterioStore':
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> 'RasterioStore':
        self.dst = self.rio_open()
        self.update_tags()

        return self

    def rio_open(self):
        return rio.open(self.filename, mode=self.mode, **self.kwargs)

    def update_tags(self):
        if self.tags is not None:
            self.dst.update_tags(**self.tags)

    def write_delayed(self, data: xr.DataArray):
        store = da.store(
            data.transpose('band', 'y', 'x').squeeze().data,
            self,
            lock=True,
            compute=False,
        )

        return self.close_delayed(store)

    @dask.delayed
    def close_delayed(self, store):
        return self.close()

    def write(self, data: xr.DataArray, compute: bool = False) -> Delayed:
        if isinstance(data.data, da.Array):
            return da.store(data.data, self, lock=True, compute=compute)
        else:
            self.dst.write(
                data.data,
                indexes=list(range(1, data.data.shape[0] + 1)),
            )

    def close(self):
        self.dst.close()


def check_res(
    res: T.Union[
        T.Tuple[T.Union[float, int], T.Union[float, int]],
        T.Sequence[T.Union[float, int]],
        float,
        int,
    ]
) -> T.Tuple[float, float]:
    """Checks a resolution.

    Args:
        res (int | float | tuple): The resolution.

    Returns:
        ``tuple``
    """
    if isinstance(res, (list, tuple)):
        dst_res = (float(res[0]), float(res[1]))
    elif isinstance(res, (float, int)):
        dst_res = (float(res), float(res))
    else:
        logger.exception(
            '  The resolution should be given as an integer, float, or tuple.'
        )
        raise TypeError

    return dst_res


def check_src_crs(
    src: T.Union[rio.io.DatasetReader, rio.io.DatasetWriter]
) -> rio.crs.CRS:
    """Checks a rasterio open() instance.

    Args:
        src (object): An instance of ``rasterio.io.DatasetReader`` or ``rasterio.io.DatasetWriter``.

    Returns:
        ``rasterio.crs.CRS``
    """
    return src.crs if src.crs else src.gcps[1]


def check_crs(crs: T.Union[CRS, rio.CRS, dict, int, np.number, str]) -> CRS:
    """Checks a CRS instance.

    Args:
        crs (``CRS`` | int | dict | str): The CRS instance.

    Returns:
        ``pyproj.CRS``
    """
    with rio.Env():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', (FutureWarning, UserWarning))
            if isinstance(crs, (CRS, rio.CRS)):
                dst_crs = CRS.from_wkt(crs.to_wkt())
            elif isinstance(crs, (int, np.number)):
                try:
                    dst_crs = CRS.from_epsg(crs)
                except CRSError:
                    try:
                        dst_crs = CRS.from_user_input(f"epsg:{crs}")
                    except CRSError as e:
                        raise ValueError(e)
            elif isinstance(crs, dict):
                dst_crs = CRS.from_dict(crs)
            elif isinstance(crs, str):
                if crs.startswith('+proj'):
                    dst_crs = CRS.from_proj4(crs)
                else:
                    crs = crs.replace('+init=', '')
                    try:
                        dst_crs = CRS.from_user_input(crs)
                    except CRSError:
                        try:
                            dst_crs = CRS.from_string(crs)
                        except CRSError as e:
                            raise ValueError(e)

            else:
                logger.exception('  The CRS was not understood.')
                raise TypeError

    return dst_crs


def check_file_crs(filename: T.Union[str, Path]) -> CRS:
    """Checks a file CRS.

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

    return check_crs(src_crs)


def unpack_bounding_box(bounds: str) -> T.Tuple[float, float, float, float]:
    """Unpacks a BoundBox() string.

    Args:
        bounds (str)

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


def unpack_window(bounds: str) -> Window:
    """Unpacks a Window() string.

    Args:
        bounds (str)

    Returns:
        ``rasterio.windows.Window``
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


def window_to_bounds(
    filenames: T.Union[str, Path, T.Sequence[T.Union[str, Path]]], w: Window
) -> T.Tuple[float, float, float, float]:
    """Transforms a rasterio Window() object to image bounds.

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


def align_bounds(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    res: T.Union[T.Tuple[float, float], T.Sequence[float], float, int],
) -> T.Tuple[Affine, int, int]:
    """Aligns bounds to resolution.

    Args:
        minx (float)
        miny (float)
        maxx (float)
        maxy (float)
        res (tuple | float | int)

    Returns:
        ``affine.Affine``, ``int``, ``int``
    """
    if isinstance(res, (int, float)):
        res = (float(res), float(res))

    try:
        xres, yres = res
    except Exception as e:
        raise TypeError(e)

    new_height = int(np.floor((maxy - miny) / yres))
    new_width = int(np.floor((maxx - minx) / xres))
    new_transform = Affine(xres, 0.0, minx, 0.0, -yres, maxy)

    return aligned_target(new_transform, new_width, new_height, res)


def get_file_bounds(
    filenames: T.Sequence[T.Union[str, Path]],
    bounds_by: str = 'union',
    crs: T.Optional[T.Any] = None,
    res: T.Optional[T.Union[T.Tuple[float, float], float, int]] = None,
    return_bounds: T.Optional[bool] = False,
):
    """Gets the union of all files.

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
            (
                bounds_left,
                bounds_bottom,
                bounds_right,
                bounds_top,
            ) = transform_bounds(
                src_crs,
                dst_crs,
                src_info.src_bounds.left,
                src_info.src_bounds.bottom,
                src_info.src_bounds.right,
                src_info.src_bounds.top,
                densify_pts=21,
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
                        densify_pts=21,
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
                bounds_left, bounds_bottom, bounds_right, bounds_top, dst_res
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
                bounds_height,
            )

        if return_bounds:
            return array_bounds(bounds_height, bounds_width, bounds_transform)
        else:
            return bounds_transform, bounds_width, bounds_height


def warp_images(
    filenames: T.Sequence[T.Union[str, Path]],
    bounds_by: str = 'union',
    bounds: T.Optional[T.Sequence[float]] = None,
    crs: T.Optional[T.Union[CRS, dict, int, str]] = None,
    res: T.Optional[T.Tuple[float, float]] = None,
    nodata: T.Union[float, int] = 0,
    resampling: str = 'nearest',
    warp_mem_limit: int = 512,
    num_threads: int = 1,
    tac: T.Optional[T.Tuple[np.ndarray, np.ndarray]] = None,
) -> T.List[xr.DataArray]:
    """Transforms a list of images to a common grid.

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
    if resampling not in [
        rmethod for rmethod in dir(Resampling) if not rmethod.startswith('__')
    ]:
        logger.warning(
            "  The resampling method is not supported by rasterio. Setting to 'nearest'"
        )
        resampling = 'nearest'

    warp_kwargs = {
        'resampling': resampling,
        'crs': crs,
        'res': res,
        'nodata': nodata,
        'warp_mem_limit': warp_mem_limit,
        'num_threads': num_threads,
        'tac': tac,
    }

    if bounds is not None:
        warp_kwargs['bounds'] = bounds
    else:

        # Get the union bounds of all images.
        #   *Target-aligned-pixels are returned.
        warp_kwargs['bounds'] = get_file_bounds(
            filenames,
            bounds_by=bounds_by,
            crs=crs,
            res=res,
            return_bounds=True,
        )

    return [warp(fn, **warp_kwargs) for fn in filenames]


def get_ref_image_meta(filename):
    """Gets warping information from a reference image.

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


def warp(
    filename: T.Union[str, Path],
    resampling: str = 'nearest',
    bounds: T.Optional[T.Sequence[float]] = None,
    crs: T.Optional[T.Union[CRS, dict, int, str]] = None,
    res: T.Optional[T.Tuple[float, float]] = None,
    nodata: T.Union[float, int] = 0,
    warp_mem_limit: int = 512,
    num_threads: int = 1,
    tap: bool = False,
    tac: T.Optional[T.Tuple[np.ndarray, np.ndarray]] = None,
) -> WarpedVRT:
    """Warps an image to a VRT object.

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

        if res is not None:
            dst_res = check_res(res)
        else:
            dst_res = src_info.src_res

        # Check if the data need to be subset
        if (bounds is None) or (tuple(bounds) == tuple(src_info.src_bounds)):
            if crs:
                (
                    left_coord,
                    bottom_coord,
                    right_coord,
                    top_coord,
                ) = transform_bounds(
                    src_crs,
                    dst_crs,
                    src_info.src_bounds.left,
                    src_info.src_bounds.bottom,
                    src_info.src_bounds.right,
                    src_info.src_bounds.top,
                    densify_pts=21,
                )

                dst_bounds = BoundingBox(
                    left=left_coord,
                    bottom=bottom_coord,
                    right=right_coord,
                    top=top_coord,
                )

            else:
                dst_bounds = src_info.src_bounds

        else:
            # Ensure that the user bounds object is a ``BoundingBox``
            if isinstance(bounds, BoundingBox):
                dst_bounds = bounds
            elif isinstance(bounds, str):

                if bounds.startswith('BoundingBox'):
                    (
                        left_coord,
                        bottom_coord,
                        right_coord,
                        top_coord,
                    ) = unpack_bounding_box(bounds)
                else:
                    logger.exception('  The bounds were not accepted.')
                    raise TypeError

                dst_bounds = BoundingBox(
                    left=left_coord,
                    bottom=bottom_coord,
                    right=right_coord,
                    top=top_coord,
                )

            elif isinstance(bounds, (list, np.ndarray, tuple)):
                dst_bounds = BoundingBox(
                    left=bounds[0],
                    bottom=bounds[1],
                    right=bounds[2],
                    top=bounds[3],
                )

            else:
                logger.exception(
                    f'  The bounds type was not understood. Bounds should be given as a '
                    f'rasterio.coords.BoundingBox, tuple, or ndarray, not a {type(bounds)}.'
                )
                raise TypeError

        dst_height, dst_width = get_dims_from_bounds(dst_bounds, dst_res)

        # Do all the key metadata match the reference information?
        if (
            (tuple(src_info.src_bounds) == tuple(bounds))
            and (src_info.src_res == dst_res)
            and (src_crs == dst_crs)
            and (src_info.src_width == dst_width)
            and (src_info.src_height == dst_height)
            and ('.nc' not in filename.lower())
        ):
            vrt_options = {
                'resampling': getattr(Resampling, resampling),
                'src_crs': src_crs,
                'crs': dst_crs,
                'src_transform': src.transform,
                'transform': src.transform,
                'height': dst_height,
                'width': dst_width,
                'nodata': nodata,
                'warp_mem_limit': warp_mem_limit,
                'warp_extras': {
                    'multi': True,
                    'warp_option': f'NUM_THREADS={num_threads}',
                },
            }

        else:
            src_transform = Affine(
                src_info.src_res[0],
                0.0,
                src_info.src_bounds.left,
                0.0,
                -src_info.src_res[1],
                src_info.src_bounds.top,
            )
            dst_transform = Affine(
                dst_res[0],
                0.0,
                dst_bounds.left,
                0.0,
                -dst_res[1],
                dst_bounds.top,
            )

            if tac is not None:
                # Align the cells to target coordinates
                tap_left = tac[0][np.abs(tac[0] - dst_bounds.left).argmin()]
                tap_top = tac[1][np.abs(tac[1] - dst_bounds.top).argmin()]

                dst_transform = Affine(
                    dst_res[0], 0.0, tap_left, 0.0, -dst_res[1], tap_top
                )

            if tap:
                # Align the cells to the resolution
                dst_transform, dst_width, dst_height = aligned_target(
                    dst_transform, dst_width, dst_height, dst_res
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
                    'warp_option': f'NUM_THREADS={num_threads}',
                },
            }

        output = WarpedVRT(src, **vrt_options)

    return output


def reproject_array(
    data: xr.DataArray,
    dst_height: int,
    dst_width: int,
    dst_transform: Affine,
    dst_crs: T.Any,
    src_nodata: T.Union[float, int],
    dst_nodata: T.Union[float, int],
    resampling: Resampling,
    dst_res: tuple,
    warp_mem_limit: int,
    num_threads: int,
) -> np.ndarray:
    """Reprojects a DataArray and translates to a numpy ndarray."""
    dst_array = np.zeros(
        (data.gw.nbands, dst_height, dst_width), dtype=data.dtype
    )
    dst_array, dst_transform = reproject(
        data.gw.compute(num_workers=num_threads),
        dst_array,
        src_transform=data.gw.transform,
        src_crs=data.gw.crs_to_pyproj,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        resampling=getattr(Resampling, resampling),
        dst_resolution=dst_res,
        warp_mem_limit=warp_mem_limit,
        num_threads=num_threads,
    )

    return dst_array


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
    num_threads=1,
    return_as_dict=False,
    delayed_array=False,
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
        return_as_dict (Optional[bool]): Whether to return data as a dictionary. Otherwise, return as a ``tuple``.
        delayed_array (Optional[bool]): Whether to return the transformed array as a delayed object.

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

    if isinstance(dst_res, (int, float, tuple)):
        dst_res = check_res(dst_res)
        dst_width = None
        dst_height = None
    else:
        if not isinstance(dst_width, int) and not isinstance(dst_height, int):
            dst_res = data_src.res

    if dst_res is None:
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
        resolution=dst_res,
    )

    if not isinstance(dst_height, int):
        if data_src.res == dst_res:
            dst_height = data_src.gw.nrows
        else:
            dst_height = dst_height_
    if not isinstance(dst_width, int):
        if data_src.res == dst_res:
            dst_width = data_src.gw.ncols
        else:
            dst_width = dst_width_

    if coords_only:
        if isinstance(dst_width, int) and isinstance(dst_height, int):
            xs = (
                dst_transform
                * (
                    np.arange(0, dst_width) + 0.5,
                    np.arange(0, dst_width) + 0.5,
                )
            )[0]
            ys = (
                dst_transform
                * (
                    np.arange(0, dst_height) + 0.5,
                    np.arange(0, dst_height) + 0.5,
                )
            )[1]
        else:
            xs = (
                dst_transform
                * (
                    np.arange(0, dst_width_) + 0.5,
                    np.arange(0, dst_width_) + 0.5,
                )
            )[0]
            ys = (
                dst_transform
                * (
                    np.arange(0, dst_height_) + 0.5,
                    np.arange(0, dst_height_) + 0.5,
                )
            )[1]

        XYCoords = namedtuple('XYCoords', 'xs ys')
        xy_coords = XYCoords(xs=xs, ys=ys)

        if return_as_dict:
            return {
                'coords': xy_coords,
                'transform': dst_transform,
                'crs': dst_crs,
                'height': dst_height,
                'width': dst_width,
            }
        else:
            return xy_coords, dst_transform, dst_crs

    if not dst_bounds:
        dst_left = dst_transform[2]
        dst_top = dst_transform[5]
        dst_right = dst_left + (dst_width * dst_transform[0])
        dst_bottom = dst_top - (dst_height * dst_transform[4])

        dst_bounds = (dst_left, dst_bottom, dst_right, dst_top)

    if not isinstance(dst_bounds, BoundingBox):
        dst_bounds = BoundingBox(
            left=dst_bounds[0],
            bottom=dst_bounds[1],
            right=dst_bounds[2],
            top=dst_bounds[3],
        )

    if not dst_res:
        cellx = (dst_bounds.right - dst_bounds.left) / dst_width
        celly = (dst_bounds.top - dst_bounds.bottom) / dst_height
        dst_res = (cellx, celly)

    # Ensure the final transform is set based on adjusted bounds
    dst_transform = Affine(
        abs(dst_res[0]),
        0.0,
        dst_bounds.left,
        0.0,
        -abs(dst_res[1]),
        dst_bounds.top,
    )

    proj_func = (
        dask.delayed(reproject_array) if delayed_array else reproject_array
    )
    transformed_array = proj_func(
        data_src,
        dst_height,
        dst_width,
        dst_transform,
        dst_crs,
        src_nodata,
        dst_nodata,
        resampling,
        dst_res,
        warp_mem_limit,
        num_threads,
    )

    if return_as_dict:
        return {
            'array': transformed_array,
            'transform': dst_transform,
            'crs': dst_crs,
            'height': dst_height,
            'width': dst_width,
        }
    else:
        return transformed_array, dst_transform, dst_crs
