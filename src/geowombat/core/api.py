# https://github.com/pydata/xarray/issues/2560
try:
    import netCDF4
except:
    pass

try:
    import h5netcdf
except:
    pass

import warnings
from pathlib import Path
import logging
import threading
from contextlib import contextmanager
import typing as T

import concurrent.futures

from . import geoxarray
from .series import BaseSeries, SeriesStats, TransferLib
from .util import Chunks, get_file_extension, parse_wildcard
from ..handler import add_handler
from ..config import config, _set_defaults
from ..backends import concat as gw_concat
from ..backends import mosaic as gw_mosaic
from ..backends import warp_open
from ..backends.rasterio_ import check_src_crs

import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.windows import from_bounds, Window
from rasterio.coords import BoundingBox
import dask
import dask.array as da
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
logger = add_handler(logger)
warnings.filterwarnings('ignore')

ch = Chunks()

IO_DICT = dict(
    rasterio=[
        '.tif',
        '.tiff',
        '.TIF',
        '.TIFF',
        '.img',
        '.IMG',
        '.kea',
        '.vrt',
        '.VRT',
        '.jp2',
        '.JP2',
        '.hgt',
        '.HGT',
        '.hdf',
        '.HDF',
        '.h5',
        '.H5'
    ],
    xarray=['.nc']
)


@contextmanager
def _tqdm(*args, **kwargs):
    yield None


def get_attrs(src, **kwargs):
    cellxh = src.res[0] / 2.0
    cellyh = src.res[1] / 2.0

    left_ = src.bounds.left + (kwargs['window'].col_off * src.res[0]) + cellxh
    top_ = src.bounds.top - (kwargs['window'].row_off * src.res[1]) - cellyh

    xcoords = np.arange(left_, left_ + kwargs['window'].width * src.res[0], src.res[0])
    ycoords = np.arange(top_, top_ - kwargs['window'].height * src.res[1], -src.res[1])

    attrs = {}
    attrs['transform'] = src.gw.transform if hasattr(src, 'gw') else src.transform

    if hasattr(src, 'crs'):
        src_crs = check_src_crs(src)
        try:
            attrs['crs'] = src_crs.to_proj4()
        except:
            attrs['crs'] = src_crs.to_string()

    if hasattr(src, 'res'):
        attrs['res'] = src.res

    if hasattr(src, 'is_tiled'):
        attrs['is_tiled'] = np.uint8(src.is_tiled)

    if hasattr(src, 'nodatavals'):
        attrs['nodatavals'] = tuple(np.nan if nodataval is None else nodataval for nodataval in src.nodatavals)

    if hasattr(src, 'offsets'):
        attrs['offsets'] = src.scales

    if hasattr(src, 'offsets'):
        attrs['offsets'] = src.offsets

    if hasattr(src, 'descriptions') and any(src.descriptions):
        attrs['descriptions'] = src.descriptions

    if hasattr(src, 'units') and any(src.units):
        attrs['units'] = src.units

    return ycoords, xcoords, attrs


@dask.delayed
def read_delayed(fname, chunks, **kwargs):
    with rio.open(fname) as src:
        data_slice = src.read(**kwargs)
        single_band = True if len(data_slice.shape) == 2 else False

        if isinstance(chunks, int):
            chunks_ = (1, chunks, chunks)
        elif isinstance(chunks, tuple):
            chunks_ = (1,) + chunks if len(chunks) < 3 else chunks

        if single_band:
            # Expand to 1 band
            data_slice = da.from_array(data_slice[np.newaxis], chunks=chunks_)
        else:
            data_slice = da.from_array(data_slice, chunks=chunks)

        return data_slice


def read_list(file_list, chunks, **kwargs):
    return [read_delayed(fn, chunks, **kwargs) for fn in file_list]


def read(
    filename,
    band_names=None,
    time_names=None,
    bounds=None,
    chunks=256,
    num_workers=1,
    **kwargs
):
    """Reads a window slice in-memory

    Args:
        filename (str or list): A file name or list of file names to open read.
        band_names (Optional[list]): A list of names to give the output band dimension.
        time_names (Optional[list]): A list of names to give the time dimension.
        bounds (Optional[1d array-like]): A bounding box to subset to, given as
            [minx, miny, maxx, maxy] or [left, bottom, right, top].
        chunks (Optional[tuple]): The data chunk size.
        num_workers (Optional[int]): The number of parallel ``dask`` workers.
        kwargs (Optional[dict]): Keyword arguments to pass to ``rasterio.write``.

    Returns:
        ``xarray.DataArray``
    """
    # Cannot pass 'chunks' to rasterio
    if 'chunks' in kwargs:
        del kwargs['chunks']

    if isinstance(filename, str):
        with rio.open(filename) as src:
            src_transform = src.gw.transform if hasattr(src, 'gw') else src.transform

            if bounds and ('window' not in kwargs):
                kwargs['window'] = from_bounds(*bounds, transform=src_transform)

            ycoords, xcoords, attrs = get_attrs(src, **kwargs)

        data = dask.compute(
            read_delayed(
                filename,
                chunks,
                **kwargs
            ),
            num_workers=num_workers
        )[0]

        if not band_names:
            band_names = np.arange(1, data.shape[0]+1)

        if len(band_names) != data.shape[0]:
            logger.exception('  The band names do not match the output dimensions.')
            raise ValueError

        data = xr.DataArray(
            data,
            dims=('band', 'y', 'x'),
            coords={
                'band': band_names,
                'y': ycoords[:data.shape[-2]],
                'x': xcoords[:data.shape[-1]]
            },
            attrs=attrs
        )

    else:
        with rio.open(filename[0]) as src:
            src_transform = src.gw.transform if hasattr(src, 'gw') else src.transform
            if bounds and ('window' not in kwargs):
                kwargs['window'] = from_bounds(*bounds, transform=src_transform)

            ycoords, xcoords, attrs = get_attrs(src, **kwargs)

        data = da.concatenate(
            dask.compute(
                read_list(
                    filename,
                    chunks,
                    **kwargs
                ),
                num_workers=num_workers
            ),
            axis=0
        )

        if not band_names:
            band_names = np.arange(1, data.shape[-3]+1)

        if len(band_names) != data.shape[-3]:
            logger.exception('  The band names do not match the output dimensions.')
            raise ValueError

        if not time_names:
            time_names = np.arange(1, len(filename)+1)

        if len(time_names) != data.shape[-4]:
            logger.exception('  The time names do not match the output dimensions.')
            raise ValueError

        data = xr.DataArray(
            data,
            dims=('time', 'band', 'y', 'x'),
            coords={
                'time': time_names,
                'band': band_names,
                'y': ycoords[:data.shape[-2]],
                'x': xcoords[:data.shape[-1]]
            },
            attrs=attrs
        )

    return data


data_ = None


class open(object):
    """Opens one or more raster files

    Args:
        filename (str or list): The file name, search string, or a list of files to open.
        band_names (Optional[1d array-like]): A list of band names if ``bounds`` is given or ``window``
            is given. Default is None.
        time_names (Optional[1d array-like]): A list of names to give the time dimension if ``bounds`` is given.
            Default is None.
        stack_dim (Optional[str]): The stack dimension. Choices are ['time', 'band'].
        bounds (Optional[1d array-like]): A bounding box to subset to, given as [minx, maxy, miny, maxx].
            Default is None.
        bounds_by (Optional[str]): How to concatenate the output extent if ``filename`` is a ``list`` and
            ``mosaic`` = ``False``. Choices are ['intersection', 'union', 'reference'].
            * reference: Use the bounds of the reference image. If a ``ref_image`` is not given, the first image in
                the ``filename`` list is used.
            * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
            * union: Use the union (i.e., maximum extent) of all the image bounds

        resampling (Optional[str]): The resampling method if ``filename`` is a ``list``. Choices are
            ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss',
            'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        persist_filenames (Optional[bool]): Whether to persist the filenames list with the ``DataArray`` attributes.
            By default, ``persist_filenames=False`` to avoid storing large file lists.
        netcdf_vars (Optional[list]): NetCDF variables to open as a band stack.
        mosaic (Optional[bool]): If ``filename`` is a ``list``, whether to mosaic the arrays instead of stacking.
        overlap (Optional[str]): The keyword that determines how to handle overlapping data if ``filenames``
            is a ``list``. Choices are ['min', 'max', 'mean'].
        nodata (Optional[float | int]): A 'no data' value to set. Default is 0.
            The 'no data' value is only used under these conditions:
                1. if a ``geowombat.config.update`` context is used
                2. if ``filename`` is a list or search string
            Otherwise, if ``geowombat.open`` is used to open a single image without a config context then no
            warping will be applied and, therefore, `nodata` will not apply.

            See Examples: below for use of ``geowombat.config.update``.
        dtype (Optional[str]): A data type to force the output to. If not given, the data type is extracted
            from the file.
        num_workers (Optional[int]): The number of parallel workers for Dask if ``bounds``
            is given or ``window`` is given. Default is 1.
        kwargs (Optional[dict]): Keyword arguments passed to the file opener.

    Returns:
        ``xarray.DataArray`` or ``xarray.Dataset``

    Examples:
        >>> import geowombat as gw
        >>>
        >>> # Open an image
        >>> with gw.open('image.tif') as ds:
        >>>     print(ds)
        >>>
        >>> # Open a list of images, stacking along the 'time' dimension
        >>> with gw.open(['image1.tif', 'image2.tif']) as ds:
        >>>     print(ds)
        >>>
        >>> # Open all GeoTiffs in a directory, stack along the 'time' dimension
        >>> with gw.open('*.tif') as ds:
        >>>     print(ds)
        >>>
        >>> # Use a context manager to handle images of difference sizes and projections
        >>> with gw.config.update(ref_image='image1.tif'):
        >>>     # Use 'time' names to stack and mosaic non-aligned images with identical dates
        >>>     with gw.open(['image1.tif', 'image2.tif', 'image3.tif'],
        >>>
        >>>         # The first two images were acquired on the same date
        >>>         #   and will be merged into a single time layer
        >>>         time_names=['date1', 'date1', 'date2']) as ds:
        >>>
        >>>         print(ds)
        >>>
        >>> # Mosaic images across space using a reference
        >>> #   image for the CRS and cell resolution
        >>> with gw.config.update(ref_image='image1.tif'):
        >>>     with gw.open(['image1.tif', 'image2.tif'], mosaic=True) as ds:
        >>>         print(ds)
        >>>
        >>> # Mix configuration keywords
        >>> with gw.config.update(ref_crs='image1.tif', ref_res='image1.tif', ref_bounds='image2.tif'):
        >>>     # The ``bounds_by`` keyword overrides the extent bounds
        >>>     with gw.open(['image1.tif', 'image2.tif'], bounds_by='union') as ds:
        >>>         print(ds)
        >>>
        >>> # Resample an image to 10m x 10m cell size
        >>> with gw.config.update(ref_crs=(10, 10)):
        >>>     with gw.open('image.tif', resampling='cubic') as ds:
        >>>         print(ds)
        >>>
        >>> # Open a list of images at a window slice
        >>> from rasterio.windows import Window
        >>> w = Window(row_off=0, col_off=0, height=100, width=100)
        >>>
        >>> # Stack two images, opening band 3
        >>> with gw.open(['image1.tif', 'image2.tif'],
        >>>              band_names=['date1', 'date2'],
        >>>              num_workers=8,
        >>>              indexes=3,
        >>>              window=w,
        >>>              dtype='float32') as ds:
        >>>
        >>>     print(ds)
        >>>
        >>> # Open a NetCDF variable
        >>> with gw.open('netcdf:image.nc:blue') as src:
        >>>     print(src)
        >>>
        >>> # Open multiple NetCDF variables as an array stack
        >>> with gw.open('netcdf:image.nc', netcdf_vars=['blue', 'green', 'red']) as src:
        >>>     print(src)
    """
    def __init__(
        self,
        filename,
        band_names=None,
        time_names=None,
        stack_dim='time',
        bounds=None,
        bounds_by='reference',
        resampling='nearest',
        persist_filenames=False,
        netcdf_vars=None,
        mosaic=False,
        overlap='max',
        nodata=0,
        dtype=None,
        num_workers=1,
        **kwargs
    ):
        if not isinstance(nodata, int) and not isinstance(nodata, float):
            logger.exception("  The 'nodata' keyword argument must be an integer or a float.")
            raise TypeError

        if stack_dim not in ['band', 'time']:
            logger.exception(f"  The 'stack_dim' keyword argument must be either 'band' or 'time', but not {stack_dim}")
            raise NameError

        if isinstance(filename, Path):
            filename = str(filename)

        self.data = data_
        self.__is_context_manager = False
        self.__data_are_separate = False
        self.__data_are_stacked = False
        self.__filenames = []

        if 'chunks' in kwargs:
            if kwargs['chunks']:
                ch.check_chunktype(kwargs['chunks'], output='3d')

        if bounds or ('window' in kwargs and isinstance(kwargs['window'], Window)):
            if 'chunks' not in kwargs:
                if isinstance(filename, list):
                    with rio.open(filename[0]) as src_:
                        w = src_.block_window(1, 0, 0)
                        chunks = (1, w.height, w.width)

                else:
                    with rio.open(filename) as src_:
                        w = src_.block_window(1, 0, 0)
                        chunks = (1, w.height, w.width)

            else:
                chunks = kwargs['chunks']
                del kwargs['chunks']

            self.data = read(
                filename,
                band_names=band_names,
                time_names=time_names,
                bounds=bounds,
                chunks=chunks,
                num_workers=num_workers,
                **kwargs
            )

            self.__filenames = [filename]

        else:
            if (isinstance(filename, str) and '*' in filename) or isinstance(filename, list):
                # Build the filename list
                if isinstance(filename, str):
                    filename = parse_wildcard(filename)

                if 'chunks' not in kwargs:
                    with rio.open(filename[0]) as src:
                        w = src.block_window(1, 0, 0)
                        kwargs['chunks'] = (1, w.height, w.width)

                if mosaic:
                    # Mosaic images over space
                    self.data = gw_mosaic(
                        filename,
                        overlap=overlap,
                        bounds_by=bounds_by,
                        resampling=resampling,
                        band_names=band_names,
                        nodata=nodata,
                        dtype=dtype,
                        **kwargs
                    )

                else:
                    # Stack images along the 'time' axis
                    self.data = gw_concat(
                        filename,
                        stack_dim=stack_dim,
                        bounds_by=bounds_by,
                        resampling=resampling,
                        time_names=time_names,
                        band_names=band_names,
                        nodata=nodata,
                        overlap=overlap,
                        dtype=dtype,
                        netcdf_vars=netcdf_vars,
                        **kwargs
                    )
                    
                    self.__data_are_stacked = True

                self.__data_are_separate = True
                self.__filenames = [str(fn) for fn in filename]

            else:
                self.__filenames = [filename]
                file_names = get_file_extension(filename)

                if (file_names.f_ext.lower() not in IO_DICT['rasterio'] + IO_DICT['xarray']) and not \
                        filename.lower().startswith('netcdf:'):
                    logger.exception('  The file format is not recognized.')
                    raise OSError

                if (file_names.f_ext.lower() in IO_DICT['rasterio']) or filename.lower().startswith('netcdf:'):
                    if 'chunks' not in kwargs:
                        with rio.open(filename) as src:
                            w = src.block_window(1, 0, 0)
                            kwargs['chunks'] = (1, w.height, w.width)

                    self.data = warp_open(
                        filename,
                        band_names=band_names,
                        resampling=resampling,
                        dtype=dtype,
                        netcdf_vars=netcdf_vars,
                        nodata=nodata,
                        **kwargs
                    )

                else:
                    if 'chunks' in kwargs and not isinstance(kwargs['chunks'], dict):
                        logger.exception('  The chunks should be a dictionary.')
                        raise TypeError

                    with xr.open_dataset(filename, **kwargs) as src:
                        self.data = src.to_array(dim='band')

        self.data.attrs['data_are_separate'] = int(self.__data_are_separate)
        self.data.attrs['data_are_stacked'] = int(self.__data_are_stacked)

        if persist_filenames:
            self.data.attrs['filenames'] = self.__filenames

    def __enter__(self):
        self.__is_context_manager = True
        return self.data

    def __exit__(self, *args, **kwargs):
        if not self.data.gw.config['with_config']:
            _set_defaults(config)

        self.close()
        d = self.data
        self._reset(d)

    @staticmethod
    def _reset(d):
        d = None

    @contextmanager
    def _optional_lock(self, needs_lock):
        """Context manager for optionally acquiring a lock."""
        if needs_lock:
            with threading.Lock():
                yield
        else:
            yield

    def close(self):
        if hasattr(self, 'data'):
            if hasattr(self.data, 'gw'):
                if hasattr(self.data.gw, '_obj'):
                    self.data.gw._obj = None

            if hasattr(self.data, 'close'):
                self.data.close()

        if 'gw' in self.data._cache:
            with self._optional_lock(True):
                file = self.data._cache.pop('gw', None)

        self.data = None


def load(
    image_list,
    time_names,
    band_names,
    chunks=512,
    nodata=65535,
    in_range=None,
    out_range=None,
    data_slice=None,
    num_workers=1,
    src=None,
    scheduler='ray'
):
    """Loads data into memory using ``xarray.open_mfdataset`` and ``ray``. This function does not check data
    alignments and CRSs. It assumes each image in ``image_list`` has the same y and x dimensions and
    that the coordinates align.

    The `load` function cannot be used if `dataclasses` was pip installed.

    Args:
        image_list (list): The list of image file paths.
        time_names (list): The list of image ``datetime`` objects.
        band_names (list): The list of bands to open.
        chunks (Optional[int]): The dask chunk size.
        nodata (Optional[float | int]): The 'no data' value.
        in_range (Optional[tuple]): The input (min, max) range. If not given, defaults to (0, 10000).
        out_range (Optional[tuple]): The output (min, max) range. If not given, defaults to (0, 1).
        data_slice (Optional[tuple]): The slice object to read, given as (time, bands, rows, columns).
        num_workers (Optional[int]): The number of threads.
        scheduler (Optional[str]): The distributed scheduler. Currently not implemented.

    Returns:
        ``list``, ``ndarray``:
            Datetime list, array of (time x bands x rows x columns)

    Example:
        >>> import datetime
        >>> import geowombat as gw
        >>>
        >>> image_names = ['LT05_L1TP_227082_19990311_20161220_01_T1.nc',
        >>>                'LT05_L1TP_227081_19990311_20161220_01_T1.nc',
        >>>                'LT05_L1TP_227082_19990327_20161220_01_T1.nc']
        >>>
        >>> image_dates = [datetime.datetime(1999, 3, 11, 0, 0),
        >>>                datetime.datetime(1999, 3, 11, 0, 0),
        >>>                datetime.datetime(1999, 3, 27, 0, 0)]
        >>>
        >>> data_slice = (slice(0, None), slice(0, None), slice(0, 64), slice(0, 64))
        >>>
        >>> # Load data into memory
        >>> dates, y = gw.load(image_names,
        >>>                    image_dates,
        >>>                    ['red', 'nir'],
        >>>                    chunks=512,
        >>>                    nodata=65535,
        >>>                    data_slice=data_slice,
        >>>                    num_workers=4)
    """
    netcdf_prepend = [True for fn in image_list if str(fn).startswith('netcdf:')]

    if any(netcdf_prepend):
        raise NameError('The NetCDF names cannot be prepended with netcdf: when using `geowombat.load()`.')

    if not in_range:
        in_range = (0, 10000)

    if not out_range:
        out_range = (0, 1)

    scale_factor = float(out_range[1]) / float(in_range[1])

    import dask
    from dask.diagnostics import ProgressBar
    import ray
    from ray.util.dask import ray_dask_get

    if src is None:

        with open(
            f'netcdf:{image_list[0]}' if str(image_list[0]).endswith('.nc') else image_list[0],
            time_names=time_names[0],
            band_names=band_names if not str(image_list[0]).endswith('.nc') else None,
            netcdf_vars=band_names if str(image_list[0]).endswith('.nc') else None,
            chunks=chunks
        ) as src:
            pass

    attrs = src.attrs.copy()
    nrows = src.gw.nrows
    ncols = src.gw.ncols
    ycoords = src.y
    xcoords = src.x

    if not data_slice:
        data_slice = (
            slice(0, None),
            slice(0, None),
            slice(0, None),
            slice(0, None)
        )

    def expand_time(dataset):
        """`open_mfdataset` preprocess function
        """
        # Convert the Dataset into a DataArray,
        # rename the band coordinate,
        # select the required VI bands,
        # assign y/x coordinates from a reference,
        # add the time coordiante, and
        # get the sub-array slice
        darray = (
            dataset.to_array()
            .rename({'variable': 'band'})[:, :nrows, :ncols]
            .sel(band=band_names)
            .assign_coords(y=ycoords, x=xcoords)
            .expand_dims(dim='time')
            .clip(0, max(in_range[1], nodata))[data_slice]
        )

        # Scale from [0-10000] -> [0,1]
        darray = xr.where(darray == nodata, 0, darray*scale_factor).astype('float64')

        return (
            darray.where(xr.ufuncs.isfinite(darray))
            .fillna(0)
            .clip(min=out_range[0], max=out_range[1])
        )

    ray.shutdown()
    ray.init(num_cpus=num_workers)

    with dask.config.set(scheduler=ray_dask_get):
        # Open all arrays
        ds = (
            xr.open_mfdataset(
                image_list,
                concat_dim='time',
                chunks={'y': chunks, 'x': chunks},
                combine='nested',
                engine='h5netcdf',
                preprocess=expand_time,
                parallel=True
            )
            .assign_coords(time=time_names)
            .groupby('time.date').max()
            .rename({'date': 'time'})
            .assign_attrs(**attrs)
        )

        # Get the time series dates after grouping
        real_proc_times = ds.gw.pydatetime

        # Convert the DataArray into a NumPy array
        # ds.data.visualize(filename='graph.svg')
        # with performance_report(filename='dask-report.html'):
        with ProgressBar():
            y = ds.data.compute()

    ray.shutdown()

    return real_proc_times, y


class _ImportGPU(object):

    try:
        import jax.numpy as jnp
        JAX_INSTALLED = True
    except:
        JAX_INSTALLED = False

    try:
        import torch
        PYTORCH_INSTALLED = True
    except:
        PYTORCH_INSTALLED = False

    try:
        import tensorflow as tf
        TENSORFLOW_INSTALLED = True
    except:
        TENSORFLOW_INSTALLED = False

    try:
        from tensorflow import keras
        KERAS_INSTALLED = True
    except:
        KERAS_INSTALLED = False


class series(BaseSeries):
    """A class for time series concurrent processing on a GPU

    Args:
        filenames (list): The list of filenames to open.
        band_names (Optional[list]): The band associated names.
        transfer_lib (Optional[str]): The library to transfer data to.
            Choices are ['jax', 'keras', 'numpy', 'pytorch', 'tensorflow'].
        crs (Optional[str]): The coordinate reference system.
        res (Optional[list | tuple]): The cell resolution.
        bounds (Optional[object]): The coordinate bounds.
        resampling (Optional[str]): The resampling method.
        nodata (Optional[float | int]): The 'no data' value.
        warp_mem_limit (Optional[int]): The ``rasterio`` warping memory limit (in MB).
        num_threads (Optional[int]): The number of ``rasterio`` warping threads.
        window_size (Optional[int | list | tuple]): The concurrent processing window size (height, width)
            or -1 (i.e., entire array).
        padding (Optional[list | tuple]): Padding for each window. ``padding`` should be given as a tuple
            of (left pad, bottom pad, right pad, top pad). If ``padding`` is given, the returned list will contain
            a tuple of ``rasterio.windows.Window`` objects as (w1, w2), where w1 contains the normal window offsets
            and w2 contains the padded window offsets.

    Requirement:
        > # CUDA 11.1
        > pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
    """
    def __init__(
        self,
        filenames: list,
        time_names: list = None,
        band_names: list = None,
        transfer_lib: str = 'jax',
        crs: str = None,
        res: T.Union[list, tuple] = None,
        bounds: T.Union[BoundingBox, list, tuple] = None,
        resampling: str = 'nearest',
        nodata: T.Union[float, int] = 0,
        warp_mem_limit: int = 256,
        num_threads: int = 1,
        window_size: T.Union[int, list, tuple] = None,
        padding: T.Union[list, tuple] = None
    ):
        imports_ = _ImportGPU()

        if not imports_.JAX_INSTALLED and (transfer_lib == 'jax'):
            logger.exception('JAX must be installed.')
            raise ImportError('JAX must be installed.')

        if not imports_.PYTORCH_INSTALLED and (transfer_lib == 'pytorch'):
            logger.exception('PyTorch must be installed.')
            raise ImportError('PyTorch must be installed.')

        if not imports_.TENSORFLOW_INSTALLED and (transfer_lib == 'tensorflow'):
            logger.exception('Tensorflow must be installed.')
            raise ImportError('Tensorflow must be installed.')

        if not imports_.KERAS_INSTALLED and (transfer_lib == 'keras'):
            logger.exception('Keras must be installed.')
            raise ImportError('Keras must be installed.')

        self.filenames = filenames
        self.time_names = time_names
        self.band_names = band_names
        self.padding = padding

        self.srcs_ = None
        self.vrts_ = None
        self.windows_ = None

        if transfer_lib == 'jax':
            self.out_array_type = imports_.jnp.DeviceArray
        elif transfer_lib == 'numpy':
            self.out_array_type = np.ndarray
        elif transfer_lib == 'pytorch':
            self.out_array_type = imports_.torch.Tensor
        elif transfer_lib in ['keras', 'tensorflow']:
            self.out_array_type = imports_.tf.Tensor

        self.put = TransferLib(transfer_lib)

        self.open(filenames)

        self.warp(
            dst_crs=crs,
            dst_res=res,
            dst_bounds=bounds,
            resampling=resampling,
            nodata=nodata,
            warp_mem_limit=warp_mem_limit,
            num_threads=num_threads,
            window_size=window_size,
            padding=self.padding
        )

    def read(
        self,
        bands: T.Union[int, list],
        window: Window = None,
        gain: float = 1.0,
        offset: T.Union[float, int] = 0.0,
        pool: T.Any = None,
        num_workers: int = None,
        tqdm_obj: T.Any = None
    ) -> T.Any:
        """Reads a window
        """
        if isinstance(bands, int):
            if bands == -1:
                band_list = list(range(1, self.count+1))
            else:
                band_list = [bands]

        else:
            band_list = bands

        def _read(vrt_ptr, bd):
            array = vrt_ptr.read(bd, window=window)
            mask = vrt_ptr.read_masks(bd, window=window)

            array = array * gain + offset
            array[mask == 0] = np.nan

            return array

        if pool is not None:
            def _read_bands(vrt_):
                return np.stack(
                    [_read(vrt_, band) for band in band_list]
                )

            with pool(num_workers) as executor:
                data_gen = (vrt for vrt in self.vrts_)

                results = []
                for res in tqdm_obj(executor.map(_read_bands, data_gen), total=len(self.vrts_)):
                    results.append(res)

            return self.put(np.array(results))

        else:
            return self.put(
                np.array([
                    np.stack([
                        _read(vrt, band) for band in band_list
                    ]) for vrt in self.vrts_
                ])
            )

    @staticmethod
    def _create_file(filename, **profile):
        if Path(filename).is_file():
            Path(filename).unlink()

        with rio.open(filename, mode='w', **profile) as dst:
            pass

    def apply(
        self,
        func: T.Union[T.Callable, str, list, tuple],
        bands: T.Union[list, int],
        gain: float = 1.0,
        offset: T.Union[float, int] = 0.0,
        processes: bool = False,
        num_workers: int = 1,
        monitor_progress: bool = True,
        outfile: T.Union[Path, str] = None
    ):
        """Applies a function concurrently over windows

        Args:
            func (object | str | list | tuple): The function to apply. If ``func`` is a string,
                choices are ['cv', 'max', 'mean', 'min'].
            bands (list | int): The bands to read.
            gain (Optional[float]): A gain factor to apply.
            offset (Optional[float | int]): An offset factor to apply.
            processes (Optional[bool]): Whether to use process workers, otherwise use threads.
            num_workers (Optional[int]): The number of concurrent workers.
            monitor_progress (Optional[bool]): Whether to monitor progress with a ``tqdm`` bar.
            outfile (Optional[Path | str]): The output file.

        Returns:
            If outfile is None:
                Window, array, [datetime, ...]
            If outfile is not None:
                None, writes to ``outfile``

        Example:
            >>> import itertools
            >>> import geowombat as gw
            >>> import rasterio as rio
            >>>
            >>> # Import an image with 3 bands
            >>> from geowombat.data import l8_224078_20200518
            >>>
            >>> # Create a custom class
            >>> class TemporalMean(gw.TimeModule):
            >>>
            >>>     def __init__(self):
            >>>         super(TemporalMean, self).__init__()
            >>>
            >>>     # The main function
            >>>     def calculate(self, array):
            >>>
            >>>         sl1 = (slice(0, None), slice(self.band_dict['red'], self.band_dict['red']+1), slice(0, None), slice(0, None))
            >>>         sl2 = (slice(0, None), slice(self.band_dict['green'], self.band_dict['green']+1), slice(0, None), slice(0, None))
            >>>
            >>>         vi = (array[sl1] - array[sl2]) / ((array[sl1] + array[sl2]) + 1e-9)
            >>>
            >>>         return vi.mean(axis=0).squeeze()
            >>>
            >>> with rio.open(l8_224078_20200518) as src:
            >>>     res = src.res
            >>>     bounds = src.bounds
            >>>     nodata = 0
            >>>
            >>> # Open many files, each with 3 bands
            >>> with gw.series([l8_224078_20200518]*100,
            >>>                band_names=['blue', 'green', 'red'],
            >>>                crs='epsg:32621',
            >>>                res=res,
            >>>                bounds=bounds,
            >>>                nodata=nodata,
            >>>                num_threads=4,
            >>>                window_size=(1024, 1024)) as src:
            >>>
            >>>     src.apply(TemporalMean(),
            >>>               bands=-1,             # open all bands
            >>>               gain=0.0001,          # scale from [0,10000] -> [0,1]
            >>>               processes=False,      # use threads
            >>>               num_workers=4,        # use 4 concurrent threads, one per window
            >>>               outfile='vi_mean.tif')
            >>>
            >>> # Import a single-band image
            >>> from geowombat.data import l8_224078_20200518_B4
            >>>
            >>> # Open many files, each with 1 band
            >>> with gw.series([l8_224078_20200518_B4]*100,
            >>>                band_names=['red'],
            >>>                crs='epsg:32621',
            >>>                res=res,
            >>>                bounds=bounds,
            >>>                nodata=nodata,
            >>>                num_threads=4,
            >>>                window_size=(1024, 1024)) as src:
            >>>
            >>>     src.apply('mean',               # built-in function over single-band images
            >>>               bands=1,              # open all bands
            >>>               gain=0.0001,          # scale from [0,10000] -> [0,1]
            >>>               num_workers=4,        # use 4 concurrent threads, one per window
            >>>               outfile='red_mean.tif')
            >>>
            >>> with gw.series([l8_224078_20200518_B4]*100,
            >>>                band_names=['red'],
            >>>                crs='epsg:32621',
            >>>                res=res,
            >>>                bounds=bounds,
            >>>                nodata=nodata,
            >>>                num_threads=4,
            >>>                window_size=(1024, 1024)) as src:
            >>>
            >>>     src.apply(['mean', 'max', 'cv'],    # calculate multiple statistics
            >>>               bands=1,                  # open all bands
            >>>               gain=0.0001,              # scale from [0,10000] -> [0,1]
            >>>               num_workers=4,            # use 4 concurrent threads, one per window
            >>>               outfile='stack_mean.tif')
        """
        pool = concurrent.futures.ProcessPoolExecutor if processes else concurrent.futures.ThreadPoolExecutor

        tqdm_obj = tqdm if monitor_progress else _tqdm

        if isinstance(func, str) or isinstance(func, list) or isinstance(func, tuple):
            if isinstance(bands, list) or isinstance(bands, tuple):
                logger.exception('Only single-band images can be used with built-in functions.')
                raise ValueError('Only single-band images can be used with built-in functions.')

            apply_func_ = SeriesStats(func)

        else:
            apply_func_ = func

        if outfile is not None:
            profile = {
                'count': apply_func_.count,
                'width': self.width,
                'height': self.height,
                'crs': self.crs,
                'transform': self.transform,
                'driver': 'GTiff',
                'dtype': apply_func_.dtype,
                'compress': apply_func_.compress,
                'sharing': False,
                'tiled': True,
                'nodata': self.nodata,
                'blockxsize': self.blockxsize,
                'blockysize': self.blockysize
            }

            # Create the file
            self._create_file(outfile, **profile)

        if outfile is not None:
            with rio.open(outfile, mode='r+', sharing=False) as dst:
                with pool(num_workers) as executor:
                    data_gen = (
                        (w, self.read(bands, window=w[1], gain=gain, offset=offset), self.band_dict)
                        if self.padding
                        else (w, self.read(bands, window=w, gain=gain, offset=offset), self.band_dict)
                        for w in self.windows_
                    )

                    for w, res in tqdm_obj(executor.map(lambda f: apply_func_(*f), data_gen), total=self.nchunks):

                        with threading.Lock():
                            self._write_window(dst, res, apply_func_.count, w)
        else:
            if self.padding:
                w, res = apply_func_(
                    self.windows_[0],
                    self.read(
                        bands, window=self.windows_[0][1], gain=gain, offset=offset,
                        pool=pool, num_workers=num_workers, tqdm_obj=tqdm_obj
                    ),
                    self.band_dict
                )
            else:
                w, res = apply_func_(
                    self.windows_[0],
                    self.read(
                        bands, window=self.windows_[0], gain=gain, offset=offset,
                        pool=pool, num_workers=num_workers, tqdm_obj=tqdm_obj
                    ),
                    self.band_dict
                )

            # Group duplicate dates
            res, image_dates = self.group_dates(res, self.time_names, self.band_names)

            return w, res, image_dates

    def _write_window(self, dst_, out_data_, count, w):
        if self.padding:
            window_ = w[0]
            padded_window_ = w[1]

            # Get the non-padded array slice
            row_diff = abs(window_.row_off - padded_window_.row_off)
            col_diff = abs(window_.col_off - padded_window_.col_off)

            out_data_ = out_data_[:, row_diff:row_diff+window_.height, col_diff:col_diff+window_.width]

        dst_.write(
            out_data_,
            indexes=1 if count == 1 else range(1, count+1),
            window=w[0] if self.padding else w
        )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        for src in self.srcs_:
            src.close()

        for vrt in self.vrts_:
            vrt.close()


from dataclasses import dataclass
import enum

from ..radiometry import QABits

import pandas as pd
import geopandas as gpd
from rasterio.enums import Resampling
import pyproj
from pystac_client import Client
import pystac.errors as pystac_errors
from pystac import Catalog, ItemCollection
from pystac.extensions.eo import EOExtension as eo
from satsearch import Search
import stackstac
import planetary_computer as pc
from rich.console import Console
from rich.table import Table
import wget


class STACNames(enum.Enum):
    element84 = 'element84'
    google = 'google'
    microsoft = 'microsoft'


class STACCollectionTypes(enum.Enum):
    landsat = 'landsat'
    landsat_c2_l2 = 'landsat'
    landsat_l8_c2_l2 = 'landsat'
    sentinel2 = 'sentinel2'
    sentinel_s2_l2a = 'sentinel2'
    sentinel_s2_l1c = 'sentinel2'


@dataclass
class STACCatalogs:
    element84 = 'https://earth-search.aws.element84.com/v0'
    google = 'https://earthengine-stac.storage.googleapis.com/catalog/catalog.json'
    microsoft = 'https://planetarycomputer.microsoft.com/api/stac/v1'


@dataclass
class STACCollections:
    # All Landsat, Collection 2, Level 2 (surface reflectance)
    landsat_c2_l2 = {
        STACNames.google.value: [
            'LC09/C02/T1_L2', 'LC08/C02/T1_L2', 'LE07/C02/T1_L2', 'LT05/C02/T1_L2'
        ],
        STACNames.microsoft.value: 'landsat-c2-l2',
    }
    # Sentinel-2, Level 2A (surface reflectance missing cirrus band)
    sentinel_s2_l2a = {
        STACNames.element84.value: 'sentinel-s2-l2a-cogs',
        STACNames.google.value: 'sentinel-2-l2a',
        STACNames.microsoft.value: 'sentinel-2-l2a'
    }
    # Sentinel-2, Level 1c (top of atmosphere with all 13 bands available)
    sentinel_s2_l1c = {
        STACNames.element84.value: 'sentinel-s2-l1c'
    }
    # Landsat 8, Collection 2, Tier 1 (Level 2 (surface reflectance))
    landsat_l8_c2_l2 = {
        STACNames.google.value: 'LC08_C02_T1_L2',
        STACNames.microsoft.value: 'landsat-8-c2-l2'
    }


@dataclass
class STACCatalogOpeners:
    element84 = Client.open
    google = Catalog.from_file
    microsoft = Client.open


def merge_stac(data: xr.DataArray, *other: T.Sequence[xr.DataArray]) -> xr.DataArray:
    name = data.collection
    for darray in other:
        name += f'+{darray.collection}'
    collections = [data.collection]*data.gw.ntime
    for darray in other:
        collections += [darray.collection]*darray.gw.ntime

    stack = (
        xr.DataArray(
            da.concatenate(
                (
                    data.transpose('time', 'band', 'y', 'x').data,
                    *(darray.transpose('time', 'band', 'y', 'x').data for darray in other)
                ), axis=0
            ),
            dims=('time', 'band', 'y', 'x'),
            coords={
                'time': np.concatenate(
                    (data.time.values, *(darray.time.values for darray in other))
                ),
                'band': data.band.values,
                'y': data.y.values,
                'x': data.x.values
            },
            attrs=data.attrs,
            name=name
        )
        .sortby('time')
    )

    return (
        stack
        .groupby('time')
        .mean(dim='time', skipna=True)
        .assign_attrs(**data.attrs)
        .assign_attrs(collection=None)
    )


def open_stac(
    stac_catalog: str = 'microsoft',
    collection: str = None,
    bounds: T.Union[str, Path, gpd.GeoDataFrame] = None,
    start_date: str = None,
    end_date: str = None,
    cloud_cover_perc: T.Union[float, int] = 50,
    bands: T.Sequence[str] = None,
    chunksize: int = 256,
    mask_items: T.Sequence[str] = None,
    bounds_query: T.Optional[str] = None,
    mask_data: T.Optional[bool] = False,
    epsg: T.Optional[int] = None,
    resolution: T.Optional[T.Union[float, int]] = None,
    resampling: T.Optional[Resampling] = Resampling.nearest,
    nodata_fill: T.Optional[T.Union[float, int]] = None,
    view_asset_keys: T.Optional[bool] = False,
    extra_assets: T.Optional[T.Sequence[str]] = None,
    out_path: T.Optional[T.Union[Path, str]] = '.'
) -> xr.DataArray:
    """Opens a collection from a spatio-temporal asset catalog (STAC).

    Args:
        stac_catalog (str): Choices are ['element84', 'google', 'microsoft'].
        collection (str): The STAC collection to open.
        bounds (sequence | str | Path | GeoDataFrame): The search bounding box. This can also be given with the
            configuration manager (e.g., ``gw.config.update(ref_bounds=bounds)``)
        start_date (str): The start search date (yyyy-mm-dd).
        end_date (str): The end search date (yyyy-mm-dd).
        cloud_cover_perc (float | int): The maximum percentage cloud cover.
        bands (sequence): The bands to open.
        chunksize (int): The dask chunk size.
        mask_items (sequence): The items to mask.
        bounds_query (Optional[str]): A query to select bounds from the ``GeoDataFrame``.
        mask_data (Optional[bool]): Whether to mask the data. Only relevant if ``mask_items=True``.
        epsg (Optional[int]): An EPSG code to warp to.
        resolution (Optional[float | int]): The cell resolution to resample to.
        resampling (Optional[rasterio.enumsResampling enum]): The resampling method.
        nodata_fill (Optional[float | int]): A fill value to replace 'no data' NaNs.
        view_asset_keys (Optional[bool]): Whether to view asset ids.
        extra_assets (Optional[list]): Extra assets (non-image assets) to download.
        out_path (Optional[str | Path]): The output path to save files to.

    Returns:
        xarray.DataArray

    Example:
        >>> import geowombat as gw
        >>>
        >>> data_l = gw.open_stac(
        >>>     stac_catalog='microsoft',
        >>>     collection='landsat_c2_l2',
        >>>     bounds='map.geojson',
        >>>     bands=['red', 'green', 'blue', 'qa_pixel'],
        >>>     mask_data=True
        >>> )
        >>>
        >>> from rasterio.enums import Resampling
        >>>
        >>> data_s2 = gw.open_stac(
        >>>     stac_catalog='microsoft',
        >>>     collection='sentinel_s2_l2a',
        >>>     bounds='map.geojson',
        >>>     bands=['B04', 'B03', 'B02'],
        >>>     resolution=30,
        >>>     resampling=Resampling.cubic,
        >>>     epsg=32621
        >>> )
        >>>
        >>> # Merge two temporal stacks
        >>> stack = (
        >>>     gw.merge_stac(data_l, data_s2)
        >>>     .sel(band='red')
        >>>     .mean(dim='time')
        >>> )
    """
    if collection is None:
        raise NameError('A collection must be given.')

    df = pd.DataFrame()

    # Date range
    date_range = f"{start_date}/{end_date}"
    # Search bounding box
    if bounds is None:
        bounds = config['ref_bounds']
    assert bounds is not None, 'The bounds must be given in some format.'
    if not isinstance(bounds, gpd.GeoDataFrame):
        bounds = gpd.read_file(bounds)
        assert bounds.crs == pyproj.CRS.from_epsg(4326), 'The CRS should be WGS84/latlon (EPSG=4326)'
    if bounds_query is not None:
        bounds = bounds.query(bounds_query)
    bounds = bounds.bounds.values.flatten().tolist()

    stac_catalog_url = getattr(STACCatalogs, stac_catalog)
    # Open the STAC catalog
    catalog = getattr(STACCatalogOpeners, getattr(STACNames, stac_catalog).value)(stac_catalog_url)
    search = None
    # Get Google collections
    # NOTE: Google STAC does not work
    # if getattr(STACNames, stac_catalog) is STACNames.google:
    #     parent = getattr(STACCollectionTypes, collection).value
    #     catalog_ee = catalog.get_child(parent)
    #     if isinstance(getattr(STACCollections, collection)[getattr(STACNames, stac_catalog).value], list):
    #         catalog_collections = []
    #         ee_children = getattr(STACCollections, collection)[getattr(STACNames, stac_catalog).value]
    #         for child in tqdm(ee_children, desc='Get EE collections'):
    #             catalog_ee_sub = catalog_ee.get_child(f"{parent}/{child}")
    #             catalog_ee_sub.to_dict()
    #             search = Search(
    #                 url=catalog_ee_sub.get_self_href(),
    #                 bbox=bounds,
    #                 datetime=date_range,
    #                 query={"eo:cloud_cover": {"lt": cloud_cover_perc}}
    #             )
    #             # catalog_collections += list(catalog_ee_sub.get_collections())
    # else:
    catalog_collections = [
        getattr(STACCollections, collection)[getattr(STACNames, stac_catalog).value]
    ]
    # Search the STAC
    search = catalog.search(
        collections=catalog_collections,
        bbox=bounds,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_perc}}
    )

    if search is None:
        raise ValueError('No items found.')

    if list(search.get_items()):
        if getattr(STACNames, stac_catalog) is STACNames.microsoft:
            items = pc.sign(search)
        else:
            items = ItemCollection(items=list(search.items()))

        if view_asset_keys:
            try:
                selected_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
            except pystac_errors.ExtensionNotImplemented:
                selected_item = items.items[0]
            table = Table("Asset Key", "Description")
            for asset_key, asset in selected_item.assets.items():
                table.add_row(asset_key, asset.title)
            console = Console()
            console.print(table)

            return None, None

        # Download metadata and coefficient files
        if extra_assets is not None:
            for item in tqdm(items, desc='Extra assets', position=0):
                df_dict = {'id': item.id}
                for extra in tqdm(extra_assets, desc='Asset', position=1, leave=False):
                    url = item.assets[extra].to_dict()['href']
                    out_name = Path(out_path) / f"{item.id}_{Path(url.split('?')[0]).name}"
                    df_dict[extra] = str(out_name)
                    if not out_name.is_file():
                        wget.download(url, out=str(out_name), bar=None)
                df = df.append(df_dict, ignore_index=True)

        data = stackstac.stack(
            items,
            bounds_latlon=bounds,
            assets=bands,
            chunksize=chunksize,
            epsg=epsg,
            resolution=resolution,
            resampling=resampling
        )
        data = data.assign_attrs(
            res=(data.resolution, data.resolution),
            collection=collection
        )
        # if (
        #     hasattr(data, 'common_name')
        #     and (getattr(STACCollectionTypes, collection) is not STACCollectionTypes.landsat)
        # ):
        #     data = data.assign_coords(
        #         band=lambda x: x.common_name.rename('band')
        #     )

        if mask_data:
            if mask_items is None:
                mask_items = ['fill', 'dilated_cloud', 'cirrus', 'cloud', 'cloud_shadow', 'snow']
            mask_bitfields = [getattr(QABits, collection).value[mask_item] for mask_item in mask_items]
            # Source: https://stackstac.readthedocs.io/en/v0.3.0/examples/gif.html
            bitmask = 0
            for field in mask_bitfields:
                bitmask |= 1 << field
            # TODO: get qa_pixel name for different sensors
            qa = data.sel(band='qa_pixel').astype('uint16')
            mask = qa & bitmask
            data = data.sel(band=[band for band in bands if band != 'qa_pixel']).where(mask == 0)

        data.where(lambda x: x != 0)
        if nodata_fill is not None:
            data = data.fillna(nodata_fill)

        return data, df

    return None, None
