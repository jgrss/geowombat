# https://github.com/pydata/xarray/issues/2560
try:
    import netCDF4
except:
    pass

try:
    import h5netcdf
except:
    pass

from abc import abstractmethod
import warnings
import itertools
from pathlib import Path
import logging
import threading
from contextlib import contextmanager
from typing import Callable, Union
import concurrent.futures

from . import geoxarray
from ..handler import add_handler
from ..config import config, _set_defaults
from ..backends import concat as gw_concat
from ..backends import mosaic as gw_mosaic
from ..backends import warp_open
from ..backends.rasterio_ import check_src_crs
from .util import Chunks, get_file_extension, parse_wildcard

import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.windows import from_bounds, Window
from rasterio.coords import BoundingBox
import dask
import dask.array as da

from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from tqdm.auto import tqdm

try:
    import jax.numpy as jnp
    from jax import device_put
    JAX_INSTALLED = True
except:
    JAX_INSTALLED = False


logger = logging.getLogger(__name__)
logger = add_handler(logger)
warnings.filterwarnings('ignore')

ch = Chunks()

IO_DICT = dict(rasterio=['.tif',
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
                         '.H5'],
               xarray=['.nc'])


def get_attrs(src, **kwargs):

    cellxh = src.res[0] / 2.0
    cellyh = src.res[1] / 2.0

    left_ = src.bounds.left + (kwargs['window'].col_off * src.res[0]) + cellxh
    top_ = src.bounds.top - (kwargs['window'].row_off * src.res[1]) - cellyh

    xcoords = np.arange(left_, left_ + kwargs['window'].width * src.res[0], src.res[0])
    ycoords = np.arange(top_, top_ - kwargs['window'].height * src.res[1], -src.res[1])

    attrs = dict()

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
            data_slice = da.from_array(data_slice[np.newaxis, :, :],
                                       chunks=chunks_)

        else:

            data_slice = da.from_array(data_slice,
                                       chunks=chunks)

        return data_slice


def read_list(file_list, chunks, **kwargs):
    return [read_delayed(fn, chunks, **kwargs) for fn in file_list]


def read(filename,
         band_names=None,
         time_names=None,
         bounds=None,
         chunks=256,
         num_workers=1,
         **kwargs):

    """
    Reads a window slice in-memory

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

        data = dask.compute(read_delayed(filename,
                                         chunks,
                                         **kwargs),
                            num_workers=num_workers)[0]

        if not band_names:
            band_names = np.arange(1, data.shape[0]+1)

        if len(band_names) != data.shape[0]:
            logger.exception('  The band names do not match the output dimensions.')
            raise ValueError

        data = xr.DataArray(data,
                            dims=('band', 'y', 'x'),
                            coords={'band': band_names,
                                    'y': ycoords[:data.shape[-2]],
                                    'x': xcoords[:data.shape[-1]]},
                            attrs=attrs)

    else:

        with rio.open(filename[0]) as src:

            src_transform = src.gw.transform if hasattr(src, 'gw') else src.transform

            if bounds and ('window' not in kwargs):
                kwargs['window'] = from_bounds(*bounds, transform=src_transform)

            ycoords, xcoords, attrs = get_attrs(src, **kwargs)

        data = da.concatenate(dask.compute(read_list(filename,
                                                     chunks,
                                                     **kwargs),
                                           num_workers=num_workers),
                              axis=0)

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

        data = xr.DataArray(data,
                            dims=('time', 'band', 'y', 'x'),
                            coords={'time': time_names,
                                    'band': band_names,
                                    'y': ycoords[:data.shape[-2]],
                                    'x': xcoords[:data.shape[-1]]},
                            attrs=attrs)

    return data


data_ = None


class open(object):

    """
    Opens one or more raster files

    Args:
        filename (str or list): The file name, search string, or a list of files to open.
        band_names (Optional[1d array-like]): A list of band names if ``bounds`` is given or ``window``
            is given. Default is None.
        time_names (Optional[1d array-like]): A list of names to give the time dimension if ``bounds`` is given.
            Default is None.
        stack_dim (Optional[str]): The stack dimension. Choices are ['time', 'band'].
        bounds (Optional[1d array-like]): A bounding box to subset to, given as [minx, maxy, miny, maxx].
            Default is None.
        bounds_by (Optional[str]): How to concatenate the output extent if ``filename`` is a ``list`` and ``mosaic`` = ``False``.
            Choices are ['intersection', 'union', 'reference'].

            * reference: Use the bounds of the reference image. If a ``ref_image`` is not given, the first image in the ``filename`` list is used.
            * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
            * union: Use the union (i.e., maximum extent) of all the image bounds

        resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        persist_filenames (Optional[bool]): Whether to persist the filenames list with the ``DataArray`` attributes.
            By default, ``persist_filenames=False`` to avoid storing large file lists.
        netcdf_vars (Optional[list]): NetCDF variables to open as a band stack.
        mosaic (Optional[bool]): If ``filename`` is a ``list``, whether to mosaic the arrays instead of stacking.
        overlap (Optional[str]): The keyword that determines how to handle overlapping data if ``filenames`` is a ``list``.
            Choices are ['min', 'max', 'mean'].
        nodata (Optional[float | int]): A 'no data' value to set. Default is 0.
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
        >>>
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
        >>>
        >>>     # The ``bounds_by`` keyword overrides the extent bounds
        >>>     with gw.open(['image1.tif', 'image2.tif'], bounds_by='union') as ds:
        >>>         print(ds)
        >>>
        >>> # Resample an image to 10m x 10m cell size
        >>> with gw.config.update(ref_crs=(10, 10)):
        >>>
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

    def __init__(self,
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
                 **kwargs):

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

            self.data = read(filename,
                             band_names=band_names,
                             time_names=time_names,
                             bounds=bounds,
                             chunks=chunks,
                             num_workers=num_workers,
                             **kwargs)

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
                    self.data = gw_mosaic(filename,
                                          overlap=overlap,
                                          bounds_by=bounds_by,
                                          resampling=resampling,
                                          band_names=band_names,
                                          nodata=nodata,
                                          dtype=dtype,
                                          **kwargs)

                else:

                    # Stack images along the 'time' axis
                    self.data = gw_concat(filename,
                                          stack_dim=stack_dim,
                                          bounds_by=bounds_by,
                                          resampling=resampling,
                                          time_names=time_names,
                                          band_names=band_names,
                                          nodata=nodata,
                                          overlap=overlap,
                                          dtype=dtype,
                                          netcdf_vars=netcdf_vars,
                                          **kwargs)
                    
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

                if (file_names.f_ext.lower() in IO_DICT['rasterio']) or \
                        (filename.lower().startswith('netcdf:')):

                    if 'chunks' not in kwargs:

                        with rio.open(filename) as src:

                            w = src.block_window(1, 0, 0)
                            kwargs['chunks'] = (1, w.height, w.width)

                    self.data = warp_open(filename,
                                          band_names=band_names,
                                          resampling=resampling,
                                          dtype=dtype,
                                          netcdf_vars=netcdf_vars,
                                          **kwargs)

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


def load(image_list,
         time_names,
         band_names,
         chunks=512,
         nodata=65535,
         in_range=None,
         out_range=None,
         data_slice=None,
         num_workers=1,
         src=None,
         scheduler='ray'):

    """
    Loads data into memory using ``xarray.open_mfdataset`` and ``ray``. This function does not check data
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

        with open(f'netcdf:{image_list[0]}' if str(image_list[0]).endswith('.nc') else image_list[0],
                  time_names=time_names[0],
                  band_names=band_names if not str(image_list[0]).endswith('.nc') else None,
                  netcdf_vars=band_names if str(image_list[0]).endswith('.nc') else None,
                  chunks=chunks) as src:

            pass

    attrs = src.attrs.copy()
    nrows = src.gw.nrows
    ncols = src.gw.ncols
    ycoords = src.y
    xcoords = src.x

    if not data_slice:

        data_slice = (slice(0, None),
                      slice(0, None),
                      slice(0, None),
                      slice(0, None))

    def expand_time(dataset):

        """
        `open_mfdataset` preprocess function
        """

        # Convert the Dataset into a DataArray,
        # rename the band coordinate,
        # select the required VI bands,
        # assign y/x coordinates from a reference,
        # add the time coordiante, and
        # get the sub-array slice
        darray = dataset.to_array()\
                        .rename({'variable': 'band'})[:, :nrows, :ncols]\
                        .sel(band=band_names)\
                        .assign_coords(y=ycoords, x=xcoords)\
                        .expand_dims(dim='time')\
                        .clip(0, max(in_range[1], nodata))[data_slice]

        # Scale from [0-10000] -> [0,1]
        darray = xr.where(darray == nodata, 0, darray*scale_factor).astype('float64')

        return darray.where(xr.ufuncs.isfinite(darray))\
                        .fillna(0)\
                        .clip(min=out_range[0], max=out_range[1])

    ray.shutdown()
    ray.init(num_cpus=num_workers)

    with dask.config.set(scheduler=ray_dask_get):

        # Open all arrays
        ds = xr.open_mfdataset(image_list,
                               concat_dim='time',
                               chunks={'y': chunks, 'x': chunks},
                               combine='nested',
                               engine='h5netcdf',
                               preprocess=expand_time,
                               parallel=True)\
                    .assign_coords(time=time_names)\
                    .groupby('time.date').max()\
                    .rename({'date': 'time'})\
                    .assign_attrs(**attrs)

        # Get the time series dates after grouping
        real_proc_times = ds.gw.pydatetime

        # Convert the DataArray into a NumPy array
        # ds.data.visualize(filename='graph.svg')
        # with performance_report(filename='dask-report.html'):
        with ProgressBar():
            y = ds.data.compute()

    ray.shutdown()

    return real_proc_times, y


class _Warp(object):

    def warp(self,
             dst_crs=None,
             dst_res=None,
             dst_bounds=None,
             resampling='nearest',
             nodata=None,
             warp_mem_limit=None,
             num_threads=None,
             window_size=None):

        dst_transform = Affine(dst_res[0], 0.0, dst_bounds.left, 0.0, -dst_res[1], dst_bounds.top)

        dst_width = int((dst_bounds.right - dst_bounds.left) / dst_res[0])
        dst_height = int((dst_bounds.top - dst_bounds.bottom) / dst_res[1])

        vrt_options = {'resampling': getattr(Resampling, resampling),
                       'crs': dst_crs,
                       'transform': dst_transform,
                       'height': dst_height,
                       'width': dst_width,
                       'nodata': nodata,
                       'warp_mem_limit': warp_mem_limit,
                       'warp_extras': {'multi': True,
                                       'warp_option': f'NUM_THREADS={num_threads}'}}

        self.vrts_ = [WarpedVRT(src,
                                src_crs=src.crs,
                                src_transform=src.transform,
                                **vrt_options) for src in self.srcs_]

        if window_size:

            def adjust_window(pixel_index, block_size, rows_cols):
                return block_size if (pixel_index + block_size) < rows_cols else rows_cols - pixel_index

            self.windows_ = []

            for row_off in range(0, dst_height, window_size[0]):
                wheight = adjust_window(row_off, window_size[0], dst_height)
                for col_off in range(0, dst_width, window_size[1]):
                    wwidth = adjust_window(col_off, window_size[1], dst_width)
                    self.windows_.append(Window(row_off=row_off, col_off=col_off, height=wheight, width=wwidth))

        else:
            self.windows_ = [[w[1] for w in src.block_windows(1)] for src in self.vrts_][0]


class _Open(_Warp):

    def __init__(self, filenames):
        self.srcs_ = [rio.open(fn) for fn in filenames]


class _PropsMixin(object):

    @property
    def crs(self):
        return self.handler.vrts_[0].crs

    @property
    def transform(self):
        return self.handler.vrts_[0].transform

    @property
    def count(self):
        return self.handler.vrts_[0].count

    @property
    def width(self):
        return self.handler.vrts_[0].width

    @property
    def height(self):
        return self.handler.vrts_[0].height

    @property
    def blockxsize(self):
        return self.handler.windows_[0].width

    @property
    def blockysize(self):
        return self.handler.windows_[0].height

    @property
    def nchunks(self):
        return len(self.handler.windows_)

    @property
    def band_dict(self):
        return dict(zip(self.band_names, range(0, self.count))) if self.band_names else None


class TimeModule(_AbstractTimeModule):

    def __init__(self):

        self.dtype = 'float64'
        self.count = 1
        self.compress = 'lzw'

    def __call__(self, *args):

        w, array, band_dict = list(itertools.chain(*args))

        return self.run(w, array, band_dict=band_dict)

    def __repr__(self):
        print(f'TimeModule() -> self.dtype={self.dtype}, self.count={self.count}')

    @abstractmethod
    def run(self, w, array, band_dict=None):
        raise NotImplementedError


class TimeSeriesStats(TimeModule):

    def __init__(self, time_stats):

        super(TimeSeriesStats, self).__init__()

        self.time_stats = time_stats

        if isinstance(self.time_stats, str):
            self.count = 1
        else:
            self.count = len(list(self.time_stats))

    def run(self, w, array, band_dict=None):

        if isinstance(self.time_stats, str):
            return getattr(self, self.time_stats)(w, array)
        else:
            return self._stack(w, array, self.time_stats)

    @staticmethod
    def _scale_min_max(xv, mni, mxi, mno, mxo):
        return ((((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno).clip(mno, mxo)

    @staticmethod
    def _lstsq(data):

        ndims, nbands, nrows, ncols = data.shape

        M = data.squeeze().transpose(1, 2, 0).reshape(nrows*ncols, ndims).T

        x = jnp.arange(0, M.shape[0])

        # Fit a least squares solution to each sample
        return jnp.linalg.lstsq(jnp.c_[x, jnp.ones_like(x)], M, rcond=None)[0]

    @staticmethod
    def amp(w, array):
        """Calculates the amplitude"""
        return w, jnp.nanmax(array, axis=0).squeeze() - jnp.nanmin(array, axis=0).squeeze()

    @staticmethod
    def cv(w, array):
        """Calculates the coefficient of variation"""
        return w, jnp.nanstd(array, axis=0).squeeze() / (jnp.nanmean(array, axis=0).squeeze() + 1e-9)

    @staticmethod
    def max(w, array):
        """Calculates the max"""
        return w, jnp.nanmax(array, axis=0).squeeze()

    @staticmethod
    def mean(w, array):
        """Calculates the mean"""
        return w, jnp.nanmean(array, axis=0).squeeze()

    def mean_abs_diff(self, w, array):
        """Calculates the mean absolute difference"""
        d = jnp.nanmean(jnp.fabs(jnp.diff(array, n=1, axis=0)), axis=0).squeeze()
        return w, self._scale_min_max(d, 0.0, 0.05, 0.0, 1.0)

    @staticmethod
    def min(w, array):
        """Calculates the min"""
        return w, jnp.nanmin(array, axis=0).squeeze()

    @staticmethod
    def norm_abs_energy(w, array):
        """Calculates the normalized absolute energy"""
        return w, jnp.nansum(array**2, axis=0).squeeze() / (jnp.nanmax(array, axis=0)**2 * array.shape[0]).squeeze()

    def _stack(self, w, array, stats):
        """Calculates a stack of statistics"""
        return w, jnp.stack([getattr(self, stat)(w, array)[1][np.newaxis, :, :] for stat in stats]).squeeze()


@contextmanager
def _tqdm(*args, **kwargs):
    yield None


class series(_PropsMixin):

    """
    A class for time series concurrent processing on a GPU

    Args:
        filenames (list): The list of filenames to open.
        band_names (Optional[list]): The band associated names.
        crs (Optional[str]): The coordinate reference system.
        res (Optional[list | tuple]): The cell resolution.
        bounds (Optional[object]): The coordinate bounds.
        resampling (Optional[str]): The resampling method.
        nodata (Optional[float | int]): The 'no data' value.
        warp_mem_limit (Optional[int]): The ``rasterio`` warping memory limit (in MB).
        num_threads (Optional[int]): The number of ``rasterio`` warping threads.
        window_size (Optional[list | tuple]): The concurrent processing window size (height, width).

    Requirement:
        > # CUDA 11.1
        > pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
    """

    def __init__(self,
                 filenames: list,
                 band_names: list = None,
                 crs: str = None,
                 res: Union[list, tuple] = None,
                 bounds: BoundingBox = None,
                 resampling: str = 'nearest',
                 nodata: Union[float, int] = 0,
                 warp_mem_limit: int = 256,
                 num_threads: int = 1,
                 window_size: Union[list, tuple] = None):

        if not JAX_INSTALLED:
            logger.exception('Jax must be installed.')
            raise ImportError('Jax must be installed.')

        self.filenames = filenames
        self.band_names = band_names

        self.handler = _Open(filenames)

        self.handler.warp(dst_crs=crs,
                          dst_res=res,
                          dst_bounds=bounds,
                          resampling=resampling,
                          nodata=nodata,
                          warp_mem_limit=warp_mem_limit,
                          num_threads=num_threads,
                          window_size=window_size)

    def read(self,
             bands: Union[int, list],
             window: Window = None,
             gain: float = 1.0,
             offset: Union[float, int] = 0.0) -> jnp.DeviceArray:

        """
        Reads a window
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

        return device_put(np.array([np.stack([_read(vrt, band) for band in band_list]).squeeze()
                                    for vrt in self.handler.vrts_]))

    @staticmethod
    def _create_file(filename, **profile):

        if Path(filename).is_file():
            Path(filename).unlink()

        with rio.open(filename, mode='w', **profile) as dst:
            pass

    def apply(self,
              func: Union[Callable, str, list, tuple],
              outfile: Union[Path, str],
              bands: Union[list, int],
              gain: float = 1.0,
              offset: Union[float, int] = 0.0,
              processes: bool = False,
              num_workers: int = 1,
              monitor_progress: bool = True):

        """
        Applies a function concurrently over windows

        Args:
            func (object | str | list | tuple): The function to apply. If ``func`` is a string,
                choices are ['cv', 'max', 'mean', 'min'].
            outfile (Path | str): The output file.
            bands (list | int): The bands to read.
            gain (Optional[float]): A gain factor to apply.
            offset (Optional[float | int]): An offset factor to apply.
            processes (Optional[bool]): Whether to use process workers, otherwise use threads.
            num_workers (Optional[int]): The number of concurrent workers.
            monitor_progress (Optional[bool]): Whether to monitor progress with a ``tqdm`` bar.

        Returns:
            None, writes to ``outfile``

        Example:
            >>> import itertools
            >>> import geowombat as gw
            >>> import rasterio as rio
            >>>
            >>> # Import an image with 3 bands
            >>> from geowombat.data import l8_224078_20200518
            >>> from geowombat.core.api import TimeModule
            >>>
            >>> # Create a custom class
            >>> class TemporalMean(TimeModule):
            >>>
            >>>     def __init__(self):
            >>>         super(TemporalMean, self).__init__()
            >>>
            >>>     # The main function
            >>>     def run(self, w, array, band_dict=None):
            >>>
            >>>         sl1 = (slice(0, None), slice(band_dict['red'], band_dict['red']+1), slice(0, None), slice(0, None))
            >>>         sl2 = (slice(0, None), slice(band_dict['green'], band_dict['green']+1), slice(0, None), slice(0, None))
            >>>
            >>>         vi = (array[sl1] - array[sl2]) / ((array[sl1] + array[sl2]) + 1e-9)
            >>>
            >>>         # The ``run`` function method must always return the (window, array results).
            >>>         return w, vi.mean(axis=0).squeeze()
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
            >>>               outfile='vi_mean.tif',
            >>>               bands=-1,             # open all bands
            >>>               gain=0.0001,          # scale from [0,10000] -> [0,1]
            >>>               processes=False,      # use threads
            >>>               num_workers=4)        # use 4 concurrent threads, one per window
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
            >>>               outfile='red_mean.tif',
            >>>               bands=1,              # open all bands
            >>>               gain=0.0001,          # scale from [0,10000] -> [0,1]
            >>>               num_workers=4)        # use 4 concurrent threads, one per window
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
            >>>     src.apply(['mean', 'max', 'cv'],    # run multiple statistics
            >>>               outfile='stack_mean.tif',
            >>>               bands=1,                  # open all bands
            >>>               gain=0.0001,              # scale from [0,10000] -> [0,1]
            >>>               num_workers=4)            # use 4 concurrent threads, one per window
        """

        pool = concurrent.futures.ProcessPoolExecutor if processes else concurrent.futures.ThreadPoolExecutor

        tqdm_obj = tqdm if monitor_progress else _tqdm

        if isinstance(func, str) or isinstance(func, list) or isinstance(func, tuple):

            if self.count > 1:
                logger.exception('Only single-band images can be used with built-in functions.')
                raise ValueError('Only single-band images can be used with built-in functions.')

            apply_func_ = TimeSeriesStats(func)

        else:
            apply_func_ = func

        profile = {'count': apply_func_.count,
                   'width': self.width,
                   'height': self.height,
                   'crs': self.crs,
                   'transform': self.transform,
                   'driver': 'GTiff',
                   'dtype': apply_func_.dtype,
                   'compress': apply_func_.compress,
                   'sharing': False,
                   'tiled': True,
                   'blockxsize': self.blockxsize,
                   'blockysize': self.blockysize}

        # Create the file
        self._create_file(outfile, **profile)

        with rio.open(outfile, mode='r+', sharing=False) as dst:

            with pool(num_workers) as executor:

                data_gen = ((w, self.read(bands, window=w, gain=gain, offset=offset), self.band_dict)
                            for w in self.handler.windows_)

                for w, res in tqdm_obj(executor.map(apply_func_, data_gen), total=self.nchunks):

                    with threading.Lock():

                        dst.write(res,
                                  indexes=1 if apply_func_.count == 1 else range(1, apply_func_.count+1),
                                  window=w)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):

        for src in self.handler.srcs_:
            src.close()

        for vrt in self.handler.vrts_:
            vrt.close()

        self.handler = None
