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
import contextlib
import threading

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
import dask
import dask.array as da

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

    @contextlib.contextmanager
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
         nodata=0,
         data_slice=None,
         num_workers=1,
         scheduler='ray'):

    """
    Loads data into memory using ``xarray.open_mfdataset`` and ``ray``. This function does not check data
    alignments and CRSs. It assumes each image in ``image_list`` has the same y and x dimensions and
    that the coordinates align.

    The `load` function cannot be used if `dataclasses` was pip installed.

    Args:
        image_list (list)
        time_names (list)
        band_names (list)
        chunks (Optional[int])
        nodata (Optional[float | int])
        data_slice (Optional[tuple])
        num_workers (Optional[int])
        scheduler (Optional[str]): Currently not implemented.

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

    import dask
    from dask.diagnostics import ProgressBar
    import ray
    from ray.util.dask import ray_dask_get

    with open(f'netcdf:{image_list[0]}' if str(image_list[0]).endswith('.nc') else image_list[0],
              time_names=time_names[0],
              band_names=band_names if not str(image_list[0]).endswith('.nc') else None,
              netcdf_vars=band_names if str(image_list[0]).endswith('.nc') else None,
              chunks=chunks) as src:

        attrs = src.attrs.copy()

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
                        .rename({'variable': 'band'})\
                        .sel(band=band_names)\
                        .assign_coords(y=src.y, x=src.x)\
                        .expand_dims(dim='time')\
                        .astype('float64')\
                        .clip(0, 10000)[data_slice]

        # Scale from [0-10000] -> [0,1]
        darray = xr.where(darray == nodata, 0, darray*0.0001)

        return darray.where(xr.ufuncs.isfinite(darray))\
                        .fillna(0)\
                        .clip(0, 1)

    ray.shutdown()
    ray.init(num_cpus=num_workers)

    with dask.config.set(scheduler=ray_dask_get):

        # Open all arrays and calculate the VI
        ds = xr.open_mfdataset(image_list,
                               concat_dim='time',
                               chunks={'y': chunks, 'x': chunks},
                               combine='nested',
                               engine='h5netcdf',
                               preprocess=expand_time,
                               parallel=True)\
                    .assign_coords(time=time_names)\
                    .groupby('time').max()\
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
