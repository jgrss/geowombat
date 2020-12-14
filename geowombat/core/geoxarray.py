from ..config import config

from . import to_raster, to_netcdf, to_vrt, array_to_polygon, moving, extract, sample, calc_area, subset, clip, mask, replace, recode
from . import dask_to_xarray, ndarray_to_xarray
from . import norm_diff as gw_norm_diff
from . import avi as gw_avi
from . import evi as gw_evi
from . import evi2 as gw_evi2
from . import nbr as gw_nbr
from . import ndvi as gw_ndvi
from . import wi as gw_wi
from . import tasseled_cap as gw_tasseled_cap
from . import transform_crs as _transform_crs
from .properties import DataProperties as _DataProperties
from .util import project_coords, n_rows_cols
from ..backends import Cluster as _Cluster
from ..util import imshow as gw_imshow
from ..radiometry import BRDF as _BRDF

import numpy as np
import xarray as xr
import dask.array as da
from rasterio.windows import Window as _Window
from shapely.geometry import Polygon as _Polygon
import joblib


class _UpdateConfig(object):

    def _update_attrs(self):

        if self.config:

            for k, v in self.config.items():
                setattr(self, k, v)

    def _update_kwargs(self, **kwargs):

        if self.config:

            for k, v in self.config.items():

                # rasterio.write keyword arguments
                if k not in kwargs:
                    kwargs[k] = v

        return kwargs


@xr.register_dataset_accessor('gw')
class GeoWombatAccessor(_UpdateConfig, _DataProperties):

    def __init__(self, xarray_obj):

        self._obj = xarray_obj
        self.sensor = None
        self.ax = None

        self.config = config

        self._update_attrs()

    def imshow(self,
               variable='bands',
               band_names=None,
               mask=False,
               nodata=0,
               flip=False,
               text_color='black',
               rot=30,
               **kwargs):

        """
        Shows an image on a plot

        Args:
            variable (Optional[str]): The ``Dataset`` variable to write.
            band_names (Optional[list or str]): The band name or list of band names to plot.
            mask (Optional[bool]): Whether to mask 'no data' values (given by ``nodata``).
            nodata (Optional[int or float]): The 'no data' value.
            flip (Optional[bool]): Whether to flip an RGB array's band order.
            text_color (Optional[str]): The text color.
            rot (Optional[int]): The degree rotation for the x-axis tick labels.
            kwargs (Optional[dict]): Keyword arguments passed to ``xarray.plot.imshow``.

        Returns:
            None

        Examples:
            >>> with gw.open('image.tif', return_as='dataset') as ds:
            >>>     ds.gw.imshow(band_names=['red', 'green', 'red'], mask=True, vmin=0.1, vmax=0.9, robust=True)
        """

        gw_imshow(self._obj[variable],
                  band_names=band_names,
                  mask=mask,
                  nodata=nodata,
                  flip=flip,
                  text_color=text_color,
                  rot=rot,
                  **kwargs)

    def to_raster(self,
                  filename,
                  variable='bands',
                  verbose=0,
                  overwrite=False,
                  driver='GTiff',
                  gdal_cache=512,
                  dtype=None,
                  nodata=None,
                  tags=None,
                  **kwargs):

        """
        Writes an Xarray Dataset to a raster file

        Args:
            filename (str): The output file name to write to.
            variable (Optional[str]): The ``Dataset`` variable to write.
            filename (str): The output file name to write to.
            n_jobs (Optional[str]): The number of parallel chunks to write.
            verbose (Optional[int]): The verbosity level.
            overwrite (Optional[bool]): Whether to overwrite an existing file.
            driver (Optional[str]): The raster driver.
            gdal_cache (Optional[int]): The ``GDAL`` cache size (in MB).
            dtype (Optional[int]): The output data type.
            row_chunks (Optional[int]): The processing row chunk size.
            col_chunks (Optional[int]): The processing column chunk size.
            pool_chunksize (Optional[int]): The `multiprocessing.Pool` chunk size.
            nodata (Optional[int]): A 'no data' value.
            tags (Optional[dict]): Image tags to write to file.
            kwargs (Optional[dict]): Additional keyword arguments to pass to ``rasterio.write``.

        Returns:
            None

        Examples:
            >>> import geowombat as gw
            >>> from geowombat.backends import Cluster
            >>>
            >>> cluster = Cluster(n_workers=4,
            >>>                   threads_per_worker=2,
            >>>                   scheduler_port=0,
            >>>                   processes=False)
            >>>
            >>> cluster.start()
            >>>
            >>> with gw.open('input.tif') as ds:
            >>>     ds.gw.to_raster('output.tif', n_jobs=8)
            >>>
            >>> cluster.stop()
        """

        if not hasattr(self._obj, 'crs'):
            raise AttributeError('The Dataset does not have a `crs` attribute.')

        if not hasattr(self._obj, 'transform'):
            raise AttributeError('The Dataset does not have a `transform` attribute.')

        kwargs = self._update_kwargs(nodata=nodata,
                                     driver=driver,
                                     **kwargs)

        to_raster(self._obj[variable],
                  filename,
                  gdal_cache=gdal_cache,
                  verbose=verbose,
                  overwrite=overwrite,
                  crs=self._obj.crs,
                  transform=self._obj.transform,
                  dtype=dtype,
                  tags=tags,
                  **kwargs)

    def moving(self,
               variable='bands',
               band_coords='band',
               stat='mean',
               perc=50.0,
               w=3,
               n_jobs=1):

        """
        Applies a moving window function to the DataArray

        Args:
            variable (Optional[str]): The variable to compute.
            band_coords (Optional[str]): The band coordinate name.
            stat (Optional[str]): The statistic to compute. Choices are ['mean', 'std', 'var', 'min', 'max'].
            perc (Optional[float]): The percentile to return if ``stat`` = 'perc'.
            w (Optional[int]): The moving window size.
            n_jobs (Optional[int]): The number of bands to process in parallel.

        Returns:
            DataArray
        """

        return moving(self._obj[variable],
                      band_names=self._obj.coords[band_coords].values,
                      w=w,
                      perc=perc,
                      stat=stat,
                      n_jobs=n_jobs)

    def norm_diff(self, b1, b2, variable='bands', nodata=None, mask=False, sensor=None, scale_factor=1.0):
        return gw_norm_diff(self._obj[variable], b1, b2, sensor=sensor, nodata=nodata, mask=mask, scale_factor=scale_factor)

    def evi(self, variable='bands', nodata=None, mask=False, sensor=None, scale_factor=1.0):
        return gw_evi(self._obj[variable], nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def evi2(self, variable='bands', nodata=None, mask=False, sensor=None, scale_factor=1.0):
        return gw_evi2(self._obj[variable], nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def nbr(self, variable='bands', nodata=None, mask=False, sensor=None, scale_factor=1.0):
        return gw_nbr(self._obj[variable], nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def ndvi(self, variable='bands', nodata=None, mask=False, sensor=None, scale_factor=1.0):
        return gw_ndvi(self._obj[variable], nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def wi(self, variable='bands', nodata=None, mask=False, sensor=None, scale_factor=1.0):
        return gw_wi(self._obj[variable], nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def tasseled_cap(self, variable='bands', nodata=None, sensor=None, scale_factor=1.0):
        return gw_tasseled_cap(self._obj[variable], nodata=nodata, sensor=sensor, scale_factor=scale_factor)


@xr.register_dataarray_accessor('gw')
class GeoWombatAccessor(_UpdateConfig, _DataProperties):

    """
    A method to access an ``xarray.DataArray``. This class is typically not accessed directly, but rather
    through a call to ``geowombat.open``.

    - A DataArray object will have a ``gw`` method.
    - To access GeoWombat methods, use ``xarray.DataArray.gw``.
    """

    def __init__(self, xarray_obj):

        self._obj = xarray_obj
        self.config = config

        self._update_attrs()

    @property
    def filenames(self):
        """Gets the data filenames"""
        return self._filenames if hasattr(self, '_filenames') else []

    @filenames.setter
    def filenames(self, file_list):
        self._filenames = file_list

    @property
    def data_are_separate(self):
        """Checks whether the data are loaded separately"""
        return True if hasattr(self, '_stack_dim') and (self._stack_dim in ['band', 'time']) else False

    @data_are_separate.setter
    def data_are_separate(self, dim):
        self._stack_dim = dim
        
    @property
    def data_are_stacked(self):
        """Checks whether the data are stacked"""
        return self._sbool if hasattr(self, '_sbool') else False

    @data_are_stacked.setter
    def data_are_stacked(self, sbool):
        self._sbool = sbool

    def compute(self, **kwargs):

        if not self._obj.chunks:
            return self._obj.data
        else:
            return self._obj.data.compute(**kwargs)

    def match_data(self, data, band_names):

        """
        Coerces the DataArray to match another GeoWombat DataArray

        Args:
            data (DataArray): The ``xarray.DataArray`` to match to.
            band_names (1d array-like): The output band names.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>> import xarray as xr
            >>>
            >>> other_array = xr.DataArray()
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     new_array = other_array.gw.match_data(src, ['bd1'])
        """

        if isinstance(self._obj.data, da.Array):

            if len(self._obj.shape) == 2:
                new_chunks = (data.gw.row_chunks, data.gw.col_chunks)
            else:
                new_chunks = (1, data.gw.row_chunks, data.gw.col_chunks)

            return dask_to_xarray(data, self._obj.data.rechunk(new_chunks), band_names)
        else:
            return ndarray_to_xarray(data, self._obj.data, band_names)

    def compare(self, op, b, return_binary=False):

        """
        Comparison operation

        Args:
            op (str): The comparison operation.
            b (int | float): The value to compare to.
            return_binary (Optional[bool]): Whether to return a binary (1 or 0) array.

        Returns:
            ``xarray.DataArray``:
                Valid data where ``op`` meets criteria ``b``, otherwise nans
        """

        if op not in ['lt', 'le', 'gt', 'ge', 'eq', 'ne']:
            raise NameError('The comparison operation is not supported.')

        if op == 'lt':
            out = self._obj.where(self._obj < b)
        elif op == 'le':
            out = self._obj.where(self._obj <= b)
        elif op == 'gt':
            out = self._obj.where(self._obj > b)
        elif op == 'ge':
            out = self._obj.where(self._obj >= b)
        elif op == 'eq':
            out = self._obj.where(self._obj == b)
        elif op == 'ne':
            out = self._obj.where(self._obj != b)

        if return_binary:
            out = xr.where(out > 0, 1, np.nan)

        return out.astype(self._obj.dtype).assign_attrs(**self._obj.attrs.copy())

    def replace(self, to_replace):

        """
        Replace values given in to_replace with value.

        Args:
            to_replace (dict): How to find the values to replace. Dictionary mappings should be given
                as {from: to} pairs. If ``to_replace`` is an integer/string mapping, the to string should be 'mode'.

                {1: 5}:
                    recode values of 1 to 5

                {1: 'mode'}:
                    recode values of 1 to the polygon mode

        Returns:
            ``xarray.DataArray``
        """

        return replace(self._obj,
                       to_replace)

    def recode(self,
               polygon,
               to_replace,
               num_workers=1):

        """
        Recodes a DataArray with polygon mappings

        Args:
            polygon (GeoDataFrame | str): The ``geopandas.DataFrame`` or file with polygon geometry.
            to_replace (dict): How to find the values to replace. Dictionary mappings should be given
                as {from: to} pairs. If ``to_replace`` is an integer/string mapping, the to string should be 'mode'.

                {1: 5}:
                    recode values of 1 to 5

                {1: 'mode'}:
                    recode values of 1 to the polygon mode
            num_workers (Optional[int]): The number of parallel Dask workers (only used if ``to_replace``
                has a 'mode' mapping).

        Returns:
            ``xarray.DataArray``
        """

        return recode(self._obj,
                      polygon,
                      to_replace,
                      num_workers=num_workers)

    def bounds_overlay(self, bounds, how='intersects'):

        """
        Checks whether the bounds overlay the image bounds

        Args:
            bounds (tuple | rasterio.coords.BoundingBox | shapely.geometry): The bounds to check. If given as a tuple,
                the order should be (left, bottom, right, top).
            how (Optional[str]): Choices are any ``shapely.geometry`` binary predicates.

        Returns:
            ``bool``

        Example:
            >>> import geowombat as gw
            >>>
            >>> bounds = (left, bottom, right, top)
            >>>
            >>> with gw.open('image.tif') as src
            >>>     intersects = src.gw.bounds_overlay(bounds)
            >>>
            >>> from rasterio.coords import BoundingBox
            >>>
            >>> bounds = BoundingBox(left, bottom, right, top)
            >>>
            >>> with gw.open('image.tif') as src
            >>>     contains = src.gw.bounds_overlay(bounds, how='contains')
        """

        if isinstance(bounds, _Polygon):
            return getattr(self._obj.gw.geometry, how)(bounds)
        else:

            left, bottom, right, top = bounds

            poly = _Polygon([(left, bottom),
                             (left, top),
                             (right, top),
                             (right, bottom),
                             (left, bottom)])

            return getattr(self._obj.gw.geometry, how)(poly)

    def n_windows(self, row_chunks=None, col_chunks=None):

        """
        Calculates the number of windows in a row/column iteration

        Args:
            row_chunks (Optional[int]): The row chunk size. If not given, defaults to opened DataArray chunks.
            col_chunks (Optional[int]): The column chunk size. If not given, defaults to opened DataArray chunks.

        Returns:
            ``int``
        """

        rchunks = row_chunks if isinstance(row_chunks, int) else self._obj.gw.row_chunks
        cchunks = col_chunks if isinstance(col_chunks, int) else self._obj.gw.col_chunks

        return len(list(range(0, self._obj.gw.nrows, rchunks))) * len(list(range(0, self._obj.gw.ncols, cchunks)))

    def windows(self, row_chunks=None, col_chunks=None, return_type='window', ndim=2):

        """
        Generates windows for a row/column iteration

        Args:
            row_chunks (Optional[int]): The row chunk size. If not given, defaults to opened DataArray chunks.
            col_chunks (Optional[int]): The column chunk size. If not given, defaults to opened DataArray chunks.
            return_type (Optional[str]): The data to return. Choices are ['data', 'slice', 'window'].
            ndim (Optional[int]): The number of required dimensions if ``return_type`` = 'data' or 'slice'.
        """

        if return_type not in ['data', 'slice', 'window']:
            raise NameError("The return type must be one of 'data', 'slice', or 'window'.")

        rchunks = row_chunks if isinstance(row_chunks, int) else self._obj.gw.row_chunks
        cchunks = col_chunks if isinstance(col_chunks, int) else self._obj.gw.col_chunks

        for row_off in range(0, self._obj.gw.nrows, rchunks):

            height = n_rows_cols(row_off, rchunks, self._obj.gw.nrows)

            for col_off in range(0, self._obj.gw.ncols, cchunks):

                width = n_rows_cols(col_off, cchunks, self._obj.gw.ncols)

                if return_type == 'data':

                    if ndim == 2:
                        yield self._obj[row_off:row_off+height, col_off:col_off+width]
                    else:
                        slicer = tuple([slice(0, None)] * (ndim-2)) + (slice(row_off, row_off+height), slice(col_off, col_off+width))
                        yield self._obj[slicer]

                elif return_type == 'slice':

                    if ndim == 2:
                        yield (slice(row_off, row_off+height), slice(col_off, col_off+width))
                    else:
                        yield tuple([slice(0, None)] * (ndim-2)) + (slice(row_off, row_off+height), slice(col_off, col_off+width))

                elif return_type == 'window':

                    yield _Window(row_off=row_off,
                                  col_off=col_off,
                                  height=height,
                                  width=width)

    def imshow(self,
               mask=False,
               nodata=0,
               flip=False,
               text_color='black',
               rot=30,
               **kwargs):

        """
        Shows an image on a plot

        Args:
            mask (Optional[bool]): Whether to mask 'no data' values (given by ``nodata``).
            nodata (Optional[int or float]): The 'no data' value.
            flip (Optional[bool]): Whether to flip an RGB array's band order.
            text_color (Optional[str]): The text color.
            rot (Optional[int]): The degree rotation for the x-axis tick labels.
            kwargs (Optional[dict]): Keyword arguments passed to ``xarray.plot.imshow``.

        Returns:
            None

        Examples:
            >>> with gw.open('image.tif') as ds:
            >>>     ds.gw.imshow(band_names=['red', 'green', 'red'], mask=True, vmin=0.1, vmax=0.9, robust=True)
        """

        gw_imshow(self._obj,
                  mask=mask,
                  nodata=nodata,
                  flip=flip,
                  text_color=text_color,
                  rot=rot,
                  **kwargs)

    def to_polygon(self, mask=None, connectivity=4):

        """
        Converts a ``dask`` array to a ``GeoDataFrame``

        Args:
            mask (Optional[numpy ndarray or rasterio Band object]): Must evaluate to bool (rasterio.bool_ or rasterio.uint8).
                Values of False or 0 will be excluded from feature generation. Note well that this is the inverse sense from
                Numpy's, where a mask value of True indicates invalid data in an array. If source is a Numpy masked array
                and mask is None, the source's mask will be inverted and used in place of mask.
            connectivity (Optional[int]): Use 4 or 8 pixel connectivity for grouping pixels into features.

        Returns:
            ``GeoDataFrame``

        Example:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as src:
            >>>
            >>>     # Convert the input image to a GeoDataFrame
            >>>     df = src.gw.to_polygon(mask='source',
            >>>                            num_workers=8)
        """

        return array_to_polygon(self._obj,
                                mask=mask,
                                connectivity=connectivity)

    def to_vector(self, filename, mask=None, connectivity=4):

        """
        Writes an Xarray DataArray to a vector file

        Args:
            filename (str): The output file name to write to.
            mask (numpy ndarray or rasterio Band object, optional): Must evaluate to bool (rasterio.bool_ or rasterio.uint8).
                Values of False or 0 will be excluded from feature generation. Note well that this is the inverse sense from
                Numpy's, where a mask value of True indicates invalid data in an array. If source is a Numpy masked array
                and mask is None, the source's mask will be inverted and used in place of mask.
            connectivity (Optional[int]): Use 4 or 8 pixel connectivity for grouping pixels into features.

        Returns:
            None
        """

        self.to_polygon(mask=mask,
                        connectivity=connectivity)\
                .to_file(filename)

    def transform_crs(self,
                      dst_crs=None,
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
            dst_crs (Optional[CRS | int | dict | str]): The destination CRS.
            dst_res (Optional[tuple]): The destination resolution.
            dst_width (Optional[int]): The destination width. Cannot be used with ``dst_res``.
            dst_height (Optional[int]): The destination height. Cannot be used with ``dst_res``.
            dst_bounds (Optional[BoundingBox | tuple]): The destination bounds, as a ``rasterio.coords.BoundingBox``
                or as a tuple of (left, bottom, right, top).
            resampling (Optional[str]): The resampling method if ``filename`` is a ``list``.
                Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
            warp_mem_lim    it (Optional[int]): The warp memory limit.
            num_threads (Optional[int]): The number of parallel threads.

        Returns:
            ``xarray.DataArray``

        Example:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     dst = src.gw.transform_crs(4326)
        """

        return _transform_crs(self._obj,
                              dst_crs=dst_crs,
                              dst_res=dst_res,
                              dst_width=dst_width,
                              dst_height=dst_height,
                              dst_bounds=dst_bounds,
                              resampling=resampling,
                              warp_mem_limit=warp_mem_limit,
                              num_threads=num_threads)

    def to_netcdf(self, filename, *args, **kwargs):

        """
        Writes an Xarray DataArray to a NetCDF file

        Args:
            filename (str): The output file name to write to.
            args (DataArray): Additional ``DataArrays`` to stack.
            kwargs (dict): Encoding arguments.

        Example:
            >>> import geowombat as gw
            >>> import xarray as xr
            >>>
            >>> # Write a single DataArray to a .nc file
            >>> with gw.config.update(sensor='l7'):
            >>>     with gw.open('LC08_L1TP_225078_20200219_20200225_01_T1.tif') as src:
            >>>         src.gw.to_netcdf('filename.nc', zlib=True, complevel=5)
            >>>
            >>> # Add extra layers
            >>> with gw.config.update(sensor='l7'):
            >>>     with gw.open('LC08_L1TP_225078_20200219_20200225_01_T1.tif') as src, \
            >>>         gw.open('LC08_L1TP_225078_20200219_20200225_01_T1_angles.tif', band_names=['zenith', 'azimuth']) as ang:
            >>>
            >>>         src = xr.where(src == 0, -32768, src)\
            >>>                     .astype('int16')\
            >>>                     .assign_attrs(**src.attrs)
            >>>
            >>>         src.gw.to_netcdf('filename.nc', ang.astype('int16'), zlib=True, complevel=5, _FillValue=-32768)
            >>>
            >>> # Open the data and convert to a DataArray
            >>> with xr.open_dataset('filename.nc', engine='h5netcdf', chunks=256) as ds:
            >>>     src = ds.to_array(dim='band')
        """

        to_netcdf(self._obj, filename, *args, **kwargs)

    def to_raster(self,
                  filename,
                  readxsize=None,
                  readysize=None,
                  separate=False,
                  use_dask_store=False,
                  out_block_type='gtiff',
                  keep_blocks=False,
                  verbose=0,
                  overwrite=False,
                  gdal_cache=512,
                  scheduler='processes',
                  n_jobs=1,
                  n_workers=None,
                  n_threads=None,
                  n_chunks=None,
                  overviews=False,
                  resampling='nearest',
                  use_client=False,
                  address=None,
                  total_memory=48,
                  driver='GTiff',
                  nodata=None,
                  blockxsize=512,
                  blockysize=512,
                  tags=None,
                  **kwargs):

        """
        Writes an Xarray DataArray to a raster file

        Args:
            filename (str): The output file name to write to.
            readxsize (Optional[int]): The size of column chunks to read. If not given, ``readxsize`` defaults to Dask chunk size.
            readysize (Optional[int]): The size of row chunks to read. If not given, ``readysize`` defaults to Dask chunk size.
            separate (Optional[bool]): Whether to write blocks as separate files. Otherwise, write to a single file.
            use_dask_store (Optional[bool]): Whether to use ``dask.array.store`` to save with Dask task graphs.
            out_block_type (Optional[str]): The output block type. Choices are ['gtiff', 'zarr'].
                Only used if ``separate`` = ``True``.
            keep_blocks (Optional[bool]): Whether to keep the blocks stored on disk. Only used if ``separate`` = ``True``.
            verbose (Optional[int]): The verbosity level.
            overwrite (Optional[bool]): Whether to overwrite an existing file.
            gdal_cache (Optional[int]): The ``GDAL`` cache size (in MB).
            scheduler (Optional[str]): The ``concurrent.futures`` scheduler to use. Choices are ['processes', 'threads'].
            n_jobs (Optional[int]): The total number of parallel jobs.
            n_workers (Optional[int]): The number of processes.
            n_threads (Optional[int]): The number of threads.
            n_chunks (Optional[int]): The chunk size of windows. If not given, equal to ``n_workers`` x 3.
            overviews (Optional[bool or list]): Whether to build overview layers.
            resampling (Optional[str]): The resampling method for overviews when ``overviews`` is ``True`` or a ``list``.
                Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
            use_client (Optional[bool]): Whether to use a ``dask`` client.
            address (Optional[str]): A cluster address to pass to client. Only used when ``use_client`` = ``True``.
            total_memory (Optional[int]): The total memory (in GB) required when ``use_client`` = ``True``.
            driver (Optional[str]): The raster driver.
            nodata (Optional[int]): A 'no data' value.
            blockxsize (Optional[int]): The output x block size. Ignored if ``separate`` = ``True``.
            blockysize (Optional[int]): The output y block size. Ignored if ``separate`` = ``True``.
            tags (Optional[dict]): Image tags to write to file.
            kwargs (Optional[dict]): Additional keyword arguments to pass to ``rasterio.write``.

        Returns:
            None

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Use dask.compute()
            >>> with gw.open('input.tif') as ds:
            >>>     ds.gw.to_raster('output.tif', n_jobs=8)
            >>>
            >>> # Use a dask client
            >>> with gw.open('input.tif') as ds:
            >>>     ds.gw.to_raster('output.tif', use_client=True, n_workers=8, n_threads=4)
            >>>
            >>> # Compress the output
            >>> with gw.open('input.tif') as ds:
            >>>     ds.gw.to_raster('output.tif', n_jobs=8, compress='lzw')
        """

        if not hasattr(self._obj, 'crs'):
            raise AttributeError('The DataArray does not have a `crs` attribute.')

        if not hasattr(self._obj, 'transform'):
            raise AttributeError('The DataArray does not have a `transform` attribute.')

        kwargs = self._update_kwargs(nodata=nodata,
                                     driver=driver,
                                     blockxsize=blockxsize,
                                     blockysize=blockysize,
                                     **kwargs)

        # Keywords for rasterio profile
        if 'crs' not in kwargs:
            kwargs['crs'] = self._obj.crs

        if 'transform' not in kwargs:
            kwargs['transform'] = self._obj.transform

        if 'width' not in kwargs:
            kwargs['width'] = self._obj.gw.ncols

        if 'height' not in kwargs:
            kwargs['height'] = self._obj.gw.nrows

        if 'count' not in kwargs:
            kwargs['count'] = self._obj.gw.nbands

        if 'dtype' not in kwargs:
            kwargs['dtype'] = self._obj.data.dtype.name

        to_raster(self._obj,
                  filename,
                  readxsize=readxsize,
                  readysize=readysize,
                  use_dask_store=use_dask_store,
                  separate=separate,
                  out_block_type=out_block_type,
                  keep_blocks=keep_blocks,
                  verbose=verbose,
                  overwrite=overwrite,
                  gdal_cache=gdal_cache,
                  scheduler=scheduler,
                  n_jobs=n_jobs,
                  n_workers=n_workers,
                  n_threads=n_threads,
                  n_chunks=n_chunks,
                  overviews=overviews,
                  resampling=resampling,
                  use_client=use_client,
                  address=address,
                  total_memory=total_memory,
                  tags=tags,
                  **kwargs)

    def to_vrt(self,
               filename,
               resampling=None,
               nodata=None,
               init_dest_nodata=True,
               warp_mem_limit=128):

        """
        Writes a file to a VRT file

        Args:
            filename (str): The output file name to write to.
            resampling (Optional[object]): The resampling algorithm for ``rasterio.vrt.WarpedVRT``.
            nodata (Optional[float or int]): The 'no data' value for ``rasterio.vrt.WarpedVRT``.
            init_dest_nodata (Optional[bool]): Whether or not to initialize output to ``nodata`` for ``rasterio.vrt.WarpedVRT``.
            warp_mem_limit (Optional[int]): The GDAL memory limit for ``rasterio.vrt.WarpedVRT``.

        Example:
            >>> import geowombat as gw
            >>> from rasterio.enums import Resampling
            >>>
            >>> # Transform a CRS and save to VRT
            >>> with gw.config.update(ref_crs=102033):
            >>>     with gw.open('image.tif') as src:
            >>>         src.gw.to_vrt('output.vrt',
            >>>                       resampling=Resampling.cubic,
            >>>                       warp_mem_limit=256)
            >>>
            >>> # Load multiple files set to a common geographic extent
            >>> bounds = (left, bottom, right, top)
            >>> with gw.config.update(ref_bounds=bounds):
            >>>     with gw.open(['image1.tif', 'image2.tif'], mosaic=True) as src:
            >>>         src.gw.to_vrt('output.vrt')
        """

        to_vrt(self._obj,
               filename,
               resampling=resampling,
               nodata=nodata,
               init_dest_nodata=init_dest_nodata,
               warp_mem_limit=warp_mem_limit)

    def predict(self,
                clf,
                outname=None,
                chunksize='same',
                x_chunks=(5000, 1),
                overwrite=False,
                return_as='array',
                n_jobs=1,
                backend='dask',
                verbose=0,
                nodata=None,
                dtype='uint8',
                gdal_cache=512,
                **kwargs):

        """
        Predicts an image using a pre-fit model

        Args:
            clf (object): A fitted classifier ``geowombat.model.Model`` instance with a ``predict`` method.
            outname (Optional[str]): An file name for the predictions.
            chunksize (Optional[str or tuple]): The chunk size for I/O. Default is 'same', or use the input chunk size.
            x_chunks (Optional[tuple]): The chunk size for the X predictors (or ``data``).
            overwrite (Optional[bool]): Whether to overwrite an existing file.
            return_as (Optional[str]): Whether to return the predictions as a ``xarray.DataArray`` or ``xarray.Dataset``.
                Only relevant if ``outname`` is not given.
            nodata (Optional[int or float]): The 'no data' value in the predictors.
            n_jobs (Optional[int]): The number of parallel jobs (chunks) for writing.
            backend (Optional[str]): The ``joblib`` backend scheduler.
            verbose (Optional[int]): The verbosity level.
            dtype (Optional[str]): The output data type passed to ``rasterio.write``.
            gdal_cache (Optional[int]): The GDAL cache (in MB) passed to ``rasterio.write``.
            kwargs (Optional[dict]): Additional keyword arguments passed to ``rasterio.write``.
                The ``blockxsize`` and ``blockysize`` should be excluded because they are taken from ``chunksize``.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>> from sklearn import ensemble
            >>>
            >>> clf = ensemble.RandomForestClassifier()
            >>> clf.fit(X, y)
            >>>
            >>> with gw.open('image.tif') as ds:
            >>>     pred = ds.gw.predict(clf)
        """

        kwargs = self._update_kwargs(**kwargs)

        return predict(self._obj,
                       clf,
                       outname=outname,
                       chunksize=chunksize,
                       x_chunks=x_chunks,
                       overwrite=overwrite,
                       return_as=return_as,
                       n_jobs=n_jobs,
                       backend=backend,
                       verbose=verbose,
                       nodata=nodata,
                       dtype=dtype,
                       gdal_cache=gdal_cache,
                       **kwargs)

    def apply(self, filename, user_func, n_jobs=1, **kwargs):

        """
        Applies a user function to an Xarray Dataset or DataArray and writes to file

        Args:
            filename (str): The output file name to write to.
            user_func (func): The user function to apply.
            n_jobs (Optional[int]): The number of parallel jobs for the cluster.
            kwargs (Optional[dict]): Keyword arguments passed to `to_raster`.

        Example:
            >>> import geowombat as gw
            >>> import xarray as xr
            >>>
            >>> def user_func(ds_):
            >>>     return ds_.max(axis=0)
            >>>
            >>> with xr.open_rasterio('image.tif', chunks=(1, 512, 512)) as ds:
            >>>     ds.gw.apply('output.tif', user_func, n_jobs=8, overwrite=True, blockxsize=512, blockysize=512)
        """

        cluster = _Cluster(n_workers=n_jobs,
                           threads_per_worker=1,
                           scheduler_port=0,
                           processes=False)

        cluster.start()

        with joblib.parallel_backend('loky', n_jobs=n_jobs):

            ds_sub = user_func(self._obj)
            ds_sub.attrs = self._obj.attrs
            ds_sub.gw.to_raster(filename, n_jobs=n_jobs, **kwargs)

        cluster.stop()

    def clip(self, df, query=None, mask_data=False):

        """
        Clips a DataArray

        Args:
            df (GeoDataFrame): The ``geopandas.GeoDataFrame`` to clip to.
            query (Optional[str]): A query to apply to ``df``.
            mask_data (Optional[bool]): Whether to mask values outside of the ``df`` geometry envelope.

        Returns:
             ``xarray.DataArray``
        """

        return clip(self._obj, df, query=query, mask_data=mask_data)

    def mask(self, df, query=None, keep='in'):

        """
        Masks a DataArray

        Args:
            df (GeoDataFrame or str): The ``geopandas.GeoDataFrame`` or filename to use for masking.
            query (Optional[str]): A query to apply to ``df``.
            keep (Optional[str]): If ``keep`` = 'in', mask values outside of the geometry (keep inside).
                Otherwise, if ``keep`` = 'out', mask values inside (keep outside).

        Returns:
             ``xarray.DataArray``
        """

        return mask(self._obj, df, query=query, keep=keep)

    def subset(self,
               left=None,
               top=None,
               right=None,
               bottom=None,
               rows=None,
               cols=None,
               center=False,
               mask_corners=False):

        """
        Subsets a DataArray

        Args:
            left (Optional[float]): The left coordinate.
            top (Optional[float]): The top coordinate.
            right (Optional[float]): The right coordinate.
            bottom (Optional[float]): The bottom coordinate.
            rows (Optional[int]): The number of output rows.
            cols (Optional[int]): The number of output rows.
            center (Optional[bool]): Whether to center the subset on ``left`` and ``top``.
            mask_corners (Optional[bool]): Whether to mask corners (requires ``pymorph``).
            chunksize (Optional[tuple]): A new chunk size for the output.

        Returns:
            ``xarray.DataArray``

        Example:
            >>> geowombat as gw
            >>>
            >>> with gw.open('image.tif', chunks=(1, 512, 512)) as ds:
            >>>     ds_sub = ds.gw.subset(-263529.884, 953985.314, rows=2048, cols=2048)
        """

        return subset(self._obj,
                      left=left,
                      top=top,
                      right=right,
                      bottom=bottom,
                      rows=rows,
                      cols=cols,
                      center=center,
                      mask_corners=mask_corners)

    def calc_area(self,
                  values,
                  op='eq',
                  units='km2',
                  row_chunks=None,
                  col_chunks=None,
                  n_workers=1,
                  n_threads=1,
                  scheduler='threads',
                  n_chunks=100):

        """
        Calculates the area of data values

        Args:
            values (list): A list of values.
            op (Optional[str]): The value sign. Choices are ['gt', 'ge', 'lt', 'le', 'eq'].
            units (Optional[str]): The units to return. Choices are ['km2', 'ha'].
            row_chunks (Optional[int]): The row chunk size to process in parallel.
            col_chunks (Optional[int]): The column chunk size to process in parallel.
            n_workers (Optional[int]): The number of parallel workers for ``scheduler``.
            n_threads (Optional[int]): The number of parallel threads for ``dask.compute()``.
            scheduler (Optional[str]): The parallel task scheduler to use. Choices are ['processes', 'threads', 'mpool'].

                mpool: process pool of workers using ``multiprocessing.Pool``
                processes: process pool of workers using ``concurrent.futures``
                threads: thread pool of workers using ``concurrent.futures``

            n_chunks (Optional[int]): The chunk size of windows. If not given, equal to ``n_workers`` x 50.

        Returns:
            ``pandas.DataFrame``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Read a land cover image with 512x512 chunks
            >>> with gw.open('land_cover.tif', chunks=512) as src:
            >>>
            >>>     df = src.gw.calc_area([1, 2, 5],        # calculate the area of classes 1, 2, and 5
            >>>                           units='km2',      # return area in kilometers squared
            >>>                           n_workers=4,
            >>>                           row_chunks=1024,  # iterate over larger chunks to use 512 chunks in parallel
            >>>                           col_chunks=1024)
        """

        return calc_area(self._obj,
                         values,
                         op=op,
                         units=units,
                         row_chunks=row_chunks,
                         col_chunks=col_chunks,
                         n_workers=n_workers,
                         n_threads=n_threads,
                         scheduler=scheduler,
                         n_chunks=n_chunks)

    def sample(self,
               method='random',
               band=None,
               n=None,
               strata=None,
               spacing=None,
               min_dist=None,
               max_attempts=10,
               **kwargs):

        """
        Generates samples from a raster

        Args:
            data (DataArray): The ``xarray.DataArray`` to extract data from.
            method (Optional[str]): The sampling method. Choices are ['random', 'systematic'].
            band (Optional[int or str]): The band name to extract from. Only required if ``method`` = 'random' and ``strata`` is given.
            n (Optional[int]): The total number of samples. Only required if ``method`` = 'random'.
            strata (Optional[dict]): The strata to sample within. The dictionary key-->value pairs should be {'conditional,value': proportion}.

                E.g.,

                    strata = {'==,1': 0.5, '>=,2': 0.5}
                    ... would sample 50% of total samples within class 1 and 50% of total samples in class >= 2.

                    strata = {'==,1': 10, '>=,2': 20}
                    ... would sample 10 samples within class 1 and 20 samples in class >= 2.

            spacing (Optional[float]): The spacing (in map projection units) when ``method`` = 'systematic'.
            min_dist (Optional[float or int]): A minimum distance allowed between samples. Only applies when ``method`` = 'random'.
            max_attempts (Optional[int]): The maximum numer of attempts to sample points > ``min_dist`` from each other.
            kwargs (Optional[dict]): Keyword arguments passed to ``geowombat.extract``.

        Returns:
            ``geopandas.GeoDataFrame``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Sample 100 points randomly across the image
            >>> with gw.open('image.tif') as ds:
            >>>     df = ds.gw.sample(n=100)
            >>>
            >>> # Sample points systematically (with 10km spacing) across the image
            >>> with gw.open('image.tif') as ds:
            >>>     df = ds.gw.sample(method='systematic', spacing=10000.0)
            >>>
            >>> # Sample 50% of 100 in class 1 and 50% in classes >= 2
            >>> strata = {'==,1': 0.5, '>=,2': 0.5}
            >>> with gw.open('image.tif') as ds:
            >>>     df = ds.gw.sample(band=1, n=100, strata=strata)
            >>>
            >>> # Specify a per-stratum minimum allowed point distance of 1,000 meters
            >>> with gw.open('image.tif') as ds:
            >>>     df = ds.gw.sample(band=1, n=100, min_dist=1000, strata=strata)
        """

        return sample(self._obj,
                      method=method,
                      band=band,
                      n=n,
                      strata=strata,
                      spacing=spacing,
                      min_dist=min_dist,
                      max_attempts=max_attempts,
                      **kwargs)

    def extract(self,
                aoi,
                bands=None,
                time_names=None,
                band_names=None,
                frac=1.0,
                all_touched=False,
                mask=None,
                n_jobs=8,
                verbose=0,
                **kwargs):

        """
        Extracts data within an area or points of interest. Projections do not
        need to match, as they are handled 'on-the-fly'.

        Args:
            aoi (str or GeoDataFrame): A file or ``geopandas.GeoDataFrame`` to extract data frame.
            bands (Optional[int or 1d array-like]): A band or list of bands to extract.
                If not given, all bands are used. Bands should be GDAL-indexed (i.e., the first band is 1, not 0).
            band_names (Optional[list]): A list of band names. Length should be the same as `bands`.
            time_names (Optional[list]): A list of time names.
            frac (Optional[float]): A fractional subset of points to extract in each polygon feature.
            all_touched (Optional[bool]): The ``all_touched`` argument is passed to ``rasterio.features.rasterize``.
            mask (Optional[GeoDataFrame or Shapely Polygon]): A ``shapely.geometry.Polygon`` mask to subset to.
            n_jobs (Optional[int]): The number of features to rasterize in parallel.
            verbose (Optional[int]): The verbosity level.
            kwargs (Optional[dict]): Keyword arguments passed to ``dask.compute``.

        Returns:
            ``geopandas.GeoDataFrame``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as ds:
            >>>     df = ds.gw.extract('poly.gpkg')
        """

        kwargs = self._update_kwargs(**kwargs)

        return extract(self._obj,
                       aoi,
                       bands=bands,
                       time_names=time_names,
                       band_names=band_names,
                       frac=frac,
                       all_touched=all_touched,
                       mask=mask,
                       n_jobs=n_jobs,
                       verbose=verbose,
                       **kwargs)

    def set_nodata(self, src_nodata, dst_nodata, clip_range, dtype, scale_factor=None):

        """
        Sets 'no data' values in the DataArray

        Args:
            src_noata (int | float): The 'no data' values to replace.
            dst_nodata (int | float): The 'no data' value to set.
            clip_range (tuple): The output clip range.
            dtype (str): The output data type.
            scale_factor (float): A scale factor to apply.

        Returns:
            ``xarray.DataArray``

        Example:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     src = src.gw.set_nodata(0, 65535, (0, 10000), 'uint16')
        """

        if not isinstance(scale_factor, float):
            scale_factor = 1.0

        attrs = self._obj.attrs.copy()

        # Create a 'no data' mask
        mask = self._obj.where(self._obj != src_nodata).count(dim='band').astype('uint8')

        if self._obj.gw.has_time_coord:
            mask = mask.transpose('time', 'y', 'x')

        # Mask the data
        data = xr.where(mask < self._obj.gw.nbands,
                        dst_nodata,
                        (self._obj*scale_factor).clip(clip_range[0], clip_range[1]))
        
        if self._obj.gw.has_time_coord:
            data = data.transpose('time', 'band', 'y', 'x').astype(dtype)
        else:
            data = data.transpose('band', 'y', 'x').astype(dtype)

        attrs['nodatavals'] = tuple([dst_nodata] * self._obj.gw.nbands)

        return data.assign_attrs(**attrs)

    def moving(self,
               band_coords='band',
               stat='mean',
               perc=50,
               nodata=None,
               w=3,
               weights=False,
               n_jobs=1):

        """
        Applies a moving window function to the DataArray

        Args:
            band_coords (Optional[str]): The band coordinate name.
            stat (Optional[str]): The statistic to compute. Choices are ['mean', 'std', 'var', 'min', 'max', 'perc'].
            perc (Optional[int]): The percentile to return if ``stat`` = 'perc'.
            nodata (Optional[int or float]): A 'no data' value to ignore.
            w (Optional[int]): The moving window size (in pixels).
            weights (Optional[bool]): Whether to weight values by distance from window center.
            n_jobs (Optional[int]): The number of rows to process in parallel.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Calculate the mean within a 5x5 window
            >>> with gw.open('image.tif') as src:
            >>>     res = src.gw.moving(stat='mean', w=5, nodata=32767.0, n_jobs=8)
            >>>
            >>> # Calculate the 90th percentile within a 15x15 window
            >>> with gw.open('image.tif') as src:
            >>>     res = src.gw.moving(stat='perc', w=15, perc=90, nodata=32767.0, n_jobs=8)
        """

        return moving(self._obj,
                      band_names=self._obj.coords[band_coords].values,
                      perc=perc,
                      nodata=nodata,
                      w=w,
                      stat=stat,
                      weights=weights,
                      n_jobs=n_jobs)

    def norm_diff(self, b1, b2, nodata=None, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the normalized difference band ratio

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            b1 (str): The band name of the first band.
            b2 (str): The band name of the second band.
            sensor (Optional[str]): sensor (Optional[str]): The data's sensor.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::
                {norm}_{diff} = \frac{b2 - b1}{b2 + b1}

        Returns:

            ``xarray.DataArray``:

                Data range: -1 to 1
        """

        return gw_norm_diff(self._obj, b1, b2, sensor=sensor, nodata=nodata, mask=mask, scale_factor=scale_factor)

    def avi(self, nodata=None, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the advanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::

                AVI = {(NIR \times (1.0 - red) \times (NIR - red))}^{0.3334}

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 1
        """

        return gw_avi(self._obj, nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def evi(self, nodata=None, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the enhanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::

                EVI = 2.5 \times \frac{NIR - red}{NIR \times 6 \times red - 7.5 \times blue + 1}

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 1
        """

        return gw_evi(self._obj, nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def evi2(self, nodata=None, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the two-band modified enhanced vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::

                EVI2 = 2.5 \times \frac{NIR - red}{NIR + 1 + 2.4 \times red}

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 1
        """

        return gw_evi2(self._obj, nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def nbr(self, nodata=None, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the normalized burn ratio

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::
                NBR = \frac{NIR - SWIR1}{NIR + SWIR1}

        Returns:

            ``xarray.DataArray``:

                Data range: -1 to 1
        """

        return gw_nbr(self._obj, nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def ndvi(self, nodata=None, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the normalized difference vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::
                NDVI = \frac{NIR - red}{NIR + red}

        Returns:

            ``xarray.DataArray``:

                Data range: -1 to 1
        """

        return gw_ndvi(self._obj, nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def wi(self, nodata=None, mask=False, sensor=None, scale_factor=1.0):

        r"""
        Calculates the woody vegetation index

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[bool]): Whether to mask the results.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Equation:

            .. math::

                WI = \Biggl \lbrace
                {
                0,\text{ if }
                   { red + SWIR1 \ge 0.5 }
                \atop
                1 - \frac{red + SWIR1}{0.5}, \text{ otherwise }
                }

        Returns:

            ``xarray.DataArray``:

                Data range: 0 to 1
        """

        return gw_wi(self._obj, nodata=nodata, mask=mask, sensor=sensor, scale_factor=scale_factor)

    def tasseled_cap(self, nodata=None, sensor=None, scale_factor=1.0):

        r"""
        Applies a tasseled cap transformation

        Args:
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            sensor (Optional[str]): The data's sensor.
            scale_factor (Optional[float]): A scale factor to apply to the data.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> with gw.config.update(sensor='qb', scale_factor=0.0001):
            >>>     with gw.open('image.tif', band_names=['blue', 'green', 'red', 'nir']) as ds:
            >>>         tcap = ds.gw.tasseled_cap()
        """

        return gw_tasseled_cap(self._obj, nodata=nodata, sensor=sensor, scale_factor=scale_factor)

    def norm_brdf(self,
                  solar_za,
                  solar_az,
                  sensor_za,
                  sensor_az,
                  sensor=None,
                  wavelengths=None,
                  nodata=None,
                  mask=None,
                  scale_factor=1.0,
                  scale_angles=True):

        """
        Applies Bidirectional Reflectance Distribution Function (BRDF) normalization

        Args:
            solar_za (2d DataArray): The solar zenith angles (degrees).
            solar_az (2d DataArray): The solar azimuth angles (degrees).
            sensor_za (2d DataArray): The sensor azimuth angles (degrees).
            sensor_az (2d DataArray): The sensor azimuth angles (degrees).
            sensor (Optional[str]): The satellite sensor.
            wavelengths (str list): The wavelength(s) to normalize.
            nodata (Optional[int or float]): A 'no data' value to fill NAs with.
            mask (Optional[DataArray]): A data mask, where clear values are 0.
            scale_factor (Optional[float]): A scale factor to apply to the input data.
            scale_angles (Optional[bool]): Whether to scale the pixel angle arrays.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Example where pixel angles are stored in separate GeoTiff files
            >>> with gw.config.update(sensor='l7', scale_factor=0.0001, nodata=0):
            >>>
            >>>     with gw.open('solarz.tif') as solarz,
            >>>         gw.open('solara.tif') as solara,
            >>>             gw.open('sensorz.tif') as sensorz,
            >>>                 gw.open('sensora.tif') as sensora:
            >>>
            >>>         with gw.open('landsat.tif') as ds:
            >>>             ds_brdf = ds.gw.norm_brdf(solarz, solara, sensorz, sensora)
        """

        # Get the central latitude
        central_lat = project_coords(np.array([self._obj.x.values[int(self._obj.x.shape[0] / 2)]], dtype='float64'),
                                     np.array([self._obj.y.values[int(self._obj.y.shape[0] / 2)]], dtype='float64'),
                                     self._obj.crs,
                                     {'init': 'epsg:4326'})[1][0]

        return _BRDF().norm_brdf(self._obj,
                                 solar_za,
                                 solar_az,
                                 sensor_za,
                                 sensor_az,
                                 central_lat,
                                 sensor=sensor,
                                 wavelengths=wavelengths,
                                 nodata=nodata,
                                 mask=mask,
                                 scale_factor=scale_factor,
                                 scale_angles=scale_angles)
