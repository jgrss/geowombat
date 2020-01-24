from ..config import config

from . import to_raster, to_vrt, to_geodataframe, moving, extract, sample, subset, clip, mask
from . import norm_diff as gw_norm_diff
from . import evi as gw_evi
from . import evi2 as gw_evi2
from . import nbr as gw_nbr
from . import ndvi as gw_ndvi
from . import wi as gw_wi
from . import tasseled_cap as gw_tasseled_cap
from .properties import DataProperties as _DataProperties
from .util import project_coords
from ..backends import Cluster as _Cluster
from ..util import imshow as gw_imshow
#from ..models import predict
from ..radiometry import BRDF as _BRDF

import numpy as np
import xarray as xr
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
    Xarray IO class
    """

    def __init__(self, xarray_obj):

        self._obj = xarray_obj
        self.config = config

        self._update_attrs()

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

    def to_geodataframe(self, mask=None, connectivity=4):

        """
        Converts a ``dask`` array to a ``GeoDataFrame``

        Args:
            mask (Optional[numpy ndarray or rasterio Band object]): Must evaluate to bool (rasterio.bool_ or rasterio.uint8).
                Values of False or 0 will be excluded from feature generation. Note well that this is the inverse sense from
                Numpy’s, where a mask value of True indicates invalid data in an array. If source is a Numpy masked array
                and mask is None, the source’s mask will be inverted and used in place of mask.
            connectivity (Optional[int]): Use 4 or 8 pixel connectivity for grouping pixels into features.

        Returns:
            ``GeoDataFrame``
        """

        return to_geodataframe(self._obj,
                               mask=mask,
                               connectivity=connectivity)

    def to_vector(self, filename, mask=None, connectivity=4):

        """
        Writes an Xarray DataArray to a vector file

        Args:
            filename (str): The output file name to write to.
            mask (numpy ndarray or rasterio Band object, optional): Must evaluate to bool (rasterio.bool_ or rasterio.uint8).
                Values of False or 0 will be excluded from feature generation. Note well that this is the inverse sense from
                Numpy’s, where a mask value of True indicates invalid data in an array. If source is a Numpy masked array
                and mask is None, the source’s mask will be inverted and used in place of mask.
            connectivity (Optional[int]): Use 4 or 8 pixel connectivity for grouping pixels into features.

        Returns:
            None
        """

        df_ = self.to_geodataframe(self._obj,
                                   mask=mask,
                                   connectivity=connectivity)

        df_.to_file(filename)

    def to_raster(self,
                  filename,
                  readxsize=None,
                  readysize=None,
                  separate=False,
                  use_dask_store=False,
                  out_block_type='zarr',
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
                  crs=self._obj.crs,
                  transform=self._obj.transform,
                  width=self._obj.gw.ncols,
                  height=self._obj.gw.nrows,
                  count=self._obj.gw.nbands,
                  dtype=self._obj.data.dtype.name,
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
            data (DataArray): The ``xarray.DataArray`` to write.
            filename (str): The output file name to write to.
            resampling (Optional[object]): The resampling algorithm for ``rasterio.vrt.WarpedVRT``.
            nodata (Optional[float or int]): The 'no data' value for ``rasterio.vrt.WarpedVRT``.
            init_dest_nodata (Optional[bool]): Whether or not to initialize output to ``nodata`` for ``rasterio.vrt.WarpedVRT``.
            warp_mem_limit (Optional[int]): The GDAL memory limit for ``rasterio.vrt.WarpedVRT``.

        Example:
            >>> import geowombat as gw
            >>> from rasterio.enums import Resampling
            >>>
            >>> with gw.config.update(ref_crs=102033):
            >>>
            >>>     with gw.open('image.tif') as ds:
            >>>
            >>>         ds.gw.to_vrt('image.vrt',
            >>>                      resampling=Resampling.cubic,
            >>>                      warp_mem_limit=256)
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
            >>>     ds.io.apply('output.tif', user_func, n_jobs=8, overwrite=True, blockxsize=512, blockysize=512)
        """

        cluster = _Cluster(n_workers=n_jobs,
                           threads_per_worker=1,
                           scheduler_port=0,
                           processes=False)

        cluster.start()

        with joblib.parallel_backend('loky', n_jobs=n_jobs):

            ds_sub = user_func(self._obj)
            ds_sub.attrs = self._obj.attrs
            ds_sub.io.to_raster(filename, n_jobs=n_jobs, **kwargs)

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

    def sample(self,
               method='random',
               band=None,
               n=None,
               strata=None,
               spacing=None,
               min_dist=None,
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

            spacing (Optional[float]): The spacing (in map projection units) when ``method`` = 'systematic'.
            min_dist (Optional[float or int]): A minimum distance allowed between samples. Only applies when ``method`` = 'random'.
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

        return extract(self._obj,
                       method=method,
                       band=band,
                       n=n,
                       strata=strata,
                       spacing=spacing,
                       min_dist=min_dist,
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

    def moving(self,
               band_coords='band',
               stat='mean',
               perc=50,
               nodata=None,
               w=3,
               n_jobs=1):

        """
        Applies a moving window function to the DataArray

        Args:
            band_coords (Optional[str]): The band coordinate name.
            stat (Optional[str]): The statistic to compute. Choices are ['mean', 'std', 'var', 'min', 'max', 'perc'].
            perc (Optional[int]): The percentile to return if ``stat`` = 'perc'.
            nodata (Optional[int or float]): A 'no data' value to ignore.
            w (Optional[int]): The moving window size (in pixels).
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
            ``xarray.DataArray``
        """

        return gw_norm_diff(self._obj, b1, b2, sensor=sensor, nodata=nodata, mask=mask, scale_factor=scale_factor)

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
            ``xarray.DataArray``
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
            ``xarray.DataArray``
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
            ``xarray.DataArray``
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
            ``xarray.DataArray``
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
                WI = SWIR1 + red

        Returns:
            ``xarray.DataArray``
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
                  solar_zenith,
                  solar_azimuth,
                  sensor_zenith,
                  sensor_azimuth,
                  sensor=None,
                  wavelengths=None,
                  nodata=None,
                  mask=None,
                  scale_factor=1.0,
                  scale_angles=True):

        r"""
        Applies Bidirectional Reflectance Distribution Function (BRDF) normalization

        Args:
            solar_zenith (DataArray): The solar zenith angles for each pixel.
            solar_azimuth (DataArray): The solar azimuth angles for each pixel.
            sensor_zenith (DataArray): The sensor zenith angles for each pixel.
            sensor_azimuth (DataArray): The sensor azimuth angles for each pixel.
            mask (DataArray): A mask array where 0 values indicate clear sky.
            sensor (Optional[str]): The satellite sensor.
            wavelengths (str list): Choices are ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'].
            nodata (Optional[int or float]): A 'no data' value.
            mask (Optional[bool]): Whether to mask the results.
            scale_factor (Optional[float]): A scale factor to apply to the data.
            scale_angles (Optional[bool]): Whether to scale the pixel angle arrays.

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

        Returns:
            ``xarray.DataArray``
        """

        # Get the central latitude
        central_lat = project_coords(np.array([self._obj.x.values[int(self._obj.x.shape[0] / 2)]], dtype='float64'),
                                     np.array([self._obj.y.values[int(self._obj.y.shape[0] / 2)]], dtype='float64'),
                                     self._obj.crs,
                                     {'init': 'epsg:4326'})[1][0]

        return _BRDF().norm_brdf(self._obj,
                                 solar_zenith,
                                 solar_azimuth,
                                 sensor_zenith,
                                 sensor_azimuth,
                                 central_lat,
                                 sensor=sensor,
                                 wavelengths=wavelengths,
                                 nodata=nodata,
                                 mask=mask,
                                 scale_factor=scale_factor,
                                 scale_angles=scale_angles)
