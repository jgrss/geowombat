from ..config import config

from . import to_raster, moving, extract, subset
from . import norm_diff, evi, evi2, nbr, ndvi, wi
from ..errors import logger
from ..util import Cluster, DataProperties
from ..models import predict

import xarray as xr
import joblib

import matplotlib.pyplot as plt
import matplotlib as mpl


class _UpdateConfig(object):

    def _update_attrs(self):

        if self.config:

            for k, v in self.config.items():
                setattr(self, k, v)

    def _update_kwargs(self, **kwargs):

        if self.config:

            for k, v in self.config.items():

                # rasterio.write keyword arguments
                if k in kwargs:
                    kwargs[k] = v

        return kwargs


@xr.register_dataset_accessor('gw')
class GeoWombatAccessor(_UpdateConfig, DataProperties):

    def __init__(self, xarray_obj):

        self._obj = xarray_obj
        self.sensor = None
        self.ax = None

        self.config = config

        self._update_attrs()

    def to_raster(self,
                  filename,
                  variable='bands',
                  n_jobs=1,
                  verbose=0,
                  overwrite=False,
                  driver='GTiff',
                  gdal_cache=512,
                  dtype=None,
                  row_chunks=None,
                  col_chunks=None,
                  pool_chunksize=10,
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
        """

        if not hasattr(self._obj, 'crs'):
            raise AttributeError('The Dataset does not have a `crs` attribute.')

        if not hasattr(self._obj, 'transform'):
            raise AttributeError('The Dataset does not have a `transform` attribute.')

        kwargs = self._update_kwargs(**kwargs)

        to_raster(self._obj[variable],
                  filename,
                  self._obj.crs,
                  self._obj.transform,
                  driver=driver,
                  n_jobs=n_jobs,
                  gdal_cache=gdal_cache,
                  dtype=dtype,
                  row_chunks=row_chunks,
                  col_chunks=col_chunks,
                  pool_chunksize=pool_chunksize,
                  verbose=verbose,
                  overwrite=overwrite,
                  nodata=nodata,
                  tags=tags,
                  **kwargs)

    def show(self, wavelengths=None, mask=False, flip=False, dpi=150, **kwargs):

        if (len(wavelengths) != 1) and (len(wavelengths) != 3):
            logger.exception('  Only 1-band or 3-band arrays can be plotted.')

        # plt.rcParams['figure.figsize'] = 3, 3
        plt.rcParams['axes.titlesize'] = 5
        plt.rcParams['axes.titlepad'] = 5
        # plt.rcParams['axes.grid'] = False
        # plt.rcParams['axes.spines.left'] = False
        # plt.rcParams['axes.spines.top'] = False
        # plt.rcParams['axes.spines.right'] = False
        # plt.rcParams['axes.spines.bottom'] = False
        # plt.rcParams['xtick.top'] = True
        # plt.rcParams['ytick.right'] = True
        # plt.rcParams['xtick.direction'] = 'in'
        # plt.rcParams['ytick.direction'] = 'in'
        # plt.rcParams['xtick.color'] = 'none'
        # plt.rcParams['ytick.color'] = 'none'
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.5

        fig = plt.figure()
        self.ax = fig.add_subplot(111)

        rgb = self._obj['bands'].sel(wavelength=wavelengths)

        if mask:

            if len(wavelengths) == 1:
                rgb = rgb.where((self._obj['mask'] < 3) & (rgb > 0))
            else:
                rgb = rgb.where((self._obj['mask'] < 3) & (rgb.max(axis=0) > 0))

        if len(wavelengths) == 3:

            rgb = rgb.transpose('y', 'x', 'wavelength')

            if flip:
                rgb = rgb[..., ::-1]

            rgb.plot.imshow(rgb='wavelength', ax=self.ax, **kwargs)

        else:
            rgb.plot.imshow(ax=self.ax, **kwargs)

        self._show()

    def _show(self):

        self.ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        self.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.tight_layout(pad=0.5)
        plt.show()

    def moving(self,
               variable='bands',
               band_coords='band',
               stat='mean',
               w=3,
               n_jobs=1):

        """
        Applies a moving window function to the DataArray

        Args:
            variable (Optional[str]): The variable to compute.
            band_coords (Optional[str]): The band coordinate name.
            stat (Optional[str]): The statistic to apply.
            w (Optional[int]): The moving window size.
            n_jobs (Optional[int]): The number of bands to process in parallel.

        Returns:
            DataArray
        """

        return moving(self._obj[variable].data,
                      stat,
                      w,
                      self._obj.coords[band_coords].values,
                      self._obj.y.values,
                      self._obj.x.values,
                      self._obj.attrs,
                      n_jobs)

    def norm_diff(self, b1, b2, variable='bands', mask=False):
        return norm_diff(self._obj[variable], b1, b2, mask=mask)

    def evi(self, variable='bands', mask=False):
        return evi(self._obj[variable], mask=mask)

    def evi2(self, variable='bands', mask=False):
        return evi2(self._obj[variable], mask=mask)

    def nbr(self, variable='bands', mask=False):
        return nbr(self._obj[variable], mask=mask)

    def ndvi(self, variable='bands', mask=False):
        return ndvi(self._obj[variable], mask=mask)

    def wi(self, variable='bands', mask=False):
        return wi(self._obj[variable], mask=mask)


@xr.register_dataarray_accessor('gw')
class GeoWombatAccessor(_UpdateConfig, DataProperties):

    """
    Xarray IO class
    """

    def __init__(self, xarray_obj):

        self._obj = xarray_obj
        self.config = config

        self._update_attrs()

    def to_raster(self,
                  filename,
                  n_jobs=1,
                  verbose=0,
                  overwrite=False,
                  driver='GTiff',
                  gdal_cache=512,
                  dtype=None,
                  row_chunks=None,
                  col_chunks=None,
                  pool_chunksize=10,
                  nodata=None,
                  tags=None,
                  **kwargs):

        """
        Writes an Xarray DataArray to a raster file

        Args:
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
        """

        if not hasattr(self._obj, 'crs'):
            raise AttributeError('The DataArray does not have a `crs` attribute.')

        if not hasattr(self._obj, 'transform'):
            raise AttributeError('The DataArray does not have a `transform` attribute.')

        kwargs = self._update_kwargs(**kwargs)

        to_raster(self._obj,
                  filename,
                  self._obj.crs,
                  self._obj.transform,
                  driver=driver,
                  n_jobs=n_jobs,
                  gdal_cache=gdal_cache,
                  dtype=dtype,
                  row_chunks=row_chunks,
                  col_chunks=col_chunks,
                  pool_chunksize=pool_chunksize,
                  verbose=verbose,
                  overwrite=overwrite,
                  nodata=nodata,
                  tags=tags,
                  **kwargs)

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
                *Only relevant if ``outname`` is not given.
            nodata (Optional[int or float]): The 'no data' value in the predictors.
            n_jobs (Optional[int]): The number of parallel jobs (chunks) for writing.
            backend (Optional[str]): The ``joblib`` backend scheduler.
            verbose (Optional[int]): The verbosity level.
            dtype (Optional[str]): The output data type passed to ``rasterio.write``.
            gdal_cache (Optional[int]): The GDAL cache (in MB) passed to ``rasterio.write``.
            kwargs (Optional[dict]): Additional keyword arguments passed to ``rasterio.write``.
                *The ``blockxsize`` and ``blockysize`` should be excluded because they are taken from ``chunksize``.

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
            >>> from cube import xarray_accessor
            >>> import xarray as xr
            >>>
            >>> def user_func(ds_):
            >>>     return ds_.max(axis=0)
            >>>
            >>> with xr.open_rasterio('image.tif', chunks=(1, 512, 512)) as ds:
            >>>     ds.io.apply('output.tif', user_func, n_jobs=8, overwrite=True, blockxsize=512, blockysize=512)
        """

        cluster = Cluster(n_workers=n_jobs,
                          threads_per_worker=1,
                          scheduler_port=0,
                          processes=False)

        cluster.start()

        with joblib.parallel_backend('dask', n_jobs=n_jobs):

            ds_sub = user_func(self._obj)
            ds_sub.attrs = self._obj.attrs
            ds_sub.io.to_raster(filename, n_jobs=n_jobs, **kwargs)

        cluster.stop()

    def subset(self,
               by='coords',
               left=None,
               top=None,
               right=None,
               bottom=None,
               rows=None,
               cols=None,
               center=False,
               mask_corners=False,
               chunksize=None):

        """
        Subsets a DataArray

        Args:
            by (str): TODO: give subsetting options
            left (Optional[float]): The left coordinate.
            top (Optional[float]): The top coordinate.
            right (Optional[float]): The right coordinate.
            bottom (Optional[float]): The bottom coordinate.
            rows (Optional[int]): The number of output rows.
            cols (Optional[int]): The number of output rows.
            center (Optional[bool]): Whether to center the subset on ``left`` and ``top``.
            mask_corners (Optional[bool]): Whether to mask corners (*requires ``pymorph``).
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
                      by='coords',
                      left=left,
                      top=top,
                      right=right,
                      bottom=bottom,
                      rows=rows,
                      cols=cols,
                      center=center,
                      mask_corners=mask_corners,
                      chunksize=chunksize)

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
                If not given, all bands are used. *Bands should be GDAL-indexed (i.e., the first band is 1, not 0).
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

    def moving(self, band_coords='band', stat='mean', w=3, n_jobs=1):

        """
        Applies a moving window function to the DataArray

        Args:
            band_coords (Optional[str]): The band coordinate name.
            stat (Optional[str]): The statistic to compute.
            w (Optional[int]): The moving window size (in pixels).
            n_jobs (Optional[int]): The number of bands to process in parallel.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as ds:
            >>>     ds = ds.gw.moving()
        """

        return moving(self._obj.data,
                      self._obj.coords[band_coords].values,
                      self._obj.y.values,
                      self._obj.x.values,
                      self._obj.attrs,
                      stat=stat,
                      w=w,
                      n_jobs=n_jobs)

    def norm_diff(self, b1, b2, mask=False):
        return norm_diff(self._obj, b1, b2, mask=mask)

    def evi(self, mask=False):
        return evi(self._obj, mask=mask)

    def evi2(self, mask=False):
        return evi2(self._obj, mask=mask)

    def nbr(self, mask=False):
        return nbr(self._obj, mask=mask)

    def ndvi(self, mask=False):
        return ndvi(self._obj, mask=mask)

    def wi(self, mask=False):
        return wi(self._obj, mask=mask)
