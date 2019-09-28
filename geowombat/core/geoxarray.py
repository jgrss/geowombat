import os
from collections import namedtuple

from ..errors import logger
from ..moving import moving_window
from ..util import Cluster, DataArrayProperties, DatasetProperties

from .io import xarray_to_raster
from .util import Chunks, Converters, DataArrayBandMath, DatasetBandMath

import numpy as np
import geopandas as gpd
import xarray as xr
import dask.array as da
from dask_ml.wrappers import ParallelPostFit
from rasterio.crs import CRS
from affine import Affine
import joblib
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    import pymorph
except:
    pass


@xr.register_dataset_accessor('gw')
class GeoWombatAccessor(Chunks, Converters, DatasetBandMath, DatasetProperties):

    def __init__(self, xarray_obj):

        self._obj = xarray_obj
        self.sensor = None
        self.ax = None

    def set_sensor(self, sensor):
        self.sensor = sensor

    def to_raster(self,
                  filename,
                  attribute='bands',
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
            attribute (Optional[str]): The attribute to write.
            n_jobs (Optional[str]): The number of parallel chunks to write.
            verbose (Optional[int]): The verbosity level.
            overwrite (Optional[bool]): Whether to overwrite an existing file.
            driver (Optional[str]): The raster driver.
            gdal_cache (Optional[int]): The GDAL cache size (in MB).
            dtype (Optional[int]): The output data type.
            row_chunks (Optional[int]): The processing row chunk size.
            col_chunks (Optional[int]): The processing column chunk size.
            pool_chunksize (Optional[int]): The `multiprocessing.Pool` chunk size.
            nodata (Optional[int]): A 'no data' value.
            tags (Optional[dict]): Image tags to write to file.
            kwargs (Optional[dict]):

                nodata (float or int) (should come from the Dataset if not specified)
                tiled (bool)
                compress (str)

        TODO: pass attributes to GeoTiff metadata

        Returns:
            None
        """

        if not hasattr(self._obj, 'crs'):
            raise AttributeError('The Dataset does not have a `crs` attribute.')

        if not hasattr(self._obj, 'transform'):
            raise AttributeError('The Dataset does not have a `transform` attribute.')

        xarray_to_raster(self._obj[attribute],
                         filename,
                         self._obj.crs,
                         self._obj.transform,
                         driver,
                         n_jobs,
                         gdal_cache,
                         dtype,
                         row_chunks,
                         col_chunks,
                         pool_chunksize,
                         verbose,
                         overwrite,
                         nodata,
                         tags,
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


@xr.register_dataarray_accessor('gw')
class GeoWombatAccessor(Chunks, Converters, DataArrayBandMath, DataArrayProperties):

    """
    Xarray IO class
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

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
            gdal_cache (Optional[int]): The GDAL cache size (in MB).
            dtype (Optional[int]): The output data type.
            row_chunks (Optional[int]): The processing row chunk size.
            col_chunks (Optional[int]): The processing column chunk size.
            pool_chunksize (Optional[int]): The `multiprocessing.Pool` chunk size.
            nodata (Optional[int]): A 'no data' value.
            tags (Optional[dict]): Image tags to write to file.
            kwargs (Optional[dict]):

                nodata (float or int) (should come from the Dataset if not specified)
                tiled (bool)
                blockxsize (int)
                blockysize (int)
                compress (str)

        TODO: pass attributes to GeoTiff metadata

        Returns:
            None
        """

        if not hasattr(self._obj, 'crs'):
            raise AttributeError('The DataArray does not have a `crs` attribute.')

        if not hasattr(self._obj, 'transform'):
            raise AttributeError('The DataArray does not have a `transform` attribute.')

        xarray_to_raster(self._obj,
                         filename,
                         self._obj.crs,
                         self._obj.transform,
                         driver,
                         n_jobs,
                         gdal_cache,
                         dtype,
                         row_chunks,
                         col_chunks,
                         pool_chunksize,
                         verbose,
                         overwrite,
                         nodata,
                         tags,
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
            clf (object): A fitted classifier `geowombat.model.Model` instance with a `predict` method.
            outname (Optional[str]): An outname file name for the predictions.
            chunksize (Optional[str or tuple]): The chunk size for I/O. Default is 'same', or use the input chunk size.
            x_chunks (Optional[tuple]): The chunk size for the X predictors.
            overwrite (Optional[bool]): Whether to overwrite an existing file.
            return_as (Optional[str]): Whether to return the predictions as a `DataArray` or `Dataset`.
                *Only relevant if `outname` is not given.
            nodata (Optional[int or float]): The 'no data' value in the predictors.
            n_jobs (Optional[int]): The number of parallel jobs (chunks) for writing.
            backend (Optional[str]): The `joblib` backend scheduler.
            verbose (Optional[int]): The verbosity level.
            dtype (Optional[str]): The output data type passed to `Rasterio`.
            gdal_cache (Optional[int]): The GDAL cache (in MB) passed to `Rasterio`.
            kwargs (Optional[dict]): Keyword arguments pass to `Rasterio`.
                *The `blockxsize` and `blockysize` should be excluded.

        Returns:
            Predictions (Dask array) if `outname` is None, otherwise writes to `outname`.
        """

        if not isinstance(clf, ParallelPostFit):
            clf = ParallelPostFit(estimator=clf)

        if verbose > 0:
            logger.info('  Predicting and saving to {} ...'.format(outname))

        if isinstance(chunksize, str) and chunksize == 'same':
            chunksize = self.check_chunksize(self._obj.data.chunksize, output='3d')
        else:

            if not isinstance(chunksize, tuple):
                logger.warning('  The chunksize parameter should be a tuple.')

            # TODO: make compatible with multi-layer predictions (e.g., probabilities)
            if len(chunksize) != 2:
                logger.warning('  The chunksize should be two-dimensional.')

        if backend == 'dask':

            cluster = Cluster(n_workers=1,
                              threads_per_worker=n_jobs,
                              scheduler_port=0,
                              processes=False)

            cluster.start()

        with joblib.parallel_backend(backend, n_jobs=n_jobs):

            n_dims, n_rows, n_cols = self._obj.shape

            # Reshape the data for fitting and
            #   return a Dask array
            if isinstance(nodata, int) or isinstance(nodata, float):
                X = self._obj.stack(z=('y', 'x')).transpose().chunk(x_chunks).fillna(nodata).data
            else:
                X = self._obj.stack(z=('y', 'x')).transpose().chunk(x_chunks).data

            # Apply the predictions
            predictions = clf.predict(X).reshape(1, n_rows, n_cols).rechunk(chunksize).astype(dtype)

            if return_as == 'dataset':

                # Store the predictions as an `Xarray` `Dataset`
                predictions = xr.Dataset({'pred': (['band', 'y', 'x'], predictions)},
                                         coords={'band': [1],
                                                 'y': ('y', self._obj.y),
                                                 'x': ('x', self._obj.x)},
                                         attrs=self._obj.attrs)

            else:

                # Store the predictions as an `Xarray` `DataArray`
                predictions = xr.DataArray(data=predictions,
                                           dims=('band', 'y', 'x'),
                                           coords={'band': [1],
                                                   'y': ('y', self._obj.y),
                                                   'x': ('x', self._obj.x)},
                                           attrs=self._obj.attrs)

            if isinstance(outname, str):

                predictions.gw.to_raster(outname,
                                         attribute='pred',
                                         n_jobs=n_jobs,
                                         dtype=dtype,
                                         gdal_cache=gdal_cache,
                                         overwrite=overwrite,
                                         blockxsize=io_chunks[0],
                                         blockysize=io_chunks[1],
                                         **kwargs)

        if backend == 'dask':
            cluster.stop()

        return predictions

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
        Subsets the DataArray by coordinates

        Args:
            by (str)
            left (Optional[float])
            top (Optional[float])
            right (Optional[float])
            bottom (Optional[float])
            rows (Optional[int])
            cols (Optional[int])
            center (Optional[bool])
            mask_corners (Optional[bool])
            chunksize (Optional[tuple])

        Example:
            >>> from cube import xarray_accessor
            >>> import xarray as xr
            >>>
            >>> with xr.open_rasterio('image.tif', chunks=(1, 512, 512)) as ds:
            >>>     ds_sub = ds.subset.by_coords(-263529.884, 953985.314, rows=2048, cols=2048)
        """

        if isinstance(right, int) or isinstance(right, float):
            cols = int((right - left) / self._obj.res[0])

        if not isinstance(cols, int):
            raise AttributeError('The right coordinate or columns must be specified.')

        if isinstance(bottom, int) or isinstance(bottom, float):
            rows = int((top - bottom) / self._obj.res[0])

        if not isinstance(rows, int):
            raise AttributeError('The bottom coordinate or rows must be specified.')

        x_idx = np.linspace(left, left + (cols * self._obj.res[0]), cols)
        y_idx = np.linspace(top, top - (rows * self._obj.res[0]), rows)

        if center:

            y_idx += ((rows / 2.0) * self._obj.res[0])
            x_idx -= ((cols / 2.0) * self._obj.res[0])

        if chunksize:
            chunksize_ = chunksize
        else:
            chunksize_ = (self.gw.band_chunks, self.gw.row_chunks, self.gw.col_chunks)

        ds_sub = self._obj.sel(y=y_idx,
                               x=x_idx,
                               method='nearest').chunk(chunksize_)

        if mask_corners:

            if len(chunksize_) == 2:
                chunksize_pym = chunksize_
            else:
                chunksize_pym = chunksize_[1:]

            try:

                disk = da.from_array(pymorph.sedisk(r=int(rows/2.0))[:rows, :cols], chunks=chunksize_pym).astype('uint8')
                ds_sub = ds_sub.where(disk == 1)

            except:
                logger.warning('  Cannot mask corners without Pymorph and a square subset.')

        transform = list(self._obj.transform)
        transform[2] = x_idx[0]
        transform[5] = y_idx[0]

        ds_sub.attrs['transform'] = tuple(transform)

        return ds_sub

    @property
    def meta(self):

        """
        Returns the `DataArray` bounds
        """

        Profile = namedtuple('Profile', 'left right top bottom bounds affine geometry')

        left = self._obj.x.min().values
        right = self._obj.x.max().values
        top = self._obj.y.max().values
        bottom = self._obj.y.min().values

        geometry = Polygon([(left, bottom),
                            (left, top),
                            (right, top),
                            (right, bottom),
                            (left, bottom)])

        bounds = (left, bottom, right, top)

        return Profile(left=left,
                       right=right,
                       top=top,
                       bottom=bottom,
                       bounds=bounds,
                       affine=Affine(*self._obj.transform),
                       geometry=geometry)

    def extract(self,
                aoi,
                bands=None,
                time_names=None,
                band_names=None,
                frac=1.0,
                all_touched=False,
                mask=None,
                **kwargs):

        """
        Extracts data within an area or points of interest. Projections do not
        need to match, as they are handled 'on-the-fly'.

        Args:
            aoi (str or GeoDataFrame): A file or GeoDataFrame to extract data frame.
            bands (Optional[int or 1d array-like]): A band or list of bands to extract.
                If not given, all bands are used. *Bands should be GDAL-indexed (i.e., the first band is 1, not 0).
            band_names (Optional[list]): A list of band names. Length should be the same as `bands`.
            time_names (Optional[list]): A list of time names.
            frac (Optional[float]): A fractional subset of points to extract in each polygon feature.
            all_touched (Optional[bool]): The `all_touched` argument is passed to `rasterio.features.rasterize`.
            mask (Optional[Shapely Polygon]): A `shapely.geometry.Polygon` mask to subset to.
            kwargs (Optional[dict]): Keyword arguments passed to `Dask` compute.

        Returns:
            Extracted data for every data point within or intersecting the geometry (GeoDataFrame)
        """

        if isinstance(aoi, gpd.GeoDataFrame):
            df = aoi
        else:

            if isinstance(aoi, str):

                if not os.path.isfile(aoi):
                    logger.exception('  The AOI file does not exist.')

                df = gpd.read_file(aoi)

            else:
                logger.exception('  The AOI must be a vector file or a GeoDataFrame.')

        shape_len = len(self._obj.shape)

        if isinstance(bands, list):
            bands_idx = (np.array(bands, dtype='int64') - 1).tolist()
        elif isinstance(bands, np.ndarray):
            bands_idx = (bands - 1).tolist()
        elif isinstance(bands, int):
            bands_idx = [bands]
        else:

            if shape_len > 2:
                bands_idx = slice(0, None)

        if self._obj.crs != CRS.from_dict(df.crs).to_proj4():

            # Re-project the data to match the image CRS
            df = df.to_crs(self._obj.crs)

        # Ensure all geometry is valid
        df = df[df['geometry'].apply(lambda x_: x_ is not None)]

        # Remove data outside of the image bounds
        df = gpd.overlay(df,
                         gpd.GeoDataFrame(data=[0],
                                          geometry=[self._obj.gw.meta.geometry],
                                          crs=df.crs),
                         how='intersection')

        if isinstance(mask, Polygon):

            df = df[df.within(mask)]

            if df.empty:
                logger.exception('  No geometry intersects the user-provided mask.')

        # Subset the DataArray
        # minx, miny, maxx, maxy = df.total_bounds
        #
        # obj_subset = self._obj.gw.subset(left=float(minx)-self._obj.res[0],
        #                                  top=float(maxy)+self._obj.res[0],
        #                                  right=float(maxx)+self._obj.res[0],
        #                                  bottom=float(miny)-self._obj.res[0])

        # Convert polygons to points
        if type(df.iloc[0].geometry) == Polygon:
            df = self.polygons_to_points(self._obj, df, frac=frac, all_touched=all_touched)

        x, y = df.geometry.x.values, df.geometry.y.values

        left = self._obj.transform[2]
        top = self._obj.transform[5]

        x = np.int64(np.round(np.abs(x - left) / self._obj.res[0]))
        y = np.int64(np.round(np.abs(top - y) / self._obj.res[0]))

        if shape_len == 2:
            vidx = (y, x)
        else:

            vidx = (bands_idx, y.tolist(), x.tolist())

            for b in range(0, shape_len-3):
                vidx = (slice(0, None),) + vidx

        res = self._obj.data.vindex[vidx].compute(**kwargs)

        if len(res.shape) == 1:

            if band_names:
                df[band_names[0]] = res.flatten()
            else:
                df['bd1'] = res.flatten()

        else:

            if isinstance(bands_idx, list):
                enum = bands_idx.tolist()
            elif isinstance(bands_idx, slice):

                if bands_idx.start and bands_idx.stop:
                    enum = list(range(bands_idx.start, bands_idx.stop))
                else:
                    enum = list(range(0, self._obj.shape[-3]))

            else:
                enum = list(range(0, self._obj.shape[-3]))

            if len(res.shape) > 2:

                for t in range(0, self._obj.shape[0]):

                    if time_names:
                        time_name = time_names[t]
                    else:
                        time_name = t + 1

                    for i, band in enumerate(enum):

                        if band_names:
                            band_name = band_names[i]
                        else:
                            band_name = i + 1

                        if band_names:
                            df['{}_{}'.format(time_name, band_name)] = res[:, t, i].flatten()
                        else:
                            df['t{:d}_bd{:d}'.format(time_name, band_name)] = res[:, t, i].flatten()

            else:

                for i, band in enumerate(enum):

                    if band_names:
                        df[band_names[i]] = res[:, i]
                    else:
                        df['bd{:d}'.format(i+1)] = res[:, i]

        return df

    def moving(self, stat='mean', w=3):

        """
        Applies a moving window function to the DataArray

        Args:
            stat (Optional[str]): The statistic to apply.
            w (Optional[int]): The moving window size.

        Returns:
            DataArray
        """

        def move_func(data):

            if max(data.shape) < 2:
                return data
            else:
                return moving_window(data, stat=stat, w=w)

        results = self._obj.data.squeeze().astype('float64').map_overlap(move_func,
                                                                         depth=int(w / 2.0),
                                                                         trim=True,
                                                                         boundary='reflect',
                                                                         dtype='float64').reshape(self._obj.shape)

        return xr.DataArray(data=results,
                            dims=('band', 'y', 'x'),
                            coords={'band': self._obj.coords['band'],
                                    'y': ('y', self._obj.y),
                                    'x': ('x', self._obj.x)},
                            attrs=self._obj.attrs)
