import os
import time
import ctypes
from collections import namedtuple
import multiprocessing as multi
import concurrent.futures

from . import helpers
from .errors import logger

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
import rasterio as rio
from rasterio.windows import Window
from rasterio import features
from affine import Affine
import joblib
from shapely import geometry
from tqdm import tqdm
import shapely

import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    import pymorph
except:
    pass

try:
    MKL_LIB = ctypes.CDLL('libmkl_rt.so')
except:
    MKL_LIB = None

shapely.speedups.enable()


def _window_worker(w):
    """Helper to return window slice"""
    # time.sleep(0.001)
    return w, (slice(w.row_off, w.row_off+w.height), slice(w.col_off, w.col_off+w.width))


def _xarray_writer(ds_data,
                   filename,
                   crs,
                   transform,
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
                   **kwargs):

    if MKL_LIB:
        __ = MKL_LIB.MKL_Set_Num_Threads(n_jobs)

    if overwrite:

        if os.path.isfile(filename):
            os.remove(filename)

    d_name = os.path.dirname(filename)

    if d_name:

        if not os.path.isdir(d_name):
            os.makedirs(d_name)

    data_shape = ds_data.shape

    if len(data_shape) == 2:

        n_bands = 1
        n_rows = data_shape[0]
        n_cols = data_shape[1]

        if not isinstance(row_chunks, int):
            row_chunks = ds_data.data.chunksize[0]

        if not isinstance(col_chunks, int):
            col_chunks = ds_data.data.chunksize[1]

    else:

        n_bands = data_shape[0]
        n_rows = data_shape[1]
        n_cols = data_shape[2]

        if not isinstance(row_chunks, int):
            row_chunks = ds_data.data.chunksize[1]

        if not isinstance(col_chunks, int):
            col_chunks = ds_data.data.chunksize[2]

    if isinstance(dtype, str):

        if ds_data.dtype != dtype:
            ds_data = ds_data.astype(dtype)

    else:
        dtype = ds_data.dtype

    # Setup the windows
    windows = helpers.setup_windows(n_rows, n_cols, row_chunks, col_chunks)
    # windows = helpers.setup_windows(n_rows, n_cols, row_chunks, col_chunks, return_as='dict')

    if n_bands > 1:
        indexes = list(range(1, n_bands + 1))

    outd = np.array([0], dtype='uint8')[None, None]

    if verbose > 0:
        print('Creating and writing to the file ...')

    # Rasterio environment context
    with rio.Env(GDAL_CACHEMAX=gdal_cache):

        # Open the output file for writing
        with rio.open(filename,
                      mode='w',
                      height=n_rows,
                      width=n_cols,
                      count=n_bands,
                      dtype=dtype,
                      nodata=nodata,
                      crs=crs,
                      transform=transform,
                      driver=driver,
                      sharing=False,
                      **kwargs) as dst:

            # def write_func(block, block_id=None):
            #
            #     # Current block upper left indices
            #     if len(block_id) == 2:
            #         i, j = block_id
            #     else:
            #         i, j = block_id[1:]
            #
            #     # Current block window
            #     w = windows['{:d}{:d}'.format(i, j)]
            #
            #     if n_bands == 1:
            #
            #         dst.write(np.squeeze(block),
            #                   window=w,
            #                   indexes=1)
            #
            #     else:
            #
            #         dst.write(block,
            #                   window=w,
            #                   indexes=indexes)
            #
            #     return outd
            #
            # ds_data.data.map_blocks(write_func,
            #                         dtype=ds_data.dtype,
            #                         chunks=(1, 1, 1)).compute(num_workers=n_jobs)

            if n_jobs == 1:

                if isinstance(nodata, int) or isinstance(nodata, float):
                    write_data = ds_data.squeeze().fillna(nodata).load().data
                else:
                    write_data = ds_data.squeeze().load().data

                if n_bands == 1:
                    dst.write(write_data, 1)
                else:
                    dst.write(write_data)

                if isinstance(tags, dict):

                    if tags:
                        dst.update_tags(**tags)

            else:

                # Multiprocessing pool context
                # This context is I/O bound, so use the default 'loky' scheduler
                with multi.Pool(processes=n_jobs) as pool:

                    # Iterate over each window
                    for w, window_slice in tqdm(pool.imap_unordered(_window_worker,
                                                                    windows,
                                                                    chunksize=pool_chunksize),
                                                total=len(windows)):

                # with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:

                    # for w, window_slice in tqdm(executor.map(_window_worker, windows), total=len(windows)):

                        # Prepend the band position index to the window slice
                        if n_bands == 1:

                            window_slice_ = tuple([slice(0, 1)] + list(window_slice))
                            indexes = 1

                        else:

                            window_slice_ = tuple([slice(0, n_bands)] + list(window_slice))
                            indexes = list(range(1, n_bands+1))

                        # Write the chunk to file
                        if isinstance(nodata, int) or isinstance(nodata, float):

                            dst.write(ds_data[window_slice_].squeeze().fillna(nodata).load().data,
                                      window=w,
                                      indexes=indexes)

                        else:

                            dst.write(ds_data[window_slice_].squeeze().load().data,
                                      window=w,
                                      indexes=indexes)

    if verbose > 0:
        print('Finished writing')


@xr.register_dataset_accessor('gw')
class GeoWombatAccessor(object):

    def __init__(self, xarray_obj):

        self._obj = xarray_obj
        self.ax = None

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

        _xarray_writer(self._obj[attribute],
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

    def evi2(self, mask=False):

        result = (2.5 * ((self._obj['bands'].sel(wavelength='nir') - self._obj['bands'].sel(wavelength='red')) /
                         (self._obj['bands'].sel(wavelength='nir') + 1.0 + (2.4 * (self._obj['bands'].sel(wavelength='red')))))).fillna(0)

        if mask:
            result = result.where(self._obj['mask'] < 3)

        result = da.where(result < 0, 0, result)
        result = da.where(result > 1, 1, result)

        return xr.DataArray(result,
                            dims=['y', 'x'],
                            coords={'y': self._obj.y, 'x': self._obj.x})

    def nbr(self, mask=False):

        result = ((self._obj['bands'].sel(wavelength='nir') - self._obj['bands'].sel(wavelength='swir2')) /
                  (self._obj['bands'].sel(wavelength='nir') + self._obj['bands'].sel(wavelength='swir2'))).fillna(0)

        if mask:
            result = result.where(self._obj['mask'] < 3)

        result = da.where(result < -1, 0, result)
        result = da.where(result > 1, 1, result)

        return xr.DataArray(result,
                            dims=['y', 'x'],
                            coords={'y': self._obj.y, 'x': self._obj.x})

    def ndvi(self, mask=False):

        result = ((self._obj['bands'].sel(wavelength='nir') - self._obj['bands'].sel(wavelength='red')) /
                  (self._obj['bands'].sel(wavelength='nir') + self._obj['bands'].sel(wavelength='red'))).fillna(0)

        if mask:
            result = result.where(self._obj['mask'] < 3)

        result = da.where(result < -1, 0, result)
        result = da.where(result > 1, 1, result)

        return xr.DataArray(result,
                            dims=['y', 'x'],
                            coords={'y': self._obj.y, 'x': self._obj.x})

    def wi(self, mask=False):

        result = da.where((self._obj['bands'].sel(wavelength='swir1') + self._obj['bands'].sel(wavelength='red')) > 0.5, 0,
                          1.0 - ((self._obj['bands'].sel(wavelength='swir1') + self._obj['bands'].sel(wavelength='red')) / 0.5))

        if mask:
            result = result.where(self._obj['mask'] < 3)

        result = da.where(result < 0, 0, result)
        result = da.where(result > 1, 1, result)

        return xr.DataArray(result,
                            dims=['y', 'x'],
                            coords={'y': self._obj.y, 'x': self._obj.x})

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
class GeoWombatAccessor(object):

    """
    Xarray IO class
    """

    def __init__(self, xarray_obj):

        self._obj = xarray_obj

        if len(self._obj.shape) == 2:
            self.row_chunks, self.col_chunks = self._obj.data.chunksize
        else:
            self.band_chunks, self.row_chunks, self.col_chunks = self._obj.data.chunksize

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

        _xarray_writer(self._obj,
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
                io_chunks=(512, 512),
                x_chunks=(5000, 1),
                overwrite=False,
                return_as='array',
                n_jobs=1,
                verbose=0,
                nodata=0,
                dtype='uint8',
                gdal_cache=512,
                **kwargs):

        """
        Predicts an image using a pre-fit model

        Args:
            clf (object): A fitted classifier `geowombat.model.Model` instance with a `predict` method.
            outname (Optional[str]): An outname file name for the predictions.
            io_chunks (Optional[tuple]): The chunk size for I/O.
            x_chunks (Optional[tuple]): The chunk size for the X predictors.
            overwrite (Optional[bool]): Whether to overwrite an existing file.
            return_as (Optional[str]): Whether to return the predictions as a `DataArray` or `Dataset`.
                *Only relevant if `outname` is not given.
            nodata (Optional[int or float]): The 'no data' value in the predictors.
            n_jobs (Optional[int]): The number of parallel jobs (chunks) for writing.
            verbose (Optional[int]): The verbosity level.
            dtype (Optional[str]): The output data type passed to `Rasterio`.
            gdal_cache (Optional[int]): The GDAL cache (in MB) passed to `Rasterio`.
            kwargs (Optional[dict]): Keyword arguments pass to `Rasterio`.
                *The `blockxsize` and `blockysize` should be excluded.

        Returns:
            Predictions (Dask array) if `outname` is None, otherwise writes to `outname`.
        """

        if verbose > 0:
            logger.info('  Predicting and saving to {} ...'.format(outname))

        with joblib.parallel_backend('dask'):

            n_dims, n_rows, n_cols = self._obj.shape

            # Reshape the data for fitting and
            #   return a Dask array
            X = self._obj.stack(z=('y', 'x')).transpose().chunk(x_chunks).fillna(nodata).data

            # Apply the predictions
            predictions = clf.predict(X).reshape(n_rows, n_cols).rechunk(io_chunks)

            if return_as == 'dataset':

                # Store the predictions as an `Xarray` `Dataset`
                predictions = xr.Dataset({'pred': (['y', 'x'], predictions)},
                                         coords={'y': ('y', self._obj.y),
                                                 'x': ('x', self._obj.x)},
                                         attrs=self._obj.attrs)

            else:

                # Store the predictions as an `Xarray` `DataArray`
                predictions = xr.DataArray(data=predictions,
                                           dims=('y', 'x'),
                                           coords={'y': ('y', self._obj.y),
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

            else:
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

        cluster = LocalCluster(n_workers=n_jobs,
                               threads_per_worker=1,
                               scheduler_port=0,
                               processes=False)

        client = Client(cluster)

        with joblib.parallel_backend('dask'):

            ds_sub = user_func(self._obj)
            ds_sub.attrs = self._obj.attrs
            ds_sub.io.to_raster(filename, n_jobs=1, **kwargs)

        client.close()
        cluster.close()

        client = None
        cluster = None

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
            chunksize_ = (self.band_chunks, self.row_chunks, self.col_chunks)

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
    def bounds(self):

        """
        Returns the `DataArray` bounds
        """

        Bounds = namedtuple('Bounds', 'left right top bottom')

        bounds = Bounds(left=self._obj.x.min().values,
                        right=self._obj.x.max().values,
                        top=self._obj.y.max().values,
                        bottom=self._obj.y.min().values)

        return bounds

    def poly_to_points(self, df, frac=1.0):

        """
        Converts polygons to points

        Args:
            df (GeoDataFrame)
            frac (Optional[float]): A fractional subset of points to extract in each feature.

        Returns:
            (GeoDataFrame)
        """

        array_bounds = self._obj.gw.bounds

        point_df = None

        dataframes = list()

        # TODO: parallel over features
        for i in range(0, df.shape[0]):

            # Get the current feature's geometry
            geom = df.iloc[i].geometry

            # Project to the DataArray's CRS
            # dfs = gpd.GeoDataFrame([0], geometry=[geom], crs=df.crs)
            # dfs = dfs.to_crs(self._obj.crs)
            # geom = dfs.iloc[0].geometry

            # Get the feature's bounding extent
            minx, miny, maxx, maxy = geom.bounds
            out_shape = (int((maxy - miny) / self._obj.res[0]), int((maxx - minx) / self._obj.res[0]))
            transform = Affine(self._obj.res[0], 0.0, minx, 0.0, -self._obj.res[0], maxy)

            # "Rasterize" the geometry into a NumPy array
            feature_array = features.rasterize([geom],
                                               out_shape=out_shape,
                                               fill=0,
                                               out=None,
                                               transform=transform,
                                               all_touched=False,
                                               default_value=1,
                                               dtype='int32')

            # Get the indices of the feature's envelope
            valid_samples = np.where(feature_array == 1)

            # Convert the indices to map indices
            y_samples = valid_samples[0] + int(round(abs(array_bounds.top - maxy)) / self._obj.res[0])
            x_samples = valid_samples[1] + int(round(abs(minx - array_bounds.left)) / self._obj.res[0])

            # Convert the indices to map coordinates
            y_coords = array_bounds.top - y_samples * self._obj.res[0]
            x_coords = array_bounds.left + x_samples * self._obj.res[0]

            if frac < 1:

                rand_idx = np.random.choice(np.arange(0, y_coords.shape[0]),
                                            size=int(y_coords.shape[0]*frac),
                                            replace=False)

                y_coords = y_coords[rand_idx]
                x_coords = x_coords[rand_idx]

            n_samples = y_coords.shape[0]

            # Combine the coordinates into `Shapely` point geometry
            if not isinstance(point_df, gpd.GeoDataFrame):

                point_df = gpd.GeoDataFrame(data=np.c_[np.zeros(n_samples, dtype='int64') + i,
                                                       np.arange(0, n_samples)],
                                           geometry=gpd.points_from_xy(x_coords, y_coords),
                                           crs=self._obj.crs,
                                           columns=['poly', 'point'])

                last_point = point_df.point.max() + 1

            else:

                point_df = gpd.GeoDataFrame(data=np.c_[np.zeros(n_samples, dtype='int64') + i,
                                                       np.arange(last_point, last_point + n_samples)],
                                            geometry=gpd.points_from_xy(x_coords, y_coords),
                                            crs=self._obj.crs,
                                            columns=['poly', 'point'])

                last_point = last_point + point_df.point.max() + 1

            dataframes.append(point_df)

        return pd.concat(dataframes, axis=0)

    def extract(self, aoi, bands=None, band_names=None, frac=1.0, **kwargs):

        """
        Extracts data within an area or points of interest. Projections do not
        need to match, as they are handled 'on-the-fly'.

        Args:
            aoi (str or GeoDataFrame): A file or GeoDataFrame to extract data frame.
            bands (Optional[int or 1d array-like]): A band or list of bands to extract.
                If not given, all bands are used. *Bands should be GDAL-indexed (i.e., the first band is 1, not 0).
            band_names (Optional[list]): A list of band names. Length should be the same as `bands`.
            frac (Optional[float]): A fractional subset of points to extract in each polygon feature.
            kwargs (Optional[dict]): Keyword arguments passed to `Dask` compute.

        Returns:
            Extracted data for every data point within or intersecting the geometry (GeoDataFrame)
        """

        if isinstance(aoi, str):

            if not os.path.isfile(aoi):
                logger.exception('  The AOI file does not exist.')

            df = gpd.read_file(aoi)

        elif isinstance(aoi, gpd.GeoDataFrame):
            df = aoi
        else:
            logger.exception('  The AOI must be a vector file or a GeoDataFrame.')

        shape_len = len(self._obj.shape)

        bands_type = 'array'

        if isinstance(bands, list):
            bands_idx = np.array(bands, dtype='int64') - 1
        elif isinstance(bands, np.ndarray):
            bands_idx = bands - 1
        elif isinstance(bands, int):

            bands_idx = slice(bands, bands+1)
            bands_type = 'slice'

        else:

            bands_type = 'slice'

            if shape_len > 2:
                bands_idx = slice(0, self._obj.shape[0])

        # Re-project the data to match the image CRS
        df = df.to_crs(self._obj.crs)

        # Subset the DataArray
        # minx, miny, maxx, maxy = df.total_bounds
        #
        # obj_subset = self._obj.gw.subset(left=float(minx)-self._obj.res[0],
        #                                  top=float(maxy)+self._obj.res[0],
        #                                  right=float(maxx)+self._obj.res[0],
        #                                  bottom=float(miny)-self._obj.res[0])

        # Convert polygons to points
        if type(df.iloc[0].geometry) == geometry.Polygon:
            df = self.poly_to_points(df, frac=frac)

        x, y = df.geometry.x.values, df.geometry.y.values

        left = self._obj.transform[2]
        top = self._obj.transform[5]

        x = np.int64(np.round(np.abs(x - left) / self._obj.res[0]))
        y = np.int64(np.round(np.abs(top - y) / self._obj.res[0]))

        if shape_len == 2:
            res = self._obj.data.vindex[y, x].compute(**kwargs)
        else:
            res = self._obj.data.vindex[bands_idx, y, x].compute(**kwargs)

        if shape_len == 2:

            if band_names:
                df[band_names[0]] = res.flatten()
            else:
                df['bd1'] = res.flatten()

        else:

            if bands_type in ['array', 'slice']:

                if bands_type == 'array':
                    enum = bands_idx.tolist()
                else:
                    enum = list(range(bands_idx.start, bands_idx.stop))

                for i, band in enumerate(enum):

                    if band_names:
                        df[band_names[i]] = res[:, i]
                    else:
                        df['bd{:d}'.format(i+1)] = res[:, i]

            else:

                for band in range(1, self._obj.shape[0]+1):

                    if band_names:
                        df[band_names[band-1]] = res[:, band-1]
                    else:
                        df['bd{:d}'.format(band)] = res[:, band-1]

        return df
