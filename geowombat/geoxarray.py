import os
import time
import ctypes
import multiprocessing as multi

from . import helpers
from .errors import logger

import numpy as np
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
import rasterio as rio
import joblib
from tqdm import tqdm

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


def _window_worker(w):
    """Helper to return window slice"""
    time.sleep(0.01)
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

    d_name = os.path.split(filename)[0]

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
                with multi.Pool(processes=n_jobs) as pool:

                    # Iterate over each window
                    for w, window_slice in tqdm(pool.imap_unordered(_window_worker,
                                                                    windows,
                                                                    chunksize=pool_chunksize),
                                                total=len(windows)):

                        # Prepend the band position index to the window slice
                        if len(data_shape) == 2:

                            window_slice_ = window_slice
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

                        # Iterate over each band
                        # for band_index in range(1, n_bands + 1):
                        #
                        #     # Prepend the band position index to the window slice
                        #     if len(data_shape) == 2:
                        #         window_slice_ = window_slice
                        #     else:
                        #         window_slice_ = tuple([band_index-1] + list(window_slice))
                        #
                        #     # Write the chunk to file
                        #     if isinstance(nodata, int) or isinstance(nodata, float):
                        #
                        #         dst.write(ds_data[window_slice_].squeeze().fillna(nodata).load().data,
                        #                   window=w,
                        #                   indexes=band_index)
                        #
                        #     else:
                        #
                        #         dst.write(ds_data[window_slice_].squeeze().load().data,
                        #                   window=w,
                        #                   indexes=band_index)

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

    def show_rgb(self, wavelengths, mask=False, flip=False, dpi=150, **kwargs):

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
            rgb = rgb.where((self._obj['mask'] < 3) & (rgb.max(axis=0) > 0))

        rgb = rgb.transpose('y', 'x', 'wavelength')

        if flip:
            rgb = rgb[..., ::-1]

        rgb.plot.imshow(rgb='wavelength', ax=self.ax, **kwargs)

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

    def subset_by_coords(self,
                         left,
                         top,
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
            left (float)
            top (float)
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
            cols = (right - left) / self._obj.res[0]

        if not isinstance(cols, int):
            raise AttributeError('The right coordinate or columns must be specified.')

        if isinstance(bottom, int) or isinstance(bottom, float):
            rows = (top - bottom) / self._obj.res[0]

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

            try:

                disk = pymorph.sedisk(r=int())

            except:
                logger.warning('  Cannot mask corners without Pymorph and a square subset.')

        transform = list(self._obj.transform)
        transform[2] = x_idx[0]
        transform[5] = y_idx[0]

        ds_sub.attrs['transform'] = tuple(transform)

        return ds_sub
