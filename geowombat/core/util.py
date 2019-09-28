import os
from collections import namedtuple
import multiprocessing as multi

from ..errors import logger
from .conversion import dask_to_datarray
from ..moving import moving_window

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask.array as da
from rasterio import features
import shapely
from affine import Affine
from tqdm import tqdm


shapely.speedups.enable()


def get_geometry_info(geometry, res):

    """
    Gets information from a Shapely geometry object

    Args:
        geometry (object): A `shapely.geometry` object.
        res (float): The cell resolution for the affine transform.

    Returns:
        Geometry information (namedtuple)
    """

    GeomInfo = namedtuple('GeomInfo', 'left bottom right top shape transform')

    minx, miny, maxx, maxy = geometry.bounds
    out_shape = (int((maxy - miny) / res), int((maxx - minx) / res))

    return GeomInfo(left=minx,
                    bottom=miny,
                    right=maxx,
                    top=maxy,
                    shape=out_shape,
                    transform=Affine(res, 0.0, minx, 0.0, -res, maxy))


def get_file_extension(filename):

    """
    Gets file and directory name information

    Args:
        filename (str): The file name.

    Returns:
        Name information (namedtuple)
    """

    FileNames = namedtuple('FileNames', 'd_name f_name f_base f_ext')

    d_name, f_name = os.path.splitext(filename)
    f_base, f_ext = os.path.split(f_name)

    return FileNames(d_name=d_name, f_name=f_name, f_base=f_base, f_ext=f_ext)


def n_rows_cols(pixel_index, block_size, rows_cols):

    """
    Adjusts block size for the end of image rows and columns.

    Args:
        pixel_index (int): The current pixel row or column index.
        block_size (int): The image block size.
        rows_cols (int): The total number of rows or columns in the image.

    Returns:
        Adjusted block size as int.
    """

    return block_size if (pixel_index + block_size) < rows_cols else rows_cols - pixel_index


class Chunks(object):

    @staticmethod
    def get_chunk_dim(chunksize):
        return '{:d}d'.format(len(chunksize))

    @staticmethod
    def check_chunktype(chunksize, output='3d'):

        chunk_len = len(chunksize)
        output_len = int(output[0])

        if not isinstance(chunksize, tuple):
            if not isinstance(chunksize, dict):
                logger.warning('  The chunksize parameter should be a tuple or a dictionary.')

        # TODO: make compatible with multi-layer predictions (e.g., probabilities)
        if chunk_len != output_len:
            logger.warning('  The chunksize should be two-dimensional.')

    @staticmethod
    def check_chunksize(chunksize, output='3d'):

        chunk_len = len(chunksize)
        output_len = int(output[0])

        if chunk_len != output_len:

            if (chunk_len == 2) and (output_len == 3):
                return (1,) + chunksize
            elif (chunk_len == 3) and (output_len == 2):
                return chunksize[1:]

        return chunksize


class MapProcesses(object):

    @staticmethod
    def map_moving(data, stat, w, b, y, x, attrs, n_jobs):

        """
        Applies a moving window function over dask array blocks

        Args:
            data (dask.array)
            stat (str)
            w (int)
            b (int or str or list)
            y (1d array-like)
            x (1d array-like)
            attrs (dict)
            n_jobs (int)

        Returns:
            (DataArray)
        """

        if n_jobs <= 0:

            logger.warning('  The number of parallel jobs should be a positive integer, so setting n_jobs=1.')
            n_jobs = 1

        hw = int(w / 2.0)

        def move_func(block_data):

            if max(block_data.shape) <= hw:
                return data
            else:
                return moving_window(block_data, stat=stat, w=w, n_jobs=n_jobs)

        if len(data.shape) == 2:
            out_shape = (1,) + data.shape
        else:
            out_shape = data.shape

        result = data.reshape(out_shape).astype('float64').map_overlap(move_func,
                                                                       depth=hw,
                                                                       trim=True,
                                                                       boundary='reflect',
                                                                       dtype='float64').reshape(out_shape)

        if isinstance(b, np.ndarray):
            if isinstance(b.tolist(), str):
                b = [b.tolist()]

        if not isinstance(b, list):
            if not isinstance(b, np.ndarray):
                b = [b]

        return xr.DataArray(data=result,
                            dims=('band', 'y', 'x'),
                            coords={'band': b,
                                    'y': y,
                                    'x': x},
                            attrs=attrs)


def rasterize_geometry(i, geom, crs, res, all_touched, meta, frac):

    # Get the feature's bounding extent
    geom_info = get_geometry_info(geom, res)

    if min(geom_info.shape) == 0:
        return gpd.GeoDataFrame([])

    # "Rasterize" the geometry into a NumPy array
    feature_array = features.rasterize([geom],
                                       out_shape=geom_info.shape,
                                       fill=0,
                                       out=None,
                                       transform=geom_info.transform,
                                       all_touched=all_touched,
                                       default_value=1,
                                       dtype='int32')

    # Get the indices of the feature's envelope
    valid_samples = np.where(feature_array == 1)

    # Convert the indices to map indices
    y_samples = valid_samples[0] + int(round(abs(meta.top - geom_info.top)) / res)
    x_samples = valid_samples[1] + int(round(abs(geom_info.left - meta.left)) / res)

    # Convert the map indices to map coordinates
    x_coords, y_coords = meta.affine * (x_samples, y_samples)

    # y_coords = meta.top - y_samples * data.res[0]
    # x_coords = meta.left + x_samples * data.res[0]

    if frac < 1:

        rand_idx = np.random.choice(np.arange(0, y_coords.shape[0]),
                                    size=int(y_coords.shape[0] * frac),
                                    replace=False)

        y_coords = y_coords[rand_idx]
        x_coords = x_coords[rand_idx]

    n_samples = y_coords.shape[0]

    # Combine the coordinates into `Shapely` point geometry
    return gpd.GeoDataFrame(data=np.c_[np.zeros(n_samples, dtype='int64') + i,
                                       np.arange(0, n_samples)],
                            geometry=gpd.points_from_xy(x_coords, y_coords),
                            crs=crs,
                            columns=['poly', 'point'])


def _iter_func(a):
    return a


class Converters(object):

    @staticmethod
    def polygons_to_points(data, df, frac=1.0, all_touched=False, n_jobs=1):

        """
        Converts polygons to points

        Args:
            data (DataArray or Dataset)
            df (GeoDataFrame): The `GeoDataFrame` with geometry to rasterize.
            frac (Optional[float]): A fractional subset of points to extract in each feature.
            all_touched (Optional[bool]): The `all_touched` argument is passed to `rasterio.features.rasterize`.
            n_jobs (Optional[int]): The number of features to rasterize in parallel.

        Returns:
            (GeoDataFrame)
        """

        meta = data.gw.meta

        dataframes = list()

        with multi.Pool(processes=n_jobs) as pool:

            for i in tqdm(pool.imap(_iter_func, range(0, df.shape[0])), total=df.shape[0]):

                # Get the current feature's geometry
                geom = df.iloc[i].geometry

                point_df = rasterize_geometry(i, geom, data.crs, data.res[0], all_touched, meta, frac)

                if not point_df.empty:
                    dataframes.append(point_df)

        dataframes = pd.concat(dataframes, axis=0)

        # Make the points unique
        dataframes.loc[:, 'point'] = np.arange(0, dataframes.shape[0])

        return dataframes


class BandMath(object):

    @staticmethod
    def _evi(data, sensor, wavelengths, mask=False):

        """
        Enhanced vegetation index
        """

        l = 1.0
        c1 = 6.0
        c2 = 7.5
        g = 2.5

        nir = wavelengths[sensor].nir
        red = wavelengths[sensor].red
        blue = wavelengths[sensor].blue

        result = (g * (data.sel(wavelength=nir) - data.sel(wavelength=red)) /
                  (data.sel(wavelength=nir) * c1 * data.sel(wavelength=red) - c2 * data.sel(wavelength=blue) + l)).fillna(0)

        if mask:
            result = result.where(data['mask'] < 3)

        result = da.where(result < 0, 0, result)
        result = da.where(result > 1, 1, result)

        return dask_to_datarray(data, result, 'evi2')

    @staticmethod
    def _evi2(data, sensor, wavelengths, mask=False):

        """
        Two-band enhanced vegetation index
        """

        nir = wavelengths[sensor].nir
        red = wavelengths[sensor].red

        result = (2.5 * ((data.sel(wavelength=nir) - data.sel(wavelength=red)) /
                         (data.sel(wavelength=nir) + 1.0 + (2.4 * (data.sel(wavelength=red)))))).fillna(0)

        if mask:
            result = result.where(data['mask'] < 3)

        result = da.where(result < 0, 0, result)
        result = da.where(result > 1, 1, result)

        return dask_to_datarray(data, result, 'evi2')

    @staticmethod
    def _nbr(data, sensor, wavelengths, mask=False):

        """
        Normalized burn ratio
        """

        nir = wavelengths[sensor].nir
        swir2 = wavelengths[sensor].swir2

        result = ((data.sel(wavelength=nir) - data.sel(wavelength=swir2)) /
                  (data.sel(wavelength=nir) + data.sel(wavelength=swir2))).fillna(0)

        if mask:
            result = result.where(data['mask'] < 3)

        result = da.where(result < -1, -1, result)
        result = da.where(result > 1, 1, result)

        return dask_to_datarray(data, result, 'nbr')

    @staticmethod
    def _ndvi(data, sensor, wavelengths, mask=False):

        """
        Normalized difference vegetation index
        """

        nir = wavelengths[sensor].nir
        red = wavelengths[sensor].red

        result = ((data.sel(wavelength=nir) - data.sel(wavelength=red)) /
                  (data.sel(wavelength=nir) + data.sel(wavelength=red))).fillna(0)

        if mask:
            result = result.where(data['mask'] < 3)

        result = da.where(result < -1, -1, result)
        result = da.where(result > 1, 1, result)

        return dask_to_datarray(data, result, 'ndvi')

    @staticmethod
    def _wi(data, sensor, wavelengths, mask=False):

        """
        Woody index
        """

        swir1 = wavelengths[sensor].swir1
        red = wavelengths[sensor].red

        result = da.where((data.sel(wavelength=swir1) + data.sel(wavelength=red)) > 0.5, 0,
                          1.0 - ((data.sel(wavelength=swir1) + data.sel(wavelength=red)) / 0.5))

        if mask:
            result = result.where(data['mask'] < 3)

        result = da.where(result < 0, 0, result)
        result = da.where(result > 1, 1, result)

        return dask_to_datarray(data, result, 'wi')


class DatasetBandMath(BandMath):

    def evi2(self, bands='bands', mask=False):
        return self._evi2(self._obj[bands], self._obj.gw.sensor, self._obj.gw.wavelengths, mask=mask)

    def nbr(self, bands='bands', mask=False):
        return self._nbr(self._obj[bands], self._obj.gw.sensor, self._obj.gw.wavelengths, mask=mask)

    def ndvi(self, bands='bands', mask=False):
        return self._ndvi(self._obj[bands], self._obj.gw.sensor, self._obj.gw.wavelengths, mask=mask)

    def wi(self, bands='bands', mask=False):
        return self._wi(self._obj[bands], self._obj.gw.sensor, self._obj.gw.wavelengths, mask=mask)


class DataArrayBandMath(BandMath):

    def evi2(self):
        return self._evi2(self._obj, self._obj.gw.sensor, self._obj.gw.wavelengths)

    def nbr(self):
        return self._nbr(self._obj, self._obj.gw.sensor, self._obj.gw.wavelengths)

    def ndvi(self):
        return self._ndvi(self._obj, self._obj.gw.sensor, self._obj.gw.wavelengths)

    def wi(self):
        return self._wi(self._obj, self._obj.gw.sensor, self._obj.gw.wavelengths)
