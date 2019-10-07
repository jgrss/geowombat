import os
from collections import namedtuple
import multiprocessing as multi

from ..errors import logger
from ..moving import moving_window

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterio import features
from rasterio.crs import CRS
import shapely
from shapely.geometry import Polygon
from affine import Affine
from tqdm import tqdm


shapely.speedups.enable()


def project_coords(x, y, src_crs, dst_crs):

    """
    Projects coordinates to a new CRS

    Args:
        x (1d array-like)
        y (1d array-like)
        src_crs (str, dict, object)
        dst_crs (str, dict, object)

    Returns:
        ``numpy.array``, ``numpy.array``
    """

    df_tmp = gpd.GeoDataFrame(np.arange(0, x.shape[0]),
                              geometry=gpd.points_from_xy(x, y),
                              crs=src_crs)

    df_tmp = df_tmp.to_crs(dst_crs)

    return df_tmp.x.values, df_tmp.y.values


def get_geometry_info(geometry, res):

    """
    Gets information from a Shapely geometry object

    Args:
        geometry (object): A `shapely.geometry` object.
        res (float): The cell resolution for the affine transform.

    Returns:
        Geometry information (namedtuple)
    """

    GeomInfo = namedtuple('GeomInfo', 'left bottom right top shape affine')

    minx, miny, maxx, maxy = geometry.bounds
    out_shape = (int((maxy - miny) / res), int((maxx - minx) / res))

    return GeomInfo(left=minx,
                    bottom=miny,
                    right=maxx,
                    top=maxy,
                    shape=out_shape,
                    affine=Affine(res, 0.0, minx, 0.0, -res, maxy))


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
    def moving(data, band_names=None, stat='mean', w=3, n_jobs=1):

        """
        Applies a moving window function over Dask array blocks

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            band_names (int or str or list): The output band name(s).
            stat (Optional[str]): The statistic to compute.
            w (Optional[int]): The moving window size (in pixels).
            n_jobs (Optional[int]): The number of bands to process in parallel.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as ds:
            >>>     ds = gw.moving(ds, band_names=['red'])
        """

        if not isinstance(data, xr.DataArray):
            logger.exception('  The input data must be an Xarray DataArray.')

        y = data.y.values
        x = data.x.values
        attrs = data.attrs

        data_array = data.data

        if n_jobs <= 0:

            logger.warning('  The number of parallel jobs should be a positive integer, so setting n_jobs=1.')
            n_jobs = 1

        hw = int(w / 2.0)

        def _move_func(block_data):

            if max(block_data.shape) <= hw:
                return block_data
            else:
                return moving_window(block_data, stat=stat, w=w, n_jobs=n_jobs)

        if len(data.shape) == 2:
            out_shape = (1,) + data_array.shape
        else:
            out_shape = data_array.shape

        result = data_array.reshape(out_shape).astype('float64').map_overlap(_move_func,
                                                                             depth=hw,
                                                                             trim=True,
                                                                             boundary='reflect',
                                                                             dtype='float64').reshape(out_shape)

        if isinstance(band_names, np.ndarray):
            if isinstance(band_names.tolist(), str):
                band_names = [band_names.tolist()]

        if not isinstance(band_names, list):
            if not isinstance(band_names, np.ndarray):
                band_names = np.arange(1, data_array.shape[0]+1)

        return xr.DataArray(data=result,
                            dims=('band', 'y', 'x'),
                            coords={'band': band_names,
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
                                       transform=geom_info.affine,
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
    def ij_to_xy(j, i, transform):

        """
        Converts map coordinates to array indices

        Args:
            j (float or 1d array): The column index.
            i (float or 1d array): The row index.
            transform (object): The affine transform.

        Returns:
            x, y
        """

        return transform * (j, i)

    @staticmethod
    def xy_to_ij(x, y, transform):

        """
        Converts map coordinates to array indices

        Args:
            x (float or 1d array): The x coordinates.
            y (float or 1d array): The y coordinates.
            transform (object): The affine transform.

        Returns:
            j, i
        """

        return ~transform * (x, y)

    def prepare_points(self,
                       data,
                       aoi,
                       frac=1.0,
                       all_touched=False,
                       mask=None,
                       n_jobs=8,
                       verbose=0):

        if isinstance(aoi, gpd.GeoDataFrame):
            df = aoi
        else:

            if isinstance(aoi, str):

                if not os.path.isfile(aoi):
                    logger.exception('  The AOI file does not exist.')

                df = gpd.read_file(aoi)

            else:
                logger.exception('  The AOI must be a vector file or a GeoDataFrame.')

        if data.crs != CRS.from_dict(df.crs).to_proj4():

            # Re-project the data to match the image CRS
            df = df.to_crs(data.crs)

        if verbose > 0:
            logger.info('  Checking geometry validity ...')

        # Ensure all geometry is valid
        df = df[df['geometry'].apply(lambda x_: x_ is not None)]

        if verbose > 0:
            logger.info('  Checking geometry extent ...')

        # Remove data outside of the image bounds
        if type(df.iloc[0].geometry) == Polygon:

            df = gpd.overlay(df,
                             gpd.GeoDataFrame(data=[0],
                                              geometry=[data.gw.meta.geometry],
                                              crs=df.crs),
                             how='intersection')

        else:

            # Clip points to the image bounds
            df = df[df.geometry.intersects(data.gw.meta.geometry.unary_union)]

        if isinstance(mask, Polygon) or isinstance(mask, gpd.GeoDataFrame):

            if isinstance(mask, gpd.GeoDataFrame):

                if CRS.from_dict(mask.crs).to_proj4() != CRS.from_dict(df.crs).to_proj4():
                    mask = mask.to_crs(df.crs)

            if verbose > 0:
                logger.info('  Clipping geometry ...')

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

            if verbose > 0:
                logger.info('  Converting polygons to points ...')

            df = self.polygons_to_points(data,
                                         df,
                                         frac=frac,
                                         all_touched=all_touched,
                                         n_jobs=n_jobs)

        return df

    @staticmethod
    def polygons_to_points(data,
                           df,
                           frac=1.0,
                           all_touched=False,
                           n_jobs=1):

        """
        Converts polygons to points

        Args:
            data (DataArray or Dataset): The ``xarray.DataArray`` or ``xarray.Dataset`.
            df (GeoDataFrame): The ``geopandas.GeoDataFrame`` containing the geometry to rasterize.
            frac (Optional[float]): A fractional subset of points to extract in each feature.
            all_touched (Optional[bool]): The ``all_touched`` argument is passed to ``rasterio.features.rasterize``.
            n_jobs (Optional[int]): The number of features to rasterize in parallel.

        Returns:
            ``geopandas.GeoDataFrame``
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
