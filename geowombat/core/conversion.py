import os
import multiprocessing as multi
import logging

from ..config import config
from ..handler import add_handler
from ..backends.rasterio_ import check_crs
from ..backends.xarray_ import _check_config_globals
from ..backends import transform_crs
from .util import sample_feature
from .util import lazy_wombat

import numpy as np
import dask.array as da
import xarray as xr
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize, shapes
from rasterio.warp import aligned_target
from rasterio.crs import CRS
from shapely.geometry import Polygon, MultiPolygon
from affine import Affine
import pyproj


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def _iter_func(a):
    return a


class Converters(object):

    def bounds_to_coords(self, bounds, dst_crs):

        """
        Converts bounds from longitude and latitude to native map coordinates

        Args:
            bounds (``tuple`` | ``rasterio.coords.BoundingBox``): The lat/lon bounds to transform.
            dst_crs (str, object, or DataArray): The CRS to transform to. It can be provided as a string, a
                CRS instance (e.g., ``pyproj.crs.CRS``), or a ``geowombat.DataArray``.

        Returns:

            ``tuple``:

                (left, bottom, right, top)
        """

        left, bottom, right, top = bounds

        left, bottom = self.lonlat_to_xy(left, bottom, dst_crs)
        right, top = self.lonlat_to_xy(left, top, dst_crs)

        return left, bottom, right, top

    @staticmethod
    def lonlat_to_xy(lon, lat, dst_crs):

        """
        Converts from longitude and latitude to native map coordinates

        Args:
            lon (float): The longitude to convert.
            lat (float): The latitude to convert.
            dst_crs (str, object, or DataArray): The CRS to transform to. It can be provided as a string, a
                CRS instance (e.g., ``pyproj.crs.CRS``), or a ``geowombat.DataArray``.

        Returns:

            ``tuple``:

                (x, y)

        Example:
            >>> import geowombat as gw
            >>> from geowombat.core import lonlat_to_xy
            >>>
            >>> lon, lat = -55.56822206, -25.46214220
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     x, y = lonlat_to_xy(lon, lat, src)
        """

        if isinstance(dst_crs, xr.DataArray):
            dst_crs = dst_crs.crs

        return pyproj.Proj(dst_crs)(lon, lat)

    @staticmethod
    def xy_to_lonlat(x, y, dst_crs):

        """
        Converts from native map coordinates to longitude and latitude

        Args:
            x (float): The x coordinate to convert.
            y (float): The y coordinate to convert.
            dst_crs (str, object, or DataArray): The CRS to transform to. It can be provided as a string, a
                CRS instance (e.g., ``pyproj.crs.CRS``), or a ``geowombat.DataArray``.

        Returns:

            ``tuple``:

                (longitude, latitude)

        Example:
            >>> import geowombat as gw
            >>> from geowombat.core import xy_to_lonlat
            >>>
            >>> x, y = 643944.6956113526, 7183104.984484519
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     lon, lat = xy_to_lonlat(x, y, src)
        """

        if isinstance(dst_crs, xr.DataArray):
            dst_crs = dst_crs.crs

        return pyproj.Proj(dst_crs)(x, y, inverse=True)

    @staticmethod
    def indices_to_coords(col_index, row_index, transform):

        """
        Converts array indices to map coordinates

        Args:
            col_index (float or 1d array): The column index.
            row_index (float or 1d array): The row index.
            transform (Affine, DataArray, or tuple): The affine transform.

        Returns:

            ``tuple``:

                (x, y)

        Example:
            >>> import geowombat as gw
            >>> from geowombat.core import indices_to_coords
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     x, y = indices_to_coords(j, i, src)
        """

        if not isinstance(transform, Affine):

            if isinstance(transform, tuple):
                transform = Affine(*transform)
            elif isinstance(transform, xr.DataArray):
                transform = transform.gw.affine
            else:
                logger.exception('  The transform must be an instance of affine.Affine, an xarray.DataArray, or a tuple')
                raise TypeError

        return transform * (col_index, row_index)

    @staticmethod
    def coords_to_indices(x, y, transform):

        """
        Converts map coordinates to array indices

        Args:
            x (float or 1d array): The x coordinates.
            y (float or 1d array): The y coordinates.
            transform (object): The affine transform.

        Returns:

            ``tuple``:

                (col_index, row_index)

        Example:
            >>> import geowombat as gw
            >>> from geowombat.core import coords_to_indices
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     j, i = coords_to_indices(x, y, src)
        """

        if not isinstance(transform, Affine):

            if isinstance(transform, tuple):
                transform = Affine(*transform)
            elif isinstance(transform, xr.DataArray):
                transform = transform.gw.affine
            else:
                logger.exception('  The transform must be an instance of affine.Affine, an xarray.DataArray, or a tuple')
                raise TypeError

        col_index, row_index = ~transform * (x, y)

        return np.int64(col_index), np.int64(row_index)

    @staticmethod
    def dask_to_xarray(data,
                       dask_data,
                       band_names):

        """
        Converts a Dask array to an Xarray DataArray

        Args:
            data (DataArray): The DataArray with attribute information.
            dask_data (Dask Array): The Dask array to convert.
            band_names (1d array-like): The output band names.

        Returns:
            ``xarray.DataArray``
        """

        if len(dask_data.shape) == 2:
            dask_data = dask_data.reshape(1, dask_data.shape[0], dask_data.shape[1])

        return xr.DataArray(dask_data,
                            dims=('band', 'y', 'x'),
                            coords={'band': band_names,
                                    'y': data.y,
                                    'x': data.x},
                            attrs=data.attrs)

    @staticmethod
    def ndarray_to_xarray(data,
                          numpy_data,
                          band_names):

        """
        Converts a NumPy array to an Xarray DataArray

        Args:
            data (DataArray): The DataArray with attribute information.
            numpy_data (ndarray): The ndarray to convert.
            band_names (1d array-like): The output band names.

        Returns:
            ``xarray.DataArray``
        """

        if len(numpy_data.shape) == 2:
            numpy_data = numpy_data[np.newaxis, :, :]

        return xr.DataArray(da.from_array(numpy_data,
                                          chunks=(1, data.gw.row_chunks, data.gw.col_chunks)),
                            dims=('band', 'y', 'x'),
                            coords={'band': band_names,
                                    'y': data.y,
                                    'x': data.x},
                            attrs=data.attrs)

    @staticmethod
    def xarray_to_xdataset(data_array,
                           band_names,
                           time_names,
                           ycoords=None,
                           xcoords=None,
                           attrs=None):

        """
        Converts an Xarray DataArray to a Xarray Dataset

        Args:
            data_array (DataArray)
            band_names (list)
            time_names (list)
            ycoords (1d array-like)
            xcoords (1d array-like)
            attrs (dict)

        Returns:
            Dataset
        """

        if len(data_array.shape) == 2:
            data_array = data_array.expand_dims('band')

        if len(data_array.shape) == 4:
            n_bands = data_array.shape[1]
        else:
            n_bands = data_array.shape[0]

        if not band_names:

            if n_bands == 1:
                band_names = ['1']
            else:
                band_names = list(map(str, range(1, n_bands + 1)))

        if time_names:

            return xr.Dataset({'bands': (['date', 'band', 'y', 'x'], data_array)},
                              coords={'date': time_names,
                                      'band': band_names,
                                      'y': ('y', ycoords),
                                      'x': ('x', xcoords)},
                              attrs=attrs)

        else:

            return xr.Dataset({'bands': (['band', 'y', 'x'], data_array.data)},
                              coords={'band': band_names,
                                      'y': ('y', data_array.y),
                                      'x': ('x', data_array.x)},
                              attrs=data_array.attrs)

    def prepare_points(self,
                       data,
                       aoi,
                       frac=1.0,
                       all_touched=False,
                       id_column='id',
                       mask=None,
                       n_jobs=8,
                       verbose=0,
                       **kwargs):

        if isinstance(aoi, gpd.GeoDataFrame):
            df = aoi
        else:

            if isinstance(aoi, str):

                if not os.path.isfile(aoi):
                    logger.exception('  The AOI file does not exist.')
                    raise OSError

                df = gpd.read_file(aoi)

            else:
                logger.exception('  The AOI must be a vector file or a GeoDataFrame.')
                raise TypeError

        if id_column not in df.columns.tolist():
            df.loc[:, id_column] = df.index.values

        df_crs = check_crs(df.crs).to_proj4()
        data_crs = check_crs(data.crs).to_proj4()

        # Re-project the data to match the image CRS
        if data_crs != df_crs:
            df = df.to_crs(data_crs)

        if verbose > 0:
            logger.info('  Checking geometry validity ...')

        # Ensure all geometry is valid
        df = df[df['geometry'].apply(lambda x_: x_ is not None)]

        if verbose > 0:
            logger.info('  Checking geometry extent ...')

        # Remove data outside of the image bounds
        if (type(df.iloc[0].geometry) == Polygon) or (type(df.iloc[0].geometry) == MultiPolygon):

            df = gpd.overlay(df,
                             gpd.GeoDataFrame(data=[0],
                                              geometry=[data.gw.geometry],
                                              crs=df_crs),
                             how='intersection').drop(columns=[0])

        else:

            # Clip points to the image bounds
            df = df[df.geometry.intersects(data.gw.geometry)]

        if isinstance(mask, Polygon) or isinstance(mask, MultiPolygon) or isinstance(mask, gpd.GeoDataFrame):

            if isinstance(mask, gpd.GeoDataFrame):

                if CRS.from_dict(mask.crs).to_proj4() != df_crs:
                    mask = mask.to_crs(df_crs)

            if verbose > 0:
                logger.info('  Clipping geometry ...')

            df = df[df.within(mask)]

            if df.empty:
                logger.exception('  No geometry intersects the user-provided mask.')
                raise LookupError

        # Convert polygons to points
        if (type(df.iloc[0].geometry) == Polygon) or (type(df.iloc[0].geometry) == MultiPolygon):

            if verbose > 0:
                logger.info('  Converting polygons to points ...')

            df = self.polygons_to_points(data,
                                         df,
                                         frac=frac,
                                         all_touched=all_touched,
                                         id_column=id_column,
                                         n_jobs=n_jobs,
                                         **kwargs)

        # Ensure a unique index
        df.index = list(range(0, df.shape[0]))

        return df

    @staticmethod
    def polygons_to_points(data,
                           df,
                           frac=1.0,
                           all_touched=False,
                           id_column='id',
                           n_jobs=1,
                           **kwargs):

        """
        Converts polygons to points

        Args:
            data (DataArray or Dataset): The ``xarray.DataArray`` or ``xarray.Dataset``.
            df (GeoDataFrame): The ``geopandas.GeoDataFrame`` containing the geometry to rasterize.
            frac (Optional[float]): A fractional subset of points to extract in each feature.
            all_touched (Optional[bool]): The ``all_touched`` argument is passed to ``rasterio.features.rasterize``.
            id_column (Optional[str]): The 'id' column.
            n_jobs (Optional[int]): The number of features to rasterize in parallel.
            kwargs (Optional[dict]): Keyword arguments passed to ``multiprocessing.Pool().imap``.

        Returns:
            ``geopandas.GeoDataFrame``
        """

        meta = data.gw.meta

        dataframes = []

        df_columns = df.columns.tolist()

        with multi.Pool(processes=n_jobs) as pool:

            for i in pool.imap(_iter_func, range(0, df.shape[0]), **kwargs):

                # Get the current feature's geometry
                dfrow = df.iloc[i]

                point_df = sample_feature(dfrow,
                                          id_column,
                                          df_columns,
                                          data.crs,
                                          data.res,
                                          all_touched,
                                          meta,
                                          frac)

                if not point_df.empty:
                    dataframes.append(point_df)

        dataframes = pd.concat(dataframes, axis=0)

        # Make the points unique
        dataframes.loc[:, 'point'] = np.arange(0, dataframes.shape[0])

        return dataframes

    @staticmethod
    def array_to_polygon(data, mask=None, connectivity=4, num_workers=1):

        """
        Converts an ``xarray.DataArray` to a ``geopandas.GeoDataFrame``

        Args:
            data (DataArray): The ``xarray.DataArray`` to convert.
            mask (Optional[str, numpy ndarray, or rasterio Band object]): Must evaluate to bool (rasterio.bool_ or rasterio.uint8).
                Values of False or 0 will be excluded from feature generation. Note well that this is the inverse sense from
                Numpy's, where a mask value of True indicates invalid data in an array. If source is a Numpy masked array
                and mask is None, the source's mask will be inverted and used in place of mask. if ``mask`` is equal to
                'source', then ``data`` is used as the mask.
            connectivity (Optional[int]): Use 4 or 8 pixel connectivity for grouping pixels into features.
            num_workers (Optional[int]): The number of parallel workers to send to ``dask.compute``.

        Returns:
            ``geopandas.GeoDataFrame``

        Example:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as src:
            >>>
            >>>     # Convert the input image to a GeoDataFrame
            >>>     df = gw.array_to_polygon(src,
            >>>                              mask='source',
            >>>                              num_workers=8)
        """

        if not hasattr(data.gw, 'transform'):
            logger.exception("  The data should have a 'transform' object.")
            raise AttributeError

        if not hasattr(data, 'crs'):
            logger.exception("  The data should have a 'crs' object.")
            raise AttributeError

        if isinstance(mask, str):

            if mask == 'source':
                mask = data.astype('uint8').data.compute(num_workers=num_workers)

        poly_objects = shapes(data.data.compute(num_workers=num_workers),
                              mask=mask,
                              connectivity=connectivity,
                              transform=data.gw.transform)

        poly_data = [(Polygon(p[0]['coordinates'][0]), p[1]) for p in poly_objects]

        poly_geom = list(list(zip(*poly_data))[0])
        poly_values = list(list(zip(*poly_data))[1])

        return gpd.GeoDataFrame(data=poly_values,
                                columns=['value'],
                                geometry=poly_geom,
                                crs=data.crs)

    @lazy_wombat
    def polygon_to_array(self,
                         polygon,
                         col=None,
                         data=None,
                         cellx=None,
                         celly=None,
                         band_name=None,
                         row_chunks=512,
                         col_chunks=512,
                         src_res=None,
                         fill=0,
                         default_value=1,
                         all_touched=True,
                         dtype='uint8',
                         sindex=None,
                         tap=False,
                         bounds_by='intersection'):

        """
        Converts a polygon geometry to an ``xarray.DataArray``.

        Args:
            polygon (GeoDataFrame | str): The ``geopandas.DataFrame`` or file with polygon geometry.
            col (Optional[str]): The column in ``polygon`` you want to assign values from.
                If not set, creates a binary raster.
            data (Optional[DataArray]): An ``xarray.DataArray`` to use as a reference for rasterizing.
            cellx (Optional[float]): The output cell x size.
            celly (Optional[float]): The output cell y size.
            band_name (Optional[list]): The ``xarray.DataArray`` band name.
            row_chunks (Optional[int]): The ``dask`` row chunk size.
            col_chunks (Optional[int]): The ``dask`` column chunk size.
            src_res (Optional[tuple]: A source resolution to align to.
            fill (Optional[int]): Used as fill value for all areas not covered by input geometries
                to ``rasterio.features.rasterize``.
            default_value (Optional[int]): Used as value for all geometries, if not provided in shapes
                to ``rasterio.features.rasterize``.
            all_touched (Optional[bool]): If True, all pixels touched by geometries will be burned in.
                If false, only pixels whose center is within the polygon or that are selected by Bresenham’s line
                algorithm will be burned in. The ``all_touched`` value for ``rasterio.features.rasterize``.
            dtype (Optional[rasterio | numpy data type]): The output data type for ``rasterio.features.rasterize``.
            sindex (Optional[object]): An instanced of ``geopandas.GeoDataFrame.sindex``.
            tap (Optional[bool]): Whether to target align pixels.
            bounds_by (Optional[str]): How to concatenate the output extent. Choices are ['intersection', 'union', 'reference'].

                * reference: Use the bounds of the reference image
                * intersection: Use the intersection (i.e., minimum extent) of all the image bounds
                * union: Use the union (i.e., maximum extent) of all the image bounds

        Returns:
            ``xarray.DataArray``

        Example:
            >>> import geowombat as gw
            >>> import geopandas as gpd
            >>>
            >>> df = gpd.read_file('polygons.gpkg')
            >>>
            >>> # 100x100 cell size
            >>> data = gw.polygon_to_array(df, 100.0, 100.0)
            >>>
            >>> # Align to an existing image
            >>> with gw.open('image.tif') as src:
            >>>     data = gw.polygon_to_array(df, data=src)
        """

        if not band_name:
            band_name = [1]

        if isinstance(polygon, gpd.GeoDataFrame):
            dataframe = polygon
        else:

            if os.path.isfile(polygon):
                dataframe = gpd.read_file(polygon)
            else:
                logger.exception('  The polygon file does not exist.')
                raise OSError

        ref_kwargs = {'bounds': None,
                      'crs': None,
                      'res': None,
                      'tap': tap,
                      'tac': None}

        if config['with_config'] and not isinstance(data, xr.DataArray):

            ref_kwargs = _check_config_globals(data.filename if isinstance(data, xr.DataArray) else None,
                                               bounds_by,
                                               ref_kwargs)

        if isinstance(data, xr.DataArray):

            if dataframe.crs != data.crs:

                # Transform the geometry
                dataframe = dataframe.to_crs(data.crs)

            if not sindex:

                # Get the R-tree spatial index
                sindex = dataframe.sindex

            # Get intersecting features
            int_idx = sorted(list(sindex.intersection(tuple(data.gw.geodataframe.total_bounds.flatten()))))

            if not int_idx:

                return self.dask_to_xarray(data, da.zeros((1, data.gw.nrows, data.gw.ncols),
                                                          chunks=(1, data.gw.row_chunks, data.gw.col_chunks),
                                                          dtype=data.dtype.name),
                                           band_names=band_name)

            # Subset to the intersecting features
            dataframe = dataframe.iloc[int_idx]

            # Clip the geometry
            dataframe = gpd.clip(dataframe, data.gw.geodataframe)

            if dataframe.empty:

                return self.dask_to_xarray(data, da.zeros((1, data.gw.nrows, data.gw.ncols),
                                                          chunks=(1, data.gw.row_chunks, data.gw.col_chunks),
                                                          dtype=data.dtype.name),
                                           band_names=band_name)

            cellx = data.gw.cellx
            celly = data.gw.celly
            row_chunks = data.gw.row_chunks
            col_chunks = data.gw.col_chunks
            src_res = None

            if ref_kwargs['bounds']:

                left, bottom, right, top = ref_kwargs['bounds']

                if 'res' in ref_kwargs and ref_kwargs['res'] is not None:

                    if isinstance(ref_kwargs['res'], tuple) or isinstance(ref_kwargs['res'], list):
                        cellx, celly = ref_kwargs['res']
                    elif isinstance(ref_kwargs['res'], int) or isinstance(ref_kwargs['res'], float):
                        cellx = ref_kwargs['res']
                        celly = ref_kwargs['res']
                    else:
                        logger.exception('The reference resolution must be a tuple, int, or float. Is type %s' % (type(ref_kwargs['res'])))
                        raise TypeError

            else:
                left, bottom, right, top = data.gw.bounds

        else:

            if ref_kwargs['bounds']:

                left, bottom, right, top = ref_kwargs['bounds']

                if 'res' in ref_kwargs and ref_kwargs['res'] is not None:

                    if isinstance(ref_kwargs['res'], tuple) or isinstance(ref_kwargs['res'], list):
                        cellx, celly = ref_kwargs['res']
                    elif isinstance(ref_kwargs['res'], int) or isinstance(ref_kwargs['res'], float):
                        cellx = ref_kwargs['res']
                        celly = ref_kwargs['res']
                    else:
                        logger.exception('The reference resolution must be a tuple, int, or float. Is type %s' % (type(ref_kwargs['res'])))
                        raise TypeError

            else:
                left, bottom, right, top = dataframe.total_bounds.flatten().tolist()

        dst_height = int((top - bottom) / abs(celly))
        dst_width = int((right - left) / abs(cellx))

        dst_transform = Affine(cellx, 0.0, left, 0.0, -celly, top)

        if src_res:

            dst_transform = aligned_target(dst_transform,
                                           dst_width,
                                           dst_height,
                                           src_res)[0]

            left = dst_transform[2]
            top = dst_transform[5]

            dst_transform = Affine(cellx, 0.0, left, 0.0, -celly, top)

        if col:
            shapes = ((geom,value) for geom, value in zip(dataframe.geometry, dataframe[col]))
        else: 
            shapes = dataframe.geometry.values
            
        varray = rasterize(shapes,
                           out_shape=(dst_height, dst_width),
                           transform=dst_transform,
                           fill=fill,
                           default_value=default_value,
                           all_touched=all_touched,
                           dtype=dtype)
        
        cellxh = abs(cellx) / 2.0
        cellyh = abs(celly) / 2.0

        if isinstance(data, xr.DataArray):

            # Ensure the coordinates align
            xcoords = data.x.values
            ycoords = data.y.values

        else:

            xcoords = np.arange(left + cellxh, left + cellxh + dst_width * abs(cellx), cellx)
            ycoords = np.arange(top - cellyh, top - cellyh - dst_height * abs(celly), -celly)

        if xcoords.shape[0] > dst_width:
            xcoords = xcoords[:dst_width]

        if ycoords.shape[0] > dst_height:
            ycoords = ycoords[:dst_height]

        attrs = {'transform': dst_transform[:6],
                 'crs': dataframe.crs,
                 'res': (cellx, celly),
                 'is_tiled': 1}

        return xr.DataArray(data=da.from_array(varray[np.newaxis, :, :],
                                               chunks=(1, row_chunks, col_chunks)),
                            coords={'band': band_name,
                                    'y': ycoords,
                                    'x': xcoords},
                            dims=('band', 'y', 'x'),
                            attrs=attrs)
