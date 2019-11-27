import os
import math
import itertools

from ..errors import logger
from ..backends.rasterio_ import align_bounds, array_bounds, aligned_target
from .util import Converters
from .base import PropertyMixin as _PropertyMixin

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask.array as da
from rasterio.crs import CRS
from rasterio import features
from rasterio.warp import calculate_default_transform
from affine import Affine

try:
    import pymorph
    PYMORPH_INSTALLED = True
except:
    PYMORPH_INSTALLED = False


class SpatialOperations(_PropertyMixin):

    @staticmethod
    def sample(data, frac=0.1, nodata=0):

        """
        Creates samples from the array

        Args:
            data (DataArray): An ``xarray.DataArray`` to stratify.
            frac (Optional[float]): The sample fraction for each block.
            nodata (Optional[int]): The 'no data' value, which will be ignored.

        Returns:
            ``geopandas.GeoDataFrame``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as ds:
            >>>     df = gw.sample(ds)
        """

        def _sample_func(block_data):

            results = list()

            x, y = np.meshgrid(block_data.x.values,
                               block_data.y.values)

            for cidx in block_data.unique():

                if cidx == nodata:
                    continue

                idx = np.where(block_data == cidx)
                xx = x[idx]
                yy = y[idx]

                n_samps = idx[0].shape[0]
                n_samps_frac = int(n_samps*frac)
                drange = list(range(0, n_samps))

                rand_idx = np.random.choice(drange, size=n_samps_frac, replace=False)

                xx = xx.flatten()[rand_idx]
                yy = yy.flatten()[rand_idx]

                df_tmp = gpd.GeoDataFrame(np.arange(0, n_samps_frac),
                                          geometry=gpd.points_from_xy(xx, yy),
                                          crs=data.crs)

                results.append(df_tmp)

            return pd.concat(results, axis=0)

    def extract(self,
                data,
                aoi,
                bands=None,
                time_names=None,
                band_names=None,
                frac=1.0,
                all_touched=False,
                id_column='id',
                mask=None,
                n_jobs=8,
                verbose=0,
                **kwargs):

        """
        Extracts data within an area or points of interest. Projections do not need to match,
        as they are handled 'on-the-fly'.

        Args:
            data (DataArray): An ``xarray.DataArray`` to extract data from.
            aoi (str or GeoDataFrame): A file or ``geopandas.GeoDataFrame`` to extract data frame.
            bands (Optional[int or 1d array-like]): A band or list of bands to extract.
                If not given, all bands are used. *Bands should be GDAL-indexed (i.e., the first band is 1, not 0).
            band_names (Optional[list]): A list of band names. Length should be the same as `bands`.
            time_names (Optional[list]): A list of time names.
            frac (Optional[float]): A fractional subset of points to extract in each polygon feature.
            all_touched (Optional[bool]): The ``all_touched`` argument is passed to ``rasterio.features.rasterize``.
            id_column (Optional[str]): The id column name.
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
            >>>     df = gw.extract(ds, 'poly.gpkg')
        """

        sensor = self.check_sensor(data, return_error=False)

        band_names = self.check_sensor_band_names(data, sensor, band_names)

        converters = Converters()

        shape_len = data.gw.ndims

        if isinstance(bands, list):
            bands_idx = (np.array(bands, dtype='int64') - 1).tolist()
        elif isinstance(bands, np.ndarray):
            bands_idx = (bands - 1).tolist()
        elif isinstance(bands, int):
            bands_idx = [bands]
        else:

            if shape_len > 2:
                bands_idx = slice(0, None)

        df = converters.prepare_points(data,
                                       aoi,
                                       frac=frac,
                                       all_touched=all_touched,
                                       id_column=id_column,
                                       mask=mask,
                                       n_jobs=n_jobs,
                                       verbose=verbose)

        if verbose > 0:
            logger.info('  Extracting data ...')

        # Convert the map coordinates to indices
        x, y = converters.xy_to_ij(df.geometry.x.values,
                                   df.geometry.y.values,
                                   data.transform)

        vidx = (y.tolist(), x.tolist())

        if shape_len > 2:

            vidx = (bands_idx,) + vidx

            if shape_len > 3:

                # The first 3 dimensions are (bands, rows, columns)
                # TODO: allow user-defined time slice?
                for b in range(0, shape_len - 3):
                    vidx = (slice(0, None),) + vidx

        # Get the raster values for each point
        # TODO: allow neighbor indexing
        res = data.data.vindex[vidx].compute(**kwargs)

        if len(res.shape) == 1:
            df[band_names[0]] = res.flatten()
        elif len(res.shape) == 2:

            # `res` is shaped [samples x dimensions]
            df = pd.concat((df, pd.DataFrame(data=res, columns=band_names)), axis=1)

        else:

            # `res` is shaped [samples x time x dimensions]
            if time_names:
                time_names = list(itertools.chain(*[[t]*res.shape[2] for t in time_names]))
            else:
                time_names = list(itertools.chain(*[['t{:d}'.format(t)]*res.shape[2] for t in range(1, res.shape[1]+1)]))

            band_names_concat = ['{}_{}'.format(a, b) for a, b in list(zip(time_names, band_names*res.shape[1]))]

            df = pd.concat((df,
                            pd.DataFrame(data=res.reshape(res.shape[0],
                                                          res.shape[1]*res.shape[2]),
                                         columns=band_names_concat)),
                           axis=1)

        return df

    def clip(self,
             data,
             df,
             query=None,
             mask_data=False,
             expand_by=0):

        """
        Clips a DataArray by vector polygon geometry

        Args:
            data (DataArray): An ``xarray.DataArray`` to subset.
            df (GeoDataFrame or str): The ``geopandas.GeoDataFrame`` or filename to clip to.
            query (Optional[str]): A query to apply to ``df``.
            mask_data (Optional[bool]): Whether to mask values outside of the ``df`` geometry envelope.
            expand_by (Optional[int]): Expand the clip array bounds by ``expand_by`` pixels on each side.

        Returns:
             ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as ds:
            >>>     ds = gw.clip(ds, df, query="Id == 1")
            >>>
            >>> # or
            >>>
            >>> with gw.open('image.tif') as ds:
            >>>     ds = ds.gw.clip(df, query="Id == 1")
        """

        if isinstance(df, str) and os.path.isfile(df):
            df = gpd.read_file(df)

        if query:
            df = df.query(query)

        try:

            if data.crs.strip() != CRS.from_dict(df.crs).to_proj4().strip():

                # Re-project the DataFrame to match the image CRS
                df = df.to_crs(data.crs)

        except:

            if data.crs.strip() != CRS.from_proj4(df.crs).to_proj4().strip():
                df = df.to_crs(data.crs)

        row_chunks = data.gw.row_chunks
        col_chunks = data.gw.col_chunks

        left, bottom, right, top = df.total_bounds

        # Align the geometry array grid
        align_transform, align_width, align_height = align_bounds(left,
                                                                  bottom,
                                                                  right,
                                                                  top,
                                                                  data.res)

        # Get the new bounds
        new_left, new_bottom, new_right, new_top = array_bounds(align_height,
                                                                align_width,
                                                                align_transform)

        if expand_by > 0:

            new_left -= data.gw.cellx*expand_by
            new_bottom -= data.gw.celly*expand_by
            new_right += data.gw.cellx*expand_by
            new_top += data.gw.celly*expand_by

        # Subset the array
        data = self.subset(data,
                           left=new_left,
                           bottom=new_bottom,
                           right=new_right,
                           top=new_top)

        if mask_data:

            # Rasterize the geometry and store as a DataArray
            mask = xr.DataArray(data=da.from_array(features.rasterize(list(df.geometry.values),
                                                                      out_shape=(align_height, align_width),
                                                                      transform=align_transform,
                                                                      fill=0,
                                                                      out=None,
                                                                      all_touched=True,
                                                                      default_value=1,
                                                                      dtype='int32'),
                                                   chunks=(row_chunks, col_chunks)),
                                dims=['y', 'x'],
                                coords={'y': data.y.values,
                                        'x': data.x.values})

            # Return the clipped array
            return data.where(mask == 1)

        else:
            return data

    def mask(self,
             data,
             df,
             query=None,
             keep='in'):

        """
        Masks a DataArray by vector polygon geometry

        Args:
            data (DataArray): An ``xarray.DataArray`` to mask.
            df (GeoDataFrame or str): The ``geopandas.GeoDataFrame`` or filename to use for masking.
            query (Optional[str]): A query to apply to ``df``.
            keep (Optional[str]): If ``keep`` = 'in', mask values outside of the geometry (keep inside).
                Otherwise, if ``keep`` = 'out', mask values inside (keep outside).

        Returns:
             ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as ds:
            >>>     ds = ds.gw.mask(df)
        """

        if isinstance(df, str) and os.path.isfile(df):
            df = gpd.read_file(df)

        if query:
            df = df.query(query)

        try:

            if data.crs.strip() != CRS.from_dict(df.crs).to_proj4().strip():

                # Re-project the DataFrame to match the image CRS
                df = df.to_crs(data.crs)

        except:

            if data.crs.strip() != CRS.from_proj4(df.crs).to_proj4().strip():
                df = df.to_crs(data.crs)

        # Rasterize the geometry and store as a DataArray
        mask = xr.DataArray(data=da.from_array(features.rasterize(list(df.geometry.values),
                                                                  out_shape=(data.gw.nrows, data.gw.ncols),
                                                                  transform=data.transform,
                                                                  fill=0,
                                                                  out=None,
                                                                  all_touched=True,
                                                                  default_value=1,
                                                                  dtype='int32'),
                                               chunks=(data.gw.row_chunks, data.gw.col_chunks)),
                            dims=['y', 'x'],
                            coords={'y': data.y.values,
                                    'x': data.x.values})

        # Return the masked array
        if keep == 'out':
            return data.where(mask != 1)
        else:
            return data.where(mask == 1)

    @staticmethod
    def subset(data,
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
            data (DataArray): An ``xarray.DataArray`` to subset.
            left (Optional[float]): The left coordinate.
            top (Optional[float]): The top coordinate.
            right (Optional[float]): The right coordinate.
            bottom (Optional[float]): The bottom coordinate.
            rows (Optional[int]): The number of output rows.
            cols (Optional[int]): The number of output rows.
            center (Optional[bool]): Whether to center the subset on ``left`` and ``top``.
            mask_corners (Optional[bool]): Whether to mask corners (*requires ``pymorph``).

        Returns:
            ``xarray.DataArray``

        Example:
            >>> geowombat as gw
            >>>
            >>> with gw.open('image.tif', chunks=512) as ds:
            >>>     ds_sub = gw.subset(ds, left=-263529.884, top=953985.314, rows=2048, cols=2048)
        """

        if isinstance(right, int) or isinstance(right, float):
            cols = int((right - left) / data.gw.celly)

        if not isinstance(cols, int):
            logger.exception('  The right coordinate or columns must be specified.')

        if isinstance(bottom, int) or isinstance(bottom, float):
            rows = int((top - bottom) / data.gw.celly)

        if not isinstance(rows, int):
            logger.exception('  The bottom coordinate or rows must be specified.')

        x_idx = np.linspace(math.ceil(left), math.ceil(left) + (cols * abs(data.gw.cellx)), cols) + abs(data.gw.cellxh)
        y_idx = np.linspace(math.ceil(top), math.ceil(top) - (rows * abs(data.gw.celly)), rows) - abs(data.gw.cellyh)

        if center:

            y_idx += ((rows / 2.0) * abs(data.gw.celly))
            x_idx -= ((cols / 2.0) * abs(data.gw.cellx))

        ds_sub = data.sel(y=y_idx,
                          x=x_idx,
                          method='nearest')

        if mask_corners:

            if PYMORPH_INSTALLED:

                try:

                    disk = da.from_array(pymorph.sedisk(r=int(rows/2.0))[:rows, :cols],
                                         chunks=ds_sub.data.chunksize).astype('uint8')
                    ds_sub = ds_sub.where(disk == 1)

                except:
                    logger.warning('  Cannot mask corners without a square subset.')

            else:
                logger.warning('  Cannot mask corners without Pymorph.')

        # Update the left and top coordinates
        transform = list(data.transform)

        transform[2] = x_idx[0]
        transform[5] = y_idx[0]

        # Align the coordinates to the target grid
        dst_transform, dst_width, dst_height = aligned_target(Affine(*transform),
                                                              ds_sub.shape[1],
                                                              ds_sub.shape[0],
                                                              data.res)

        ds_sub.attrs['transform'] = dst_transform

        return ds_sub
