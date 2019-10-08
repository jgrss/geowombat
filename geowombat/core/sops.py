import os

from ..errors import logger
from ..util.rasterio_ import align_bounds, array_bounds
from .util import Converters

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask.array as da
from rasterio.crs import CRS
from rasterio import features

try:
    import pymorph
    PYMORPH_INSTALLED = True
except:
    PYMORPH_INSTALLED = False


class SpatialOperations(object):

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

    @staticmethod
    def extract(data,
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
            data (DataArray): An ``xarray.DataArray`` to extract data from.
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
            >>>     df = gw.extract(ds, 'poly.gpkg')
        """

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

        df = Converters().prepare_points(data,
                                         aoi,
                                         frac=frac,
                                         all_touched=all_touched,
                                         mask=mask,
                                         n_jobs=n_jobs,
                                         verbose=verbose)

        if verbose > 0:
            logger.info('  Extracting data ...')

        # Convert the map coordinates to indices
        x, y = Converters().xy_to_ij(df.geometry.x.values,
                                     df.geometry.y.values,
                                     data.transform)

        if shape_len == 2:
            vidx = (y, x)
        else:

            vidx = (bands_idx, y.tolist(), x.tolist())

            for b in range(0, shape_len - 3):
                vidx = (slice(0, None),) + vidx

        # Get the raster values for each point
        res = data.data.vindex[vidx].compute(**kwargs)

        # TODO: reshape output ``res`` instead of iterating over dimensions
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
                    enum = list(range(0, data.gw.bands))

            else:
                enum = list(range(0, data.gw.bands))

            if len(res.shape) > 2:

                for t in range(0, res.shape[1]):

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
                        df['bd{:d}'.format(i + 1)] = res[:, i]

        return df

    def clip(self,
             data,
             df,
             query=None):

        """
        Clips a DataArray by vector polygon geometry

        Args:
            data (DataArray): An ``xarray.DataArray`` to subset.
            df (GeoDataFrame or str): The ``geopandas.GeoDataFrame`` or filename to clip to.
            query (Optional[str]): A query to apply to ``df``.

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

        if data.crs != CRS.from_dict(df.crs).to_proj4():

            # Re-project the DataFrame to match the image CRS
            df = df.to_crs(data.crs)

        left, bottom, right, top = df.total_bounds

        # Align the geometry grid to the array
        align_transform, align_width, align_height = align_bounds(left,
                                                                  bottom,
                                                                  right,
                                                                  top,
                                                                  data.res)

        # Get the new bounds
        new_bounds = array_bounds(align_height, align_width, align_transform)

        # Subset the array
        data = self.subset(data,
                           left=new_bounds.bounds.left,
                           bottom=new_bounds.bounds.bottom,
                           right=new_bounds.bounds.right,
                           top=new_bounds.bounds.top,
                           chunksize=data.data.chunksize)

        # Rasterize the geometry and store as a DataArray
        mask = xr.DataArray(data=da.from_array(features.rasterize(df.geometry.values.tolist(),
                                                                  out_shape=(align_height, align_width),
                                                                  transform=align_transform,
                                                                  fill=0,
                                                                  out=None,
                                                                  all_touched=True,
                                                                  default_value=1,
                                                                  dtype='int32'),
                                               chunks=(data.gw.row_chunks, data.gw.col_chunks)),
                            dims=['y', 'x'],
                            coords={'y': data.y.values,
                                    'x': data.x.values})

        # Return the clipped array
        return data.where(mask == 1)

    def subset(self,
               data,
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
            data (DataArray): An ``xarray.DataArray`` to subset.
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
            >>>     ds_sub = gw.subset(ds, -263529.884, 953985.314, rows=2048, cols=2048)
        """

        if isinstance(right, int) or isinstance(right, float):
            cols = int((right - left) / data.gw.celly)

        if not isinstance(cols, int):
            logger.exception('  The right coordinate or columns must be specified.')

        if isinstance(bottom, int) or isinstance(bottom, float):
            rows = int((top - bottom) / data.gw.celly)

        if not isinstance(rows, int):
            logger.exception('  The bottom coordinate or rows must be specified.')

        x_idx = np.linspace(left, left + (cols * data.gw.celly), cols)
        y_idx = np.linspace(top, top - (rows * data.gw.celly), rows)

        if center:

            y_idx += ((rows / 2.0) * data.gw.celly)
            x_idx -= ((cols / 2.0) * data.gw.celly)

        if chunksize:
            chunksize_ = chunksize
        else:
            # TODO: fix
            chunksize_ = (self.band_chunks, self.row_chunks, self.col_chunks)

        ds_sub = data.sel(y=y_idx,
                          x=x_idx,
                          method='nearest').chunk(chunksize_)

        if mask_corners:

            if PYMORPH_INSTALLED:

                if len(chunksize_) == 2:
                    chunksize_pym = chunksize_
                else:
                    chunksize_pym = chunksize_[1:]

                try:
                    disk = da.from_array(pymorph.sedisk(r=int(rows/2.0))[:rows, :cols], chunks=chunksize_pym).astype('uint8')
                    ds_sub = ds_sub.where(disk == 1)
                except:
                    logger.warning('  Cannot mask corners without a square subset.')

            else:
                logger.warning('  Cannot mask corners without Pymorph.')

        # Update the left and top coordinates
        transform = list(data.transform)
        transform[2] = ds_sub.gw.left
        transform[5] = ds_sub.gw.top

        ds_sub.attrs['transform'] = tuple(transform)

        return ds_sub
