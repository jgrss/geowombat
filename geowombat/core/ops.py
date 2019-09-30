from ..errors import logger
from .util import Converters

import numpy as np
import geopandas as gpd
from rasterio.crs import CRS
from shapely.geometry import Polygon

try:
    import pymorph
    PYMORPH_INSTALLED = True
except:
    PYMORPH_INSTALLED = False


class SpatialOperations(object):

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

        if isinstance(aoi, gpd.GeoDataFrame):
            df = aoi
        else:

            if isinstance(aoi, str):

                if not os.path.isfile(aoi):
                    logger.exception('  The AOI file does not exist.')

                df = gpd.read_file(aoi)

            else:
                logger.exception('  The AOI must be a vector file or a GeoDataFrame.')

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
        df = gpd.overlay(df,
                         gpd.GeoDataFrame(data=[0],
                                          geometry=[data.gw.meta.geometry],
                                          crs=df.crs),
                         how='intersection')

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

            df = Converters().polygons_to_points(data,
                                                 df,
                                                 frac=frac,
                                                 all_touched=all_touched,
                                                 n_jobs=n_jobs)

        if verbose > 0:
            logger.info('  Extracting data ...')

        x, y = df.geometry.x.values, df.geometry.y.values

        left = data.gw.left
        top = data.gw.top

        x = np.int64(np.round(np.abs(x - left) / data.gw.celly))
        y = np.int64(np.round(np.abs(top - y) / data.gw.celly))

        if shape_len == 2:
            vidx = (y, x)
        else:

            vidx = (bands_idx, y.tolist(), x.tolist())

            for b in range(0, shape_len - 3):
                vidx = (slice(0, None),) + vidx

        res = data.data.vindex[vidx].compute(**kwargs)

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

                for t in range(0, data.gw.rows):

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

    @staticmethod
    def subset(data,
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
            data (DataArray): An ``xarray.DataArray`` to extract data from.
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
            >>>     ds_sub = gw.subset(ds, -263529.884, 953985.314, rows=2048, cols=2048)
        """

        if isinstance(right, int) or isinstance(right, float):
            cols = int((right - left) / data.gw.celly)

        if not isinstance(cols, int):
            raise AttributeError('The right coordinate or columns must be specified.')

        if isinstance(bottom, int) or isinstance(bottom, float):
            rows = int((top - bottom) / data.gw.celly)

        if not isinstance(rows, int):
            raise AttributeError('The bottom coordinate or rows must be specified.')

        x_idx = np.linspace(left, left + (cols * data.gw.celly), cols)
        y_idx = np.linspace(top, top - (rows * data.gw.celly), rows)

        if center:

            y_idx += ((rows / 2.0) * data.gw.celly)
            x_idx -= ((cols / 2.0) * data.gw.celly)

        if chunksize:
            chunksize_ = chunksize
        else:
            chunksize_ = (self.gw.band_chunks, self.gw.row_chunks, self.gw.col_chunks)

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
        transform[2] = x_idx[0]
        transform[5] = y_idx[0]

        ds_sub.attrs['transform'] = tuple(transform)

        return ds_sub
