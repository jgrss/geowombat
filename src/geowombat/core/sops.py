import os
import itertools
from datetime import datetime
from collections import defaultdict
import logging
import multiprocessing as multi
from pathlib import Path
import tempfile
import typing as T
import warnings

import numpy as np
from scipy.stats import mode as sci_mode
from scipy.spatial import cKDTree
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from rasterio import features
from rasterio.coords import BoundingBox
from affine import Affine
from shapely.geometry import Polygon
from pyproj.enums import WktVersion
from pyproj.exceptions import CRSError

try:
    import arosics
    from geoarray import GeoArray

    AROSICS_INSTALLED = True
except ImportError:
    AROSICS_INSTALLED = False

try:
    import pymorph

    PYMORPH_INSTALLED = True
except ImportError:
    PYMORPH_INSTALLED = False

from ..handler import add_handler
from ..backends.rasterio_ import (
    array_bounds,
    check_crs,
    get_dims_from_bounds,
)
from ..backends.xarray_ import delayed_to_xarray
from .conversion import Converters
from .base import PropertyMixin as _PropertyMixin
from .util import lazy_wombat
from .parallel import ParallelTask
from .base import _client_dummy, _cluster_dummy


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def _remove_near_points(
    dataframe: gpd.GeoDataFrame, r: T.Union[float, int]
) -> gpd.GeoDataFrame:
    """Removes points less than a specified distance to another point.

    Args:
        dataframe (GeoDataFrame): The ``GeoDataFrame`` with point geometry.
        r (float or int): The minimum distance (radius) in the CRS units of ``dataframe``.

    Returns:
        ``geopandas.GeoDataFrame``
    """
    # Setup a KD tree
    tree = cKDTree(np.c_[dataframe.geometry.x, dataframe.geometry.y])

    # Query all pairs within ``min_dist`` of each other
    near_pairs = tree.query_pairs(r=r, output_type='ndarray')

    if near_pairs.shape[0] > 0:

        # Get a list of pairs to remove
        rm_idx = list(sorted(set(near_pairs[:, 0].tolist())))

        return dataframe.query("index != {}".format(rm_idx))

    return dataframe


def _transform_and_shift(
    affine_transform: Affine,
    col_indices: T.Sequence[int],
    row_indices: T.Sequence[int],
    cellxh: float,
    cellyh: float,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Transforms indices to coordinates and applies a half pixel shift.

    Args:
        affine_transform (object): The affine transform.
        col_indices (1d array): The column indices.
        row_indices (1d array): The row indices.
        cellxh (float): The half cell width in the x direction.
        cellyh (float): The half cell width in the y direction.

    Returns:
        ``numpy.ndarray``, ``numpy.ndarray``
    """
    x_coords, y_coords = affine_transform * (col_indices, row_indices)

    x_coords += abs(cellxh)
    y_coords -= abs(cellyh)

    return x_coords, y_coords


class SpatialOperations(_PropertyMixin):
    @staticmethod
    def calc_area(
        data: xr.DataArray,
        values: T.Sequence[T.Union[float, int]],
        op: str = 'eq',
        units: str = 'km2',
        row_chunks: int = None,
        col_chunks: int = None,
        n_workers: int = 1,
        n_threads: int = 1,
        scheduler: str = 'threads',
        n_chunks: int = 100,
    ) -> pd.DataFrame:
        """Calculates the area of data values.

        Args:
            data (DataArray): The ``xarray.DataArray`` to calculate area.
            values (list): A list of values.
            op (Optional[str]): The value sign. Choices are ['gt', 'ge', 'lt', 'le', 'eq'].
            units (Optional[str]): The units to return. Choices are ['km2', 'ha'].
            row_chunks (Optional[int]): The row chunk size to process in parallel.
            col_chunks (Optional[int]): The column chunk size to process in parallel.
            n_workers (Optional[int]): The number of parallel workers for ``scheduler``.
            n_threads (Optional[int]): The number of parallel threads for ``dask.compute()``.
            scheduler (Optional[str]): The parallel task scheduler to use. Choices are ['processes', 'threads', 'mpool'].

                mpool: process pool of workers using ``multiprocessing.Pool``
                processes: process pool of workers using ``concurrent.futures``
                threads: thread pool of workers using ``concurrent.futures``

            n_chunks (Optional[int]): The chunk size of windows. If not given, equal to ``n_workers`` x 50.

        Returns:
            ``pandas.DataFrame``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Read a land cover image with 512x512 chunks
            >>> with gw.open('land_cover.tif', chunks=512) as src:
            >>>
            >>>     df = gw.calc_area(src,
            >>>                       [1, 2, 5],        # calculate the area of classes 1, 2, and 5
            >>>                       units='km2',      # return area in kilometers squared
            >>>                       n_workers=4,
            >>>                       row_chunks=1024,  # iterate over larger chunks to use 512 chunks in parallel
            >>>                       col_chunks=1024)
        """

        def area_func(*args):
            data_chunk, window_id, uvalues, area_units, n_threads = list(
                itertools.chain(*args)
            )

            sqm = abs(data_chunk.gw.celly) * abs(data_chunk.gw.cellx)
            area_conversion = 1e-6 if area_units == 'km2' else 0.0001

            data_totals_ = defaultdict(float)

            for value in uvalues:
                chunk_value_total = (
                    data_chunk.gw.compare(op, value, return_binary=True)
                    .sum(skipna=True)
                    .data.compute(scheduler='threads', num_workers=n_threads)
                )

                data_totals_[value] += (
                    chunk_value_total * sqm
                ) * area_conversion

            return dict(data_totals_)

        pt = ParallelTask(
            data,
            row_chunks=row_chunks,
            col_chunks=col_chunks,
            scheduler=scheduler,
            n_workers=n_workers,
            n_chunks=n_chunks,
        )

        futures = pt.map(area_func, values, units, n_threads)

        # Combine the results
        data_totals = defaultdict(float)
        for future in futures:
            for k, v in future.items():
                data_totals[k] += v

        data_totals = dict(data_totals)
        data_totals = dict(sorted(data_totals.items()))

        df = pd.DataFrame.from_dict(
            data_totals, orient='index', columns=[units]
        )
        df['area_value'] = df.index

        return df

    def sample(
        self,
        data: xr.DataArray,
        method: str = 'random',
        band: T.Union[int, str] = None,
        n: int = None,
        strata: T.Optional[T.Dict[str, T.Union[float, int]]] = None,
        spacing: T.Optional[float] = None,
        min_dist: T.Optional[T.Union[float, int]] = None,
        max_attempts: T.Optional[int] = 10,
        num_workers: T.Optional[int] = 1,
        verbose: T.Optional[int] = 1,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Generates samples from a raster.

        Args:
            data (DataArray): The ``xarray.DataArray`` to extract data from.
            method (Optional[str]): The sampling method. Choices are ['random', 'systematic'].
            band (Optional[int or str]): The band name to extract from. Only required if ``method`` = 'random' and ``strata`` is given.
            n (Optional[int]): The total number of samples. Only required if ``method`` = 'random'.
            strata (Optional[dict]): The strata to sample within. The dictionary key-->value pairs should be {'conditional,value': sample size}.

                E.g.,

                    strata = {'==,1': 0.5, '>=,2': 0.5}
                    ... would sample 50% of total samples within class 1 and 50% of total samples in class >= 2.

                    strata = {'==,1': 10, '>=,2': 20}
                    ... would sample 10 samples within class 1 and 20 samples in class >= 2.

            spacing (Optional[float]): The spacing (in map projection units) when ``method`` = 'systematic'.
            min_dist (Optional[float or int]): A minimum distance allowed between samples. Only applies when ``method`` = 'random'.
            max_attempts (Optional[int]): The maximum numer of attempts to sample points > ``min_dist`` from each other.
            num_workers (Optional[int]): The number of parallel workers for :func:`dask.compute`.
            verbose (Optional[int]): The verbosity level.
            kwargs (Optional[dict]): Keyword arguments passed to ``geowombat.extract``.

        Returns:
            ``geopandas.GeoDataFrame``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Sample 100 points randomly across the image
            >>> with gw.open('image.tif') as src:
            >>>     df = gw.sample(src, n=100)
            >>>
            >>> # Sample points systematically (with 10km spacing) across the image
            >>> with gw.open('image.tif') as src:
            >>>     df = gw.sample(src, method='systematic', spacing=10000.0)
            >>>
            >>> # Sample 50% of 100 in class 1 and 50% in classes >= 2
            >>> strata = {'==,1': 0.5, '>=,2': 0.5}
            >>> with gw.open('image.tif') as src:
            >>>     df = gw.sample(src, band=1, n=100, strata=strata)
            >>>
            >>> # Specify a per-stratum minimum allowed point distance of 1,000 meters
            >>> with gw.open('image.tif') as src:
            >>>     df = gw.sample(src, band=1, n=100, min_dist=1000, strata=strata)
        """
        if method.strip().lower() not in ['random', 'systematic']:
            raise NameError("The method must be 'random' or 'systematic'.")

        if method.strip().lower() == 'systematic':
            if not isinstance(spacing, float):
                if not isinstance(spacing, int):

                    logger.exception(
                        "  If the method is 'systematic', the spacing should be provided as a float or integer."
                    )
                    raise TypeError

        if strata and not band and (method.strip().lower() == 'random'):

            logger.exception(
                '  The band name must be provided with random stratified sampling.'
            )
            raise NameError

        df = None

        if not strata:

            if method == 'systematic':

                x_samples = []
                y_samples = []

                for i in range(0, data.gw.nrows, int(spacing / data.gw.celly)):
                    for j in range(
                        0, data.gw.ncols, int(spacing / data.gw.cellx)
                    ):

                        x_samples.append(j)
                        y_samples.append(i)

                x_samples = np.array(x_samples, dtype='int64')
                y_samples = np.array(y_samples, dtype='int64')

                # Convert the map indices to map coordinates
                x_coords, y_coords = _transform_and_shift(
                    data.gw.meta.affine,
                    x_samples,
                    y_samples,
                    data.gw.cellxh,
                    data.gw.cellyh,
                )

                df = gpd.GeoDataFrame(
                    data=range(0, x_coords.shape[0]),
                    geometry=gpd.points_from_xy(x_coords, y_coords),
                    crs=data.crs,
                    columns=['point'],
                )

            else:

                dfs = None
                sample_size = n
                attempts = 0

                while True:
                    if attempts >= max_attempts:
                        if verbose > 0:
                            logger.warning(
                                '  Max attempts reached. Try relaxing the distance threshold.'
                            )

                        break

                        # Sample directly from the coordinates
                    y_coords = np.random.choice(
                        data.y.values,
                        size=sample_size
                        if sample_size < data.y.values.shape[0]
                        else data.y.values.shape[0] - 1,
                        replace=False,
                    )
                    x_coords = np.random.choice(
                        data.x.values,
                        size=sample_size
                        if sample_size < data.x.values.shape[0]
                        else data.x.values.shape[0] - 1,
                        replace=False,
                    )

                    if isinstance(dfs, gpd.GeoDataFrame):

                        dfs = pd.concat(
                            (
                                dfs,
                                gpd.GeoDataFrame(
                                    data=range(0, x_coords.shape[0]),
                                    geometry=gpd.points_from_xy(
                                        x_coords, y_coords
                                    ),
                                    crs=data.crs,
                                    columns=['point'],
                                ),
                            ),
                            axis=0,
                        )

                    else:

                        dfs = gpd.GeoDataFrame(
                            data=range(0, x_coords.shape[0]),
                            geometry=gpd.points_from_xy(x_coords, y_coords),
                            crs=data.crs,
                            columns=['point'],
                        )

                    if isinstance(min_dist, float) or isinstance(
                        min_dist, int
                    ):

                        # Remove samples within a minimum distance
                        dfn = _remove_near_points(dfs, min_dist)

                        df_diff = dfs.shape[0] - dfn.shape[0]

                        if df_diff > 0:
                            dfs = dfn.copy()
                            sample_size = df_diff

                            attempts += 1

                            continue

                    break

                df = dfs.copy()

        else:

            counter = 0
            dfs = None

            for cond, stratum_size in strata.items():

                conditionals = cond.split(';')

                if len(conditionals) > 1:

                    sign, value = conditionals[0].split(',')
                    sign2, value2 = conditionals[1].split(',')
                    value = float(value)
                    value2 = float(value2)

                else:

                    sign, value = cond.split(',')
                    sign = sign.strip()
                    value = float(value)

                if isinstance(stratum_size, int):
                    sample_size = stratum_size
                else:
                    sample_size = int(np.ceil(n * stratum_size))

                if sample_size == 0:
                    continue

                attempts = 0

                while True:

                    if attempts >= max_attempts:

                        if verbose > 0:
                            logger.warning(
                                "  Max attempts reached for value {:f}. Try relaxing the distance threshold.".format(
                                    value
                                )
                            )

                        if not isinstance(df, gpd.GeoDataFrame):
                            df = dfs.copy()
                        else:
                            df = pd.concat((df, dfs), axis=0)

                        break

                    if len(conditionals) > 1:

                        if (sign == '>') and (sign2 == '<'):
                            valid_samples = da.where(
                                (data.sel(band=band).data > value)
                                & data.sel(band=band).data
                                < value2
                            )
                        elif (sign == '>') and (sign2 == '>'):
                            valid_samples = da.where(
                                (data.sel(band=band).data > value)
                                & data.sel(band=band).data
                                > value2
                            )
                        elif (sign == '>') and (sign2 == '<='):
                            valid_samples = da.where(
                                (data.sel(band=band).data > value)
                                & data.sel(band=band).data
                                <= value2
                            )
                        elif (sign == '>') and (sign2 == '>='):
                            valid_samples = da.where(
                                (data.sel(band=band).data > value)
                                & data.sel(band=band).data
                                >= value2
                            )
                        elif (sign == '<') and (sign2 == '<'):
                            valid_samples = da.where(
                                (data.sel(band=band).data < value)
                                & data.sel(band=band).data
                                < value2
                            )
                        elif (sign == '<') and (sign2 == '>'):
                            valid_samples = da.where(
                                (data.sel(band=band).data < value)
                                & data.sel(band=band).data
                                > value2
                            )
                        elif (sign == '<') and (sign2 == '<='):
                            valid_samples = da.where(
                                (data.sel(band=band).data < value)
                                & data.sel(band=band).data
                                <= value2
                            )
                        elif (sign == '<') and (sign2 == '>='):
                            valid_samples = da.where(
                                (data.sel(band=band).data < value)
                                & data.sel(band=band).data
                                >= value2
                            )
                        elif (sign == '>=') and (sign2 == '<'):
                            valid_samples = da.where(
                                (data.sel(band=band).data >= value)
                                & data.sel(band=band).data
                                < value2
                            )
                        elif (sign == '>=') and (sign2 == '>'):
                            valid_samples = da.where(
                                (data.sel(band=band).data >= value)
                                & data.sel(band=band).data
                                > value2
                            )
                        elif (sign == '>=') and (sign2 == '<='):
                            valid_samples = da.where(
                                (data.sel(band=band).data >= value)
                                & data.sel(band=band).data
                                <= value2
                            )
                        elif (sign == '>=') and (sign2 == '>='):
                            valid_samples = da.where(
                                (data.sel(band=band).data >= value)
                                & data.sel(band=band).data
                                >= value2
                            )
                        elif (sign == '<=') and (sign2 == '<'):
                            valid_samples = da.where(
                                (data.sel(band=band).data <= value)
                                & data.sel(band=band).data
                                < value2
                            )
                        elif (sign == '<=') and (sign2 == '>'):
                            valid_samples = da.where(
                                (data.sel(band=band).data <= value)
                                & data.sel(band=band).data
                                > value2
                            )
                        elif (sign == '<=') and (sign2 == '<='):
                            valid_samples = da.where(
                                (data.sel(band=band).data <= value)
                                & data.sel(band=band).data
                                <= value2
                            )
                        elif (sign == '<=') and (sign2 == '>='):
                            valid_samples = da.where(
                                (data.sel(band=band).data <= value)
                                & data.sel(band=band).data
                                >= value2
                            )
                        else:
                            logger.exception(
                                '  The conditional sign was not recognized.'
                            )
                            raise NameError

                    else:

                        if sign == '>':
                            valid_samples = da.where(
                                data.sel(band=band).data > value
                            )
                        elif sign == '>=':
                            valid_samples = da.where(
                                data.sel(band=band).data >= value
                            )
                        elif sign == '<':
                            valid_samples = da.where(
                                data.sel(band=band).data < value
                            )
                        elif sign == '<=':
                            valid_samples = da.where(
                                data.sel(band=band).data <= value
                            )
                        elif sign == '==':
                            valid_samples = da.where(
                                data.sel(band=band).data == value
                            )
                        else:
                            logger.exception(
                                "  The conditional sign was not recognized. Use one of '>', '>=', '<', '<=', or '=='."
                            )
                            raise NameError

                    valid_samples = dask.compute(
                        valid_samples,
                        num_workers=num_workers,
                        scheduler='threads',
                    )[0]

                    y_samples = valid_samples[0]
                    x_samples = valid_samples[1]

                    if y_samples.shape[0] > 0:

                        ssize = (
                            sample_size
                            if sample_size < y_samples.shape[0]
                            else y_samples.shape[0] - 1
                        )

                        if ssize == 0:
                            break

                        # Get indices within the stratum
                        idx = np.random.choice(
                            range(0, y_samples.shape[0]),
                            size=ssize,
                            replace=False,
                        )

                        y_samples = y_samples[idx]
                        x_samples = x_samples[idx]

                        # Convert the map indices to map coordinates
                        x_coords, y_coords = _transform_and_shift(
                            data.gw.meta.affine,
                            x_samples,
                            y_samples,
                            data.gw.cellxh,
                            data.gw.cellyh,
                        )

                        if isinstance(dfs, gpd.GeoDataFrame):

                            dfs = pd.concat(
                                (
                                    dfs,
                                    gpd.GeoDataFrame(
                                        data=range(0, x_coords.shape[0]),
                                        geometry=gpd.points_from_xy(
                                            x_coords, y_coords
                                        ),
                                        crs=data.crs,
                                        columns=['point'],
                                    ),
                                ),
                                axis=0,
                            )

                        else:

                            dfs = gpd.GeoDataFrame(
                                data=range(0, x_coords.shape[0]),
                                geometry=gpd.points_from_xy(
                                    x_coords, y_coords
                                ),
                                crs=data.crs,
                                columns=['point'],
                            )

                        if isinstance(min_dist, float) or isinstance(
                            min_dist, int
                        ):

                            # Remove samples within a minimum distance
                            dfn = _remove_near_points(dfs, min_dist)

                            df_diff = dfs.shape[0] - dfn.shape[0]

                            if df_diff > 0:
                                dfs = dfn.copy()
                                sample_size = df_diff

                                attempts += 1

                                continue

                        if not isinstance(df, gpd.GeoDataFrame):
                            df = dfs.copy()
                        else:
                            df = pd.concat((df, dfs), axis=0)

                        dfs = None

                    break

                counter += 1

        if isinstance(df, gpd.GeoDataFrame):
            return self.extract(data, df, **kwargs)
        else:
            return None

    def extract(
        self,
        data: xr.DataArray,
        aoi: T.Union[str, Path, gpd.GeoDataFrame],
        bands: T.Union[int, T.Sequence[int]] = None,
        time_names: T.Sequence[T.Any] = None,
        band_names: T.Sequence[T.Any] = None,
        frac: float = 1.0,
        min_frac_area: T.Optional[T.Union[float, int]] = None,
        all_touched: T.Optional[bool] = False,
        id_column: T.Optional[str] = 'id',
        time_format: T.Optional[str] = '%Y%m%d',
        mask: T.Optional[T.Union[Polygon, gpd.GeoDataFrame]] = None,
        n_jobs: T.Optional[int] = 8,
        verbose: T.Optional[int] = 0,
        n_workers: T.Optional[int] = 1,
        n_threads: T.Optional[int] = -1,
        use_client: T.Optional[bool] = False,
        address: T.Optional[str] = None,
        total_memory: T.Optional[int] = 24,
        processes: T.Optional[bool] = False,
        pool_kwargs: T.Optional[dict] = None,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Extracts data within an area or points of interest. Projections do
        not need to match, as they are handled 'on-the-fly'.

        Args:
            data (DataArray): The ``xarray.DataArray`` to extract data from.
            aoi (str or GeoDataFrame): A file or ``geopandas.GeoDataFrame`` to extract data frame.
            bands (Optional[int or 1d array-like]): A band or list of bands to extract.
                If not given, all bands are used. Bands should be GDAL-indexed (i.e., the first band is 1, not 0).
            band_names (Optional[list]): A list of band names. Length should be the same as `bands`.
            time_names (Optional[list]): A list of time names.
            frac (Optional[float]): A fractional subset of points to extract in each polygon feature.
            min_frac_area (Optional[int | float]): A minimum polygon area to use ``frac``. Otherwise, use all samples
                within a polygon.
            all_touched (Optional[bool]): The ``all_touched`` argument is passed to :func:`rasterio.features.rasterize`.
            id_column (Optional[str]): The id column name.
            time_format (Optional[str]): The ``datetime`` conversion format if ``time_names`` are ``datetime`` objects.
            mask (Optional[GeoDataFrame or Shapely Polygon]): A ``shapely.geometry.Polygon`` mask to subset to.
            n_jobs (Optional[int]): The number of features to rasterize in parallel.
            verbose (Optional[int]): The verbosity level.
            n_workers (Optional[int]): The number of process workers. Only applies when ``use_client`` = ``True``.
            n_threads (Optional[int]): The number of thread workers. Only applies when ``use_client`` = ``True``.
            use_client (Optional[bool]): Whether to use a ``dask`` client.
            address (Optional[str]): A cluster address to pass to client. Only used when ``use_client`` = ``True``.
            total_memory (Optional[int]): The total memory (in GB) required when ``use_client`` = ``True``.
            processes (Optional[bool]): Whether to use process workers with the ``dask.distributed`` client.
                Only applies when ``use_client`` = ``True``.
            pool_kwargs (Optional[dict]): Keyword arguments passed to :func:`multiprocessing.Pool().imap`.
            kwargs (Optional[dict]): Keyword arguments passed to :func:`dask.compute`.

        Returns:
            ``geopandas.GeoDataFrame``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif') as src:
            >>>     df = gw.extract(src, 'poly.gpkg')
            >>>
            >>> # On a cluster
            >>> # Use a local cluster
            >>> with gw.open('image.tif') as src:
            >>>     df = gw.extract(src, 'poly.gpkg', use_client=True, n_threads=16)
            >>>
            >>> # Specify the client address with a local cluster
            >>> with LocalCluster(
            >>>     n_workers=1,
            >>>     threads_per_worker=8,
            >>>     scheduler_port=0,
            >>>     processes=False,
            >>>     memory_limit='4GB'
            >>> ) as cluster:
            >>>
            >>>     with gw.open('image.tif') as src:
            >>>         df = gw.extract(
            >>>             src,
            >>>             'poly.gpkg',
            >>>             use_client=True,
            >>>             address=cluster
            >>>         )
        """
        if not pool_kwargs:
            pool_kwargs = {}

        mem_per_core = int(total_memory / n_workers)

        if n_threads == -1:
            n_threads = multi.cpu_count()

        if use_client:
            if address:
                cluster_object = _cluster_dummy
            else:
                cluster_object = LocalCluster
            client_object = Client
        else:
            cluster_object = _cluster_dummy
            client_object = _client_dummy

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
                bands_idx = list(range(0, data.gw.nbands))

        if isinstance(aoi, gpd.GeoDataFrame):
            if id_column not in aoi.columns.tolist():
                aoi['id'] = aoi.index.values

        df = converters.prepare_points(
            data,
            aoi,
            frac=frac,
            min_frac_area=min_frac_area,
            all_touched=all_touched,
            id_column=id_column,
            mask=mask,
            n_jobs=n_jobs,
            verbose=verbose,
            **pool_kwargs,
        )

        if df.empty:
            return df

        if verbose > 0:
            logger.info('  Extracting data ...')

        # Convert the geometry map coordinates to array indices
        x, y = converters.coords_to_indices(
            df.geometry.x.values, df.geometry.y.values, data.gw.affine
        )

        if y.max() >= data.gw.nrows:
            idx_nonnull = np.where(y < data.gw.nrows)
            y = y[idx_nonnull]
            x = x[idx_nonnull]

        if x.max() >= data.gw.ncols:
            idx_nonnull = np.where(x < data.gw.ncols)
            y = y[idx_nonnull]
            x = x[idx_nonnull]

        yidx = xr.DataArray(y, dims='z')
        xidx = xr.DataArray(x, dims='z')

        # Get the raster values for each point
        # TODO: allow neighbor indexing
        if use_client:
            with cluster_object(
                n_workers=n_workers,
                threads_per_worker=n_threads,
                scheduler_port=0,
                processes=processes,
                memory_limit=f'{mem_per_core}GB',
            ) as cluster:
                cluster_address = address if address else cluster
                with client_object(address=cluster_address) as client:
                    res = client.gather(
                        client.compute(
                            data.isel(band=bands_idx, y=yidx, x=xidx).data
                        )
                    )

        else:
            res = data.isel(band=bands_idx, y=yidx, x=xidx).gw.compute(
                **kwargs
            )

        if (len(res.shape) == 1) or (
            (len(res.shape) == 2) and (res.shape[0] == 1)
        ):
            df[band_names[0]] = res.flatten()
        elif len(res.shape) == 2:
            # `res` is shaped [dimensions x samples]
            df = pd.concat(
                (df, pd.DataFrame(data=res.T, columns=band_names)), axis=1
            )
        else:
            if time_names:
                if isinstance(time_names[0], datetime):
                    time_names = [t.strftime(time_format) for t in time_names]
            else:
                time_names = [f't{t}' for t in range(1, res.shape[0] + 1)]

            band_names_concat = []
            for t in time_names:
                for b in band_names:
                    band_names_concat.append(f'{t}_{b}')

            # `res` is shaped [time x bands x samples]
            ntime, nbands, nsamples = res.shape

            df = pd.concat(
                (
                    df,
                    pd.DataFrame(
                        data=res.T.squeeze()
                        if nbands == 1
                        else res.reshape(ntime * nbands, nsamples).T,
                        columns=band_names_concat,
                    ),
                ),
                axis=1,
            )

        return df

    def clip_by_polygon(
        self,
        data: xr.DataArray,
        df: T.Union[str, Path, gpd.GeoDataFrame],
        query: T.Optional[str] = None,
        mask_data: T.Optional[bool] = False,
        expand_by: T.Optional[int] = 0,
    ) -> xr.DataArray:
        """Clips a DataArray by vector polygon geometry.

        Args:
            data (DataArray): The ``xarray.DataArray`` to subset.
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
            >>>     ds = gw.clip_by_polygon(ds, df, query="Id == 1")
        """
        if isinstance(df, (Path, str)):
            if not Path(df).is_file():
                raise FileExistsError(f'{str(df)} does not exist.')
            df = gpd.read_file(df)

        if query is not None:
            df = df.query(query)

        data_crs_ = check_crs(data.crs)
        df_crs_ = check_crs(df.crs)

        # Re-project the DataFrame to match the image CRS
        if data_crs_ != df_crs_:
            df = df.to_crs(data_crs_)
        # Get the target bounds
        left, bottom, right, top = df.total_bounds
        # Get the target dimensions
        dst_bounds = BoundingBox(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
        )
        align_height, align_width = get_dims_from_bounds(
            dst_bounds, (data.gw.cellx, data.gw.celly)
        )
        align_transform = Affine(
            data.gw.cellx, 0.0, left, 0.0, -data.gw.celly, top
        )
        # Get the new bounds
        new_left, new_bottom, new_right, new_top = array_bounds(
            align_height, align_width, align_transform
        )

        if expand_by > 0:
            new_left -= data.gw.cellx * expand_by
            new_bottom -= data.gw.celly * expand_by
            new_right += data.gw.cellx * expand_by
            new_top += data.gw.celly * expand_by

        # Subset the array
        data = self.subset(
            data,
            left=new_left,
            bottom=new_bottom,
            right=new_right,
            top=new_top,
        )

        if mask_data:
            converters = Converters()
            mask = converters.polygon_to_array(df, data=data)

            data = (
                (data * mask.where(lambda x: x == 1))
                .fillna(data.gw.nodataval)
                .assign_attrs(**data.attrs)
            )

        return data

    def clip(
        self,
        data: xr.DataArray,
        df: T.Union[str, gpd.GeoDataFrame],
        query: T.Optional[str] = None,
        mask_data: T.Optional[bool] = False,
        expand_by: T.Optional[int] = 0,
    ) -> xr.DataArray:
        """Clips a DataArray by vector polygon geometry.

        .. deprecated:: 2.1.7
            Use :func:`geowombat.clip_by_polygon`.

        Args:
            data (DataArray): The ``xarray.DataArray`` to subset.
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
        warnings.warn(
            'The method clip() will be deprecated in >=2.2.0. Use clip_by_polygon() instead.',
            DeprecationWarning,
            stacklevel=2,
        )

        if isinstance(df, (Path, str)):
            if not Path(df).is_file():
                raise FileExistsError(f'{str(df)} does not exist.')
            df = gpd.read_file(df)

        if query is not None:
            df = df.query(query)

        data_crs_ = check_crs(data.crs)
        df_crs_ = check_crs(df.crs)

        # Re-project the DataFrame to match the image CRS
        if data_crs_ != df_crs_:
            df = df.to_crs(data_crs_)
        # Get the target bounds
        left, bottom, right, top = df.total_bounds
        # Get the target dimensions
        dst_bounds = BoundingBox(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
        )
        align_height, align_width = get_dims_from_bounds(
            dst_bounds, (data.gw.cellx, data.gw.celly)
        )
        align_transform = Affine(
            data.gw.cellx, 0.0, left, 0.0, -data.gw.celly, top
        )
        # Get the new bounds
        new_left, new_bottom, new_right, new_top = array_bounds(
            align_height, align_width, align_transform
        )

        if expand_by > 0:
            new_left -= data.gw.cellx * expand_by
            new_bottom -= data.gw.celly * expand_by
            new_right += data.gw.cellx * expand_by
            new_top += data.gw.celly * expand_by

        # Subset the array
        data = self.subset(
            data,
            left=new_left,
            bottom=new_bottom,
            right=new_right,
            top=new_top,
        )

        if mask_data:
            converters = Converters()
            mask = converters.polygon_to_array(df, data=data)

            data = (
                (data * mask.where(lambda x: x == 1))
                .fillna(data.gw.nodataval)
                .assign_attrs(**data.attrs)
            )

        return data

    @staticmethod
    @lazy_wombat
    def mask(
        data: xr.DataArray,
        dataframe: T.Union[str, Path, gpd.GeoDataFrame],
        query: T.Optional[str] = None,
        keep: T.Optional[str] = 'in',
    ) -> xr.DataArray:
        """Masks a DataArray by vector polygon geometry.

        Args:
            data (DataArray): The ``xarray.DataArray`` to mask.
            dataframe (GeoDataFrame or str): The ``geopandas.GeoDataFrame`` or filename to use for masking.
            query (Optional[str]): A query to apply to ``dataframe``.
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
        if isinstance(dataframe, str) and os.path.isfile(dataframe):
            dataframe = gpd.read_file(dataframe)

        if query:
            dataframe = dataframe.query(query)

        data_crs_ = check_crs(data.crs)
        df_crs_ = check_crs(dataframe.crs)

        if data_crs_ != df_crs_:
            dataframe = dataframe.to_crs(data_crs_)

        # Rasterize the geometry and store as a DataArray
        mask = xr.DataArray(
            data=da.from_array(
                features.rasterize(
                    list(dataframe.geometry.values),
                    out_shape=(data.gw.nrows, data.gw.ncols),
                    transform=data.gw.transform,
                    fill=0,
                    out=None,
                    all_touched=True,
                    default_value=1,
                    dtype='int32',
                ),
                chunks=(data.gw.row_chunks, data.gw.col_chunks),
            ),
            dims=['y', 'x'],
            coords={'y': data.y.values, 'x': data.x.values},
        )

        # Return the masked array
        if keep == 'out':
            return data.where(mask != 1)
        else:
            return data.where(mask == 1)

    @staticmethod
    def replace(
        data: xr.DataArray, to_replace: T.Dict[int, T.Union[int, str]]
    ) -> xr.DataArray:
        """Replace values given in to_replace with value.

        Args:
            data (DataArray): The ``xarray.DataArray`` to recode.
            to_replace (dict): How to find the values to replace. Dictionary mappings should be given
                as {from: to} pairs. If ``to_replace`` is an integer/string mapping, the to string should be 'mode'.

                {1: 5}:
                    recode values of 1 to 5

                {1: 'mode'}:
                    recode values of 1 to the polygon mode

        Returns:
            ``xarray.DataArray``
        """

        attrs = data.attrs.copy()
        dtype = data.dtype.name

        if not isinstance(to_replace, dict):
            raise TypeError(
                "The replace values must be a dictionary of {from: to} mappings."
            )

        data = data.astype('int64')

        for k, v in to_replace.items():
            data = xr.where(data == k, v + 100000, data)

        for v in to_replace.values():
            data = xr.where(data == v + 100000, data - 100000, data)

        return data.assign_attrs(**attrs).astype(dtype)

    @lazy_wombat
    def recode(
        self,
        data: xr.DataArray,
        polygon: T.Union[str, Path, gpd.GeoDataFrame],
        to_replace: dict,
        num_workers: T.Optional[int] = 1,
    ) -> xr.DataArray:
        """Recodes a DataArray with polygon mappings.

        Args:
            data (DataArray): The ``xarray.DataArray`` to recode.
            polygon (GeoDataFrame | str): The ``geopandas.DataFrame`` or file with polygon geometry.
            to_replace (dict): How to find the values to replace. Dictionary mappings should be given
                as {from: to} pairs. If ``to_replace`` is an integer/string mapping, the to string should be 'mode'.

                {1: 5}:
                    recode values of 1 to 5

                {1: 'mode'}:
                    recode values of 1 to the polygon mode
            num_workers (Optional[int]): The number of parallel Dask workers (only used if ``to_replace``
                has a 'mode' mapping).

        Returns:
            ``xarray.DataArray``
        """
        dtype = data.dtype.name
        attrs = data.attrs.copy()

        converters = Converters()

        poly_array = converters.polygon_to_array(polygon, data=data)

        for k, v in to_replace.items():

            if isinstance(v, str) and (v.lower() == 'mode'):

                data_array_np = (
                    data.squeeze()
                    .where(poly_array.squeeze() == 1)
                    .data.compute(num_workers=num_workers)
                )

                to_replace[k] = int(
                    sci_mode(
                        data_array_np, axis=None, nan_policy='omit'
                    ).mode.flatten()
                )

        return (
            xr.where(poly_array == 1, self.replace(data, to_replace), data)
            .assign_attrs(**attrs)
            .astype(dtype)
        )

    @staticmethod
    def subset(
        data: xr.DataArray,
        left: T.Optional[float] = None,
        top: T.Optional[float] = None,
        right: T.Optional[float] = None,
        bottom: T.Optional[float] = None,
        rows: T.Optional[int] = None,
        cols: T.Optional[int] = None,
        center: T.Optional[bool] = False,
        mask_corners: T.Optional[bool] = False,
    ) -> xr.DataArray:
        """Subsets a DataArray.

        Args:
            data (DataArray): The ``xarray.DataArray`` to subset.
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
            >>> import geowombat as gw
            >>>
            >>> with gw.open('image.tif', chunks=512) as ds:
            >>>     ds_sub = gw.subset(
            >>>         ds,
            >>>         left=-263529.884,
            >>>         top=953985.314,
            >>>         rows=2048,
            >>>         cols=2048
            >>>     )
        """
        if isinstance(right, (float, int)):
            cols = int((right - left) / data.gw.celly)

        if not isinstance(cols, int):
            logger.exception(
                '  The right coordinate or columns must be specified.'
            )
            raise NameError

        if isinstance(bottom, (float, int)):
            rows = int((top - bottom) / data.gw.celly)

        if not isinstance(rows, int):
            logger.exception(
                '  The bottom coordinate or rows must be specified.'
            )
            raise NameError

        converters = Converters()
        x_idx, y_idx = converters.coords_to_indices(
            left + abs(data.gw.cellxh),
            top - abs(data.gw.cellyh),
            data.gw.affine,
        )
        if center:
            y_idx += (rows / 2.0) * abs(data.gw.celly)
            x_idx -= (cols / 2.0) * abs(data.gw.cellx)
        ds_sub = data.isel(
            indexers={
                'band': slice(0, None),
                'y': slice(y_idx, y_idx + rows),
                'x': slice(x_idx, x_idx + cols),
            }
        )
        ds_sub.attrs['transform'] = ds_sub.gw.transform

        if mask_corners:
            if PYMORPH_INSTALLED:
                try:
                    disk = da.from_array(
                        pymorph.sedisk(r=int(rows / 2.0))[:rows, :cols],
                        chunks=ds_sub.data.chunksize,
                    ).astype('uint8')
                    ds_sub = ds_sub.where(disk == 1)

                except ValueError:
                    logger.warning(
                        '  Cannot mask corners without a square subset.'
                    )

            else:
                logger.warning('  Cannot mask corners without Pymorph.')

        return ds_sub

    @staticmethod
    def coregister(
        target: T.Union[str, Path, xr.DataArray],
        reference: T.Union[str, Path, xr.DataArray],
        band_names_reference: T.Optional[T.Sequence[str]] = None,
        band_names_target: T.Optional[T.Sequence[str]] = None,
        wkt_version: T.Optional[str] = None,
        **kwargs,
    ) -> xr.DataArray:
        """Co-registers an image, or images, using AROSICS.

        While the required inputs are DataArrays, the intermediate results are stored as NumPy arrays.
        Therefore, memory usage is constrained to the size of the input data. Dask is not used for any of the
        computation in this function.

        Args:
            target (DataArray or str): The target ``xarray.DataArray`` or file name to co-register to ``reference``.
            reference (DataArray or str): The reference ``xarray.DataArray`` or file name used to co-register ``target``.
            band_names_reference (Optional[list | tuple]): Band names to open for the reference data.
            band_names_target (Optional[list | tuple]): Band names to open for the target data.
            wkt_version (Optional[str]): The WKT version to use with ``to_wkt()``.
            kwargs (Optional[dict]): Keyword arguments passed to ``arosics``.

        Reference:
            https://pypi.org/project/arosics

        Returns:
            ``xarray.DataArray``

        Example:
            >>> import geowombat as gw
            >>>
            >>> # Co-register a single image to a reference image
            >>> with gw.open('target.tif') as tar, gw.open('reference.tif') as ref:
            >>>     results = gw.coregister(
            >>>         tar, ref, q=True, ws=(512, 512), max_shift=3, CPUs=4
            >>>     )
            >>>
            >>> # or
            >>>
            >>> results = gw.coregister(
            >>>     'target.tif',
            >>>     'reference.tif',
            >>>     q=True,
            >>>     ws=(512, 512),
            >>>     max_shift=3,
            >>>     CPUs=4
            >>> )
        """
        import geowombat as gw_

        if not AROSICS_INSTALLED:
            logger.exception(
                '\nAROSICS must be installed to co-register data.\nSee https://pypi.org/project/arosics for details'
            )
            raise NameError

        if isinstance(reference, str):
            if not os.path.isfile(reference):
                logger.exception('  The reference file does not exist.')
                raise OSError

            with gw_.open(reference) as reference:
                pass

        if isinstance(target, str):
            if not os.path.isfile(target):
                logger.exception('  The target file does not exist.')
                raise OSError

            with gw_.open(target) as target:
                pass

        if wkt_version is None:
            wkt_version = WktVersion.WKT2_2019.value
        elif isinstance(wkt_version, str):
            try:
                wkt_version = WktVersion[wkt_version].value
            except KeyError as e:
                logger.exception(e)
                raise KeyError

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / '_reference.tif'
            tar_path = Path(tmp) / '_target.tif'
            ref_kwargs = (
                {'engine': 'h5netcdf'}
                if str(reference.filename).lower().endswith('.nc')
                else {}
            )
            with gw_.open(
                reference.filename,
                band_names=band_names_reference,
                chunks={
                    'band': reference.gw.band_chunks,
                    'y': reference.gw.row_chunks,
                    'x': reference.gw.col_chunks,
                },
                **ref_kwargs,
            ) as ref_src:
                ref_src = ref_src.assign_attrs(
                    {
                        'crs': ref_src.gw.crs_to_pyproj.to_wkt(
                            version=wkt_version
                        )
                    }
                )
                if 'nodata' in kwargs:
                    ref_src = ref_src.fillna(
                        kwargs['nodata'][0]
                    ).gw.assign_nodata_attrs(kwargs['nodata'][0])
                ref_src.gw.save(ref_path, overwrite=True, log_progress=False)
            tar_kwargs = (
                {'engine': 'h5netcdf'}
                if str(target.filename).lower().endswith('.nc')
                else {}
            )
            with gw_.open(
                target.filename,
                band_names=band_names_target,
                chunks={
                    'band': target.gw.band_chunks,
                    'y': target.gw.row_chunks,
                    'x': target.gw.col_chunks,
                },
                **tar_kwargs,
            ) as tar_src:
                tar_src = tar_src.assign_attrs(
                    {
                        'crs': tar_src.gw.crs_to_pyproj.to_wkt(
                            version=wkt_version
                        )
                    }
                )
                if 'nodata' in kwargs:
                    tar_src = tar_src.fillna(
                        kwargs['nodata'][1]
                    ).gw.assign_nodata_attrs(kwargs['nodata'][1])
                tar_src.gw.save(tar_path, overwrite=True, log_progress=False)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                try:
                    cr = arosics.COREG(
                        GeoArray(
                            GeoArray(str(ref_path)), projection=ref_src.crs
                        ),
                        GeoArray(
                            GeoArray(str(tar_path)), projection=tar_src.crs
                        ),
                        **kwargs,
                    )
                except CRSError as e:
                    logger.exception(e)
                    raise AttributeError(
                        "Try using an integer EPSG format (e.g., data.attrs['crs'] = 'epsg:32621')"
                    )

            try:
                cr.calculate_spatial_shifts()
            except RuntimeError as e:
                logger.warning(
                    f"  Could not co-register the data because -> {e}"
                )
                return target

            # Apply spatial shifts
            shift_info = cr.correct_shifts()

        left = shift_info['updated geotransform'][0]
        top = shift_info['updated geotransform'][3]
        # Update the transform
        transform = (target.gw.cellx, 0.0, left, 0.0, -target.gw.celly, top)
        target.attrs['transform'] = transform

        # Check if multi-band
        if len(shift_info['arr_shifted'].shape) > 2:
            data = shift_info['arr_shifted'].transpose(2, 0, 1)
        else:
            data = shift_info['arr_shifted'][np.newaxis]
        yidx = 1
        xidx = 2
        # Get the updated coordinates
        ycoords = np.linspace(
            top - target.gw.cellyh,
            top - target.gw.cellyh - (data.shape[yidx] * target.gw.celly),
            data.shape[yidx],
        )
        xcoords = np.linspace(
            left + target.gw.cellxh,
            left + target.gw.cellxh + (data.shape[xidx] * target.gw.cellx),
            data.shape[xidx],
        )
        target_attrs = target.attrs.copy()
        if hasattr(cr, 'x_shift_px'):
            target_attrs['x_shift_px'] = cr.x_shift_px
        if hasattr(cr, 'y_shift_px'):
            target_attrs['y_shift_px'] = cr.y_shift_px

        return xr.DataArray(
            data=da.from_array(
                data,
                chunks=(
                    target.gw.band_chunks,
                    target.gw.row_chunks,
                    target.gw.col_chunks,
                ),
            ),
            dims=('band', 'y', 'x'),
            coords={
                'band': target.band.values.tolist(),
                'y': ycoords,
                'x': xcoords,
            },
            attrs=target_attrs,
        )
