import fnmatch
import logging
import os
import typing as T
from collections import OrderedDict, namedtuple
from datetime import datetime
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from rasterio import features
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from threadpoolctl import threadpool_limits

from ..handler import add_handler
from ..moving import moving_window

try:
    from dateparser.search import search_dates

    DATEPARSER_INSTALLED = True
except ImportError:
    DATEPARSER_INSTALLED = False


logger = logging.getLogger(__name__)
logger = add_handler(logger)


def lazy_wombat(func):
    func.wombat_func_ = True
    return func


def estimate_array_mem(ntime, nbands, nrows, ncols, dtype):

    """Estimates the size of an array in-memory.

    Args:
        ntime (int): The number of time dimensions.
        nbands (int): The number of band dimensions.
        nrows (int): The number of row dimensions.
        ncols (int): The number of column dimensions.
        dtype (str): The data type.

    Returns:
        ``int`` in MB
    """

    return (
        np.random.random((ntime, nbands, nrows, ncols)).astype(dtype).nbytes
        * 1e-6
    )


def parse_filename_dates(
    filenames: T.Sequence[T.Union[str, Path]]
) -> T.Sequence:

    """Parses dates from file names.

    Args:
        filenames (list): A list of files to parse.

    Returns:
        ``list``
    """
    date_filenames = []
    for fn in filenames:
        __, f_name = os.path.split(fn)
        f_base, __ = os.path.splitext(f_name)

        dt = None

        if DATEPARSER_INSTALLED:
            try:
                __, dt = list(
                    zip(
                        *search_dates(
                            ' '.join(' '.join(f_base.split('_')).split('-')),
                            settings={
                                'DATE_ORDER': 'YMD',
                                'STRICT_PARSING': False,
                                'PREFER_LANGUAGE_DATE_ORDER': False,
                            },
                        )
                    )
                )

            except Exception:
                return list(range(1, len(filenames) + 1))

        if not dt:
            return list(range(1, len(filenames) + 1))

        date_filenames.append(dt[0])

    return date_filenames


def parse_wildcard(string: str) -> T.List[Path]:

    """Parses a search wildcard from a string.

    Args:
        string (str): The string to parse.

    Returns:
        ``list``
    """

    if os.path.dirname(string):
        dir_name, wildcard = os.path.split(string)
    else:

        dir_name = '.'
        wildcard = string

    matches = sorted(list(Path(dir_name).glob(wildcard)))

    if not matches:
        logger.exception(
            '  There were no images found with the string search.'
        )

    return matches


def sort_images_by_date(
    image_path: Path,
    image_wildcard: str,
    date_pos: int = 0,
    date_start: int = 0,
    date_end: int = 8,
    split_by: str = '_',
    date_format: str = '%Y%m%d',
    file_list: T.Optional[T.Sequence[Path]] = None,
    prepend_str: T.Optional[str] = None,
) -> OrderedDict:
    """Sorts images by date.

    Args:
        image_path (Path): The image directory.
        image_wildcard (str): The image search wildcard.
        date_pos (int): The date starting position in the file name.
        date_start (int): The date starting position in the split.
        date_end (int): The date ending position in the split.
        split_by (Optional[str]): How to split the file name.
        date_format (Optional[str]): The date format for :func:`datetime.datetime.strptime`.
        file_list (Optional[list of Paths]): A file list of names to sort. Overrides ``image_path``.
        prepend_str (Optional[str]): A string to prepend to each filename.

    Returns:
        ``collections.OrderedDict``

    Example:
        >>> from pathlib import Path
        >>> from geowombat.core import sort_images_by_date
        >>>
        >>> # image example: LC08_L1TP_176038_20190108_20190130_01_T1.tif
        >>> image_path = Path('/path/to/images')
        >>>
        >>> image_dict = sort_images_by_date(image_path, '*.tif', 3, 0, 8)
        >>> image_names = list(image_dict.keys())
        >>> time_names = list(image_dict.values())
    """

    if file_list:
        fl = file_list
    else:
        fl = list(image_path.glob(image_wildcard))

    if prepend_str:
        fl = [Path(f'{prepend_str}{str(fn)}') for fn in fl]

    dates = [
        datetime.strptime(
            fn.name.split(split_by)[date_pos][date_start:date_end], date_format
        )
        for fn in fl
    ]

    return OrderedDict(
        sorted(
            dict(zip([str(fn) for fn in fl], dates)).items(),
            key=lambda x_: x_[1],
        )
    )


def project_coords(x, y, src_crs, dst_crs, return_as='1d', **kwargs):
    """Projects coordinates to a new CRS.

    Args:
        x (1d array-like): The x coordinates.
        y (1d array-like): The y coordinates.
        src_crs (str, dict, object): The source CRS.
        dst_crs (str, dict, object): The destination CRS.
        return_as (Optional[str]): How to return the coordinates. Choices are ['1d', '2d'].
        kwargs (Optional[dict]): Keyword arguments passed to ``rasterio.warp.reproject``.

    Returns:
        ``numpy.array``, ``numpy.array`` or ``xr.DataArray``
    """

    if return_as == '1d':

        df_tmp = gpd.GeoDataFrame(
            np.arange(0, x.shape[0]),
            geometry=gpd.points_from_xy(x, y),
            crs=src_crs,
        )

        df_tmp = df_tmp.to_crs(dst_crs)

        return df_tmp.geometry.x.values, df_tmp.geometry.y.values

    else:

        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype='float64')

        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype='float64')

        yy = np.meshgrid(x, y)[1]

        latitudes = np.zeros(yy.shape, dtype='float64')

        src_transform = from_bounds(
            x[0], y[-1], x[-1], y[0], latitudes.shape[1], latitudes.shape[0]
        )

        west, south, east, north = transform_bounds(
            src_crs, CRS(dst_crs), x[0], y[-1], x[-1], y[0]
        )
        dst_transform = from_bounds(
            west, south, east, north, latitudes.shape[1], latitudes.shape[0]
        )

        latitudes = reproject(
            yy,
            destination=latitudes,
            src_transform=src_transform,
            dst_transform=dst_transform,
            src_crs=CRS.from_epsg(src_crs.split(':')[1]),
            dst_crs=CRS(dst_crs),
            **kwargs,
        )[0]

        return xr.DataArray(
            data=da.from_array(
                latitudes[np.newaxis, :, :], chunks=(1, 512, 512)
            ),
            dims=('band', 'y', 'x'),
            coords={
                'band': ['lat'],
                'y': ('y', latitudes[:, 0]),
                'x': ('x', np.arange(1, latitudes.shape[1] + 1)),
            },
        )


def get_geometry_info(geometry: object, res: tuple) -> namedtuple:
    """Gets information from a Shapely geometry object.

    Args:
        geometry (object): A ``shapely.geometry`` object.
        res (tuple): The cell resolution for the affine transform.

    Returns:
        Geometry information as ``namedtuple``.
    """
    GeomInfo = namedtuple('GeomInfo', 'left bottom right top shape affine')

    resx, resy = abs(res[1]), abs(res[0])
    minx, miny, maxx, maxy = geometry.bounds
    if isinstance(minx, str):
        minx, miny, maxx, maxy = geometry.bounds.values[0]

    out_shape = (int((maxy - miny) / resx), int((maxx - minx) / resy))

    return GeomInfo(
        left=minx,
        bottom=miny,
        right=maxx,
        top=maxy,
        shape=out_shape,
        affine=Affine(resy, 0.0, minx, 0.0, -resx, maxy),
    )


def get_file_extension(filename: str) -> namedtuple:
    """Gets file and directory name information.

    Args:
        filename (str): The file name.

    Returns:
        Name information as ``namedtuple``.
    """

    FileNames = namedtuple('FileNames', 'd_name f_name f_base f_ext')

    d_name, f_name = os.path.splitext(filename)
    f_base, f_ext = os.path.split(f_name)

    return FileNames(d_name=d_name, f_name=f_name, f_base=f_base, f_ext=f_ext)


def n_rows_cols(pixel_index: int, block_size: int, rows_cols: int) -> int:

    """Adjusts block size for the end of image rows and columns.

    Args:
        pixel_index (int): The current pixel row or column index.
        block_size (int): The image block size.
        rows_cols (int): The total number of rows or columns in the image.

    Returns:
        Adjusted block size as ``int``.
    """
    return (
        block_size
        if (pixel_index + block_size) < rows_cols
        else rows_cols - pixel_index
    )


class Chunks(object):
    @staticmethod
    def get_chunk_dim(chunksize):
        return '{:d}d'.format(len(chunksize))

    def check_chunktype(self, chunksize: int, output: str = '3d'):
        if chunksize is None:
            return chunksize

        if isinstance(chunksize, int):
            chunksize = (-1, chunksize, chunksize)

        chunk_len = len(chunksize)
        output_len = int(output[0])

        if not isinstance(chunksize, tuple):
            if not isinstance(chunksize, dict):
                if not isinstance(chunksize, int):
                    logger.warning(
                        '  The chunksize parameter should be a tuple, dictionary, or integer.'
                    )

        if chunk_len != output_len:
            return self.check_chunksize(chunksize, output=output)
        else:
            return chunksize

    @staticmethod
    def check_chunksize(chunksize: int, output: str = '3d') -> tuple:
        chunk_len = len(chunksize)
        output_len = int(output[0])

        if chunk_len != output_len:
            if (chunk_len == 2) and (output_len == 3):
                return (-1,) + chunksize
            elif (chunk_len == 3) and (output_len == 2):
                return chunksize[1:]

        return chunksize


class MapProcesses(object):
    @staticmethod
    def moving(
        data: xr.DataArray,
        stat: str = 'mean',
        perc: T.Union[float, int] = 50,
        w: int = 3,
        nodata: T.Optional[T.Union[float, int]] = None,
        weights: T.Optional[bool] = False,
    ) -> xr.DataArray:
        """Applies a moving window function over Dask array blocks.

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            stat (Optional[str]): The statistic to compute.
                Choices are ['mean', 'std', 'var', 'min', 'max', 'perc'].
            perc (Optional[int]): The percentile to return if ``stat`` = 'perc'.
            w (Optional[int]): The moving window size (in pixels).
            nodata (Optional[int or float]): A 'no data' value to ignore.
            weights (Optional[bool]): Whether to weight values by distance from window center.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Calculate the mean within a 5x5 window
            >>> with gw.open('image.tif') as src:
            >>>     res = gw.moving(ds, stat='mean', w=5, nodata=32767.0)
            >>>
            >>> # Calculate the 90th percentile within a 15x15 window
            >>> with gw.open('image.tif') as src:
            >>>     res = gw.moving(stat='perc', w=15, perc=90, nodata=32767.0)
            >>>     res.data.compute(num_workers=4)
        """
        if w % 2 == 0:
            logger.exception('  The window size must be an odd number.')

        if not isinstance(data, xr.DataArray):
            logger.exception('  The input data must be an Xarray DataArray.')

        y = data.y.values
        x = data.x.values
        attrs = data.attrs
        hw = int(w * 0.5)

        @threadpool_limits.wrap(limits=1, user_api='blas')
        def _move_func(block_data: np.ndarray) -> np.ndarray:
            """
            Args:
                block_data (2d array)
            """
            if max(block_data.shape) <= hw:
                return block_data
            else:
                return moving_window(
                    block_data,
                    stat=stat,
                    w=w,
                    perc=perc,
                    nodata=nodata,
                    weights=weights,
                    n_jobs=1,
                )

        results = []
        for band in data.band.values.tolist():
            band_array = data.sel(band=band).astype('float64')
            res = band_array.data.map_overlap(
                _move_func,
                depth=(hw, hw),
                trim=True,
                boundary='reflect',
                dtype='float64',
            )

            results.append(res)

        results = xr.DataArray(
            data=da.stack(results, axis=0),
            dims=('band', 'y', 'x'),
            coords={'band': data.band, 'y': y, 'x': x},
            attrs=attrs,
        )

        new_attrs = {
            'moving_stat': stat,
            'moving_window_size': w,
        }
        if stat == 'perc':
            new_attrs['moving_perc'] = perc

        return results.assign_attrs(**new_attrs)


def sample_feature(
    df_row,
    id_column,
    df_columns,
    crs,
    res,
    all_touched,
    meta,
    frac,
    min_frac_area,
    feature_array=None,
):
    """Samples polygon features.

    Args:
        df_row (pandas.Series)
        id_column (str)
        df_columns (list)
        crs (object)
        res (tuple)
        all_touched (bool)
        meta (namedtuple)
        frac (float)
        min_frac_area (int | float)
        feature_array (Optional[ndarray])

    Returns:
        ``geopandas.GeoDataFrame``
    """
    geom = df_row.geometry

    # Get the feature's bounding extent
    geom_info = get_geometry_info(geom, res)

    if min(geom_info.shape) == 0:
        return gpd.GeoDataFrame([])

    fid = df_row[id_column]
    other_cols = [
        col for col in df_columns if col not in [id_column, 'geometry']
    ]

    if not isinstance(feature_array, np.ndarray):
        # "Rasterize" the geometry into a NumPy array
        feature_array = features.rasterize(
            [geom],
            out_shape=geom_info.shape,
            fill=0,
            out=None,
            transform=geom_info.affine,
            all_touched=all_touched,
            default_value=1,
            dtype='int64',
        )

    # Get the indices of the feature's envelope
    valid_samples = np.where(feature_array == 1)
    # Get geometry indices
    x_samples = valid_samples[1]
    y_samples = valid_samples[0]
    # Convert the geometry indices to geometry map coordinates
    x_coords, y_coords = geom_info.affine * (x_samples, y_samples)
    # Move coordinates offset from polygon left and top edges
    x_coords += abs(res[1]) * 0.5
    y_coords -= abs(res[0]) * 0.5

    if frac < 1:
        take_subset = True
        if isinstance(min_frac_area, (float, int)):
            if y_coords.shape[0] <= min_frac_area:
                take_subset = False

        if take_subset:
            rand_idx = np.random.choice(
                np.arange(0, y_coords.shape[0]),
                size=int(y_coords.shape[0] * frac),
                replace=False,
            )
            y_coords = y_coords[rand_idx]
            x_coords = x_coords[rand_idx]

    n_samples = y_coords.shape[0]
    try:
        fid_ = int(fid)
        fid_ = np.zeros(n_samples, dtype='int64') + fid_
    except ValueError:
        fid_ = str(fid)
        fid_ = np.zeros([fid_] * n_samples, dtype=object)

    # Combine the coordinates into `Shapely` point geometry
    fea_df = gpd.GeoDataFrame(
        data=np.c_[fid_, np.arange(0, n_samples)],
        geometry=gpd.points_from_xy(x_coords, y_coords),
        crs=crs,
        columns=[id_column, 'point'],
    )

    if not fea_df.empty:
        for col in other_cols:
            fea_df = fea_df.assign(**{col: df_row[col]})

    return fea_df
