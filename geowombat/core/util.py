import os
import fnmatch
from collections import namedtuple, OrderedDict
from datetime import datetime

from ..errors import logger
from ..moving import moving_window

import numpy as np
import geopandas as gpd
import xarray as xr
import dask.array as da

from rasterio import features
from rasterio.crs import CRS
from rasterio.warp import reproject, transform_bounds
from rasterio.transform import from_bounds

import shapely
from affine import Affine
from dateparser.search import search_dates

shapely.speedups.enable()


def lazy_wombat(func):
    func.wombat_func_ = True
    return func


def parse_filename_dates(filenames):

    """
    Parses dates from file names

    Args:
        filenames (list): A list of files to parse.

    Returns:
        ``list``
    """

    date_filenames = list()

    for fn in filenames:

        d_name, f_name = os.path.split(fn)
        f_base, f_ext = os.path.splitext(f_name)

        try:

            s, dt = list(zip(*search_dates(' '.join(' '.join(f_base.split('_')).split('-')),
                                           settings={'DATE_ORDER': 'YMD',
                                                     'STRICT_PARSING': False,
                                                     'PREFER_LANGUAGE_DATE_ORDER': False})))

        except:
            return list(range(1, len(filenames) + 1))

        if not dt:
            return list(range(1, len(filenames) + 1))

        date_filenames.append(dt[0])

    return date_filenames


def parse_wildcard(string):

    """
    Parses a search wildcard from a string

    Args:
        string (str): The string to parse.

    Returns:
        ``list``
    """

    if os.path.dirname(string):
        d_name, wildcard = os.path.split(string)
    else:

        d_name = '.'
        wildcard = string

    matches = sorted(fnmatch.filter(os.listdir(d_name), wildcard))

    if matches:
        matches = [os.path.join(d_name, fn) for fn in matches]

    if not matches:
        logger.exception('  There were no images found with the string search.')

    return matches


def sort_images_by_date(image_path,
                        image_wildcard,
                        date_pos,
                        date_start,
                        date_end,
                        split_by='_',
                        date_format='%Y%m%d'):

    """
    Sorts images by date

    Args:
        image_path (Path): The image directory.
        image_wildcard (str): The image search wildcard.
        date_pos (int): The date starting position in the file name.
        date_start (int): The date starting position in the split.
        date_end (int): The date ending position in the split.
        split_by (Optional[str]): How to split the file name.
        date_format (Optional[str]): The date format for ``datetime.datetime.strptime``.

    Returns:
        ``collections.OrderedDict``

    Example:
        >>> from geowombat.core import sort_images
        >>>
        >>> # image example: LC08_L1TP_176038_20190108_20190130_01_T1.tif
        >>> image_path = '/path/to/images'
        >>>
        >>> image_dict = sort_images(image_path, '*.tif', 3, 0, 8)
        >>> image_names = list(image_dict.keys())
        >>> time_names = list(image_dict.values())
    """

    fl = fnmatch.filter(os.listdir(image_path.as_posix()), image_wildcard)
    fl = [image_path.joinpath(fn) for fn in fl]
    dates = [datetime.strptime(fn.name.split(split_by)[date_pos][date_start:date_end], date_format) for fn in fl]

    return OrderedDict(sorted(dict(zip([fn.as_posix() for fn in fl], dates)).items(), key=lambda x_: x_[1]))


def project_coords(x, y, src_crs, dst_crs, return_as='1d', **kwargs):

    """
    Projects coordinates to a new CRS

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

        df_tmp = gpd.GeoDataFrame(np.arange(0, x.shape[0]),
                                  geometry=gpd.points_from_xy(x, y),
                                  crs=src_crs)

        df_tmp = df_tmp.to_crs(dst_crs)

        return df_tmp.geometry.x.values, df_tmp.geometry.y.values

    else:

        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype='float64')

        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype='float64')

        yy = np.meshgrid(x, y)[1]

        latitudes = np.zeros(yy.shape, dtype='float64')

        src_transform = from_bounds(x[0], y[-1], x[-1], y[0], latitudes.shape[1], latitudes.shape[0])

        west, south, east, north = transform_bounds(src_crs, CRS(dst_crs), x[0], y[-1], x[-1], y[0])
        dst_transform = from_bounds(west, south, east, north, latitudes.shape[1], latitudes.shape[0])

        latitudes = reproject(yy,
                              destination=latitudes,
                              src_transform=src_transform,
                              dst_transform=dst_transform,
                              src_crs=CRS.from_epsg(src_crs.split(':')[1]),
                              dst_crs=CRS(dst_crs),
                              **kwargs)[0]

        return xr.DataArray(data=da.from_array(latitudes[np.newaxis, :, :],
                                               chunks=(1, 512, 512)),
                            dims=('band', 'y', 'x'),
                            coords={'band': ['lat'],
                                    'y': ('y', latitudes[:, 0]),
                                    'x': ('x', np.arange(1, latitudes.shape[1]+1))})


def get_geometry_info(geometry, res):

    """
    Gets information from a Shapely geometry object

    Args:
        geometry (object): A `shapely.geometry` object.
        res (tuple): The cell resolution for the affine transform.

    Returns:
        Geometry information (namedtuple)
    """

    GeomInfo = namedtuple('GeomInfo', 'left bottom right top shape affine')

    minx, miny, maxx, maxy = geometry.bounds

    if isinstance(minx, str):
        minx, miny, maxx, maxy = geometry.bounds.values[0]

    out_shape = (int((maxy - miny) / res[1]), int((maxx - minx) / res[0]))

    return GeomInfo(left=minx,
                    bottom=miny,
                    right=maxx,
                    top=maxy,
                    shape=out_shape,
                    affine=Affine(res[0], 0.0, minx, 0.0, -res[1], maxy))


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

    def check_chunktype(self, chunksize, output='3d'):

        if isinstance(chunksize, int):
            chunksize = (1, chunksize, chunksize)

        chunk_len = len(chunksize)
        output_len = int(output[0])

        if not isinstance(chunksize, tuple):
            if not isinstance(chunksize, dict):
                if not isinstance(chunksize, int):
                    logger.warning('  The chunksize parameter should be a tuple, dictionary, or integer.')

        # TODO: make compatible with multi-layer predictions (e.g., probabilities)
        if chunk_len != output_len:
            self.check_chunksize(chunksize, output=output)

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
    def moving(data,
               band_names=None,
               stat='mean',
               perc=50,
               w=3,
               nodata=None,
               n_jobs=1):

        """
        Applies a moving window function over Dask array blocks

        Args:
            data (DataArray): The ``xarray.DataArray`` to process.
            band_names (int or str or list): The output band name(s).
            stat (Optional[str]): The statistic to compute. Choices are ['mean', 'std', 'var', 'min', 'max', 'perc'].
            perc (Optional[int]): The percentile to return if ``stat`` = 'perc'.
            nodata (Optional[int or float]): A 'no data' value to ignore.
            w (Optional[int]): The moving window size (in pixels).
            n_jobs (Optional[int]): The number of rows to process in parallel.

        Returns:
            ``xarray.DataArray``

        Examples:
            >>> import geowombat as gw
            >>>
            >>> # Calculate the mean within a 5x5 window
            >>> with gw.open('image.tif') as src:
            >>>     res = gw.moving(ds, stat='mean', w=5, nodata=32767.0, n_jobs=8)
            >>>
            >>> # Calculate the 90th percentile within a 15x15 window
            >>> with gw.open('image.tif') as src:
            >>>     res = gw.moving(stat='perc', w=15, perc=90, nodata=32767.0, n_jobs=8)
        """

        if w % 2 == 0:
            logger.exception('  The window size must be an odd number.')

        if not isinstance(data, xr.DataArray):
            logger.exception('  The input data must be an Xarray DataArray.')

        y = data.y.values
        x = data.x.values
        attrs = data.attrs

        if n_jobs <= 0:

            logger.warning('  The number of parallel jobs should be a positive integer, so setting n_jobs=1.')
            n_jobs = 1

        hw = int(w / 2.0)

        def _move_func(block_data):

            """
            Args:
                block_data (2d array)
            """

            if max(block_data.shape) <= hw:
                return block_data
            else:

                return moving_window(block_data,
                                     stat=stat,
                                     w=w,
                                     perc=perc,
                                     nodata=nodata,
                                     n_jobs=n_jobs)

        results = list()

        for band in data.band.values.tolist():

            band_array = data.sel(band=band)

            res = band_array.astype('float64').data.map_overlap(_move_func,
                                                                depth=(hw, hw),
                                                                trim=True,
                                                                boundary='reflect',
                                                                dtype='float64')

            results.append(res)

        if isinstance(band_names, np.ndarray):

            if isinstance(band_names.tolist(), str):
                band_names = [band_names.tolist()]

        if not isinstance(band_names, list):

            if not isinstance(band_names, np.ndarray):
                band_names = np.arange(1, data.shape[0]+1)

        results = xr.DataArray(data=da.stack(results, axis=0),
                               dims=('band', 'y', 'x'),
                               coords={'band': band_names,
                                       'y': y,
                                       'x': x},
                               attrs=attrs)

        results.attrs['moving_stat'] = stat
        results.attrs['moving_window_size'] = w

        if stat == 'perc':
            results.attrs['moving_perc'] = perc

        return results


def sample_feature(fid, geom, crs, res, all_touched, meta, frac, feature_array=None):

    """
    Samples a polygon features

    Args:
        fid
        geom
        crs
        res
        all_touched
        meta
        frac
        feature_array

    Returns:
        ``geopandas.GeoDataFrame``
    """

    # Get the feature's bounding extent
    geom_info = get_geometry_info(geom, res)

    if min(geom_info.shape) == 0:
        return gpd.GeoDataFrame([])

    if not isinstance(feature_array, np.ndarray):
    
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
    y_samples = valid_samples[0] + int(round(abs(meta.top - geom_info.top)) / res[1])
    x_samples = valid_samples[1] + int(round(abs(geom_info.left - meta.left)) / res[0])

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

    try:

        fid_ = int(fid)
        fid_ = np.zeros(n_samples, dtype='int64') + fid_

    except:

        fid_ = str(fid)
        fid_ = np.zeros([fid_]*n_samples, dtype=object)

    # Combine the coordinates into `Shapely` point geometry
    return gpd.GeoDataFrame(data=np.c_[fid_, np.arange(0, n_samples)],
                            geometry=gpd.points_from_xy(x_coords, y_coords),
                            crs=crs,
                            columns=['poly', 'point'])
