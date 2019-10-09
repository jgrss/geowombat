# import time
import os
import fnmatch
import ctypes
from datetime import datetime
import concurrent.futures

from ..errors import logger
from .windows import get_window_offsets

import numpy as np
import dask
from dask.diagnostics import ProgressBar
import rasterio as rio
from tqdm import tqdm

try:
    MKL_LIB = ctypes.CDLL('libmkl_rt.so')
except:
    MKL_LIB = None

# SCHEDULERS = dict(threads=ThreadPoolExecutor,
#                   processes=ProcessPoolExecutor)


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


def _window_worker(w, n_bands, indexes_multi):

    """
    Helper to return window slice
    """

    window_slice = (slice(w.row_off, w.row_off + w.height), slice(w.col_off, w.col_off + w.width))

    # Prepend the band position index to the window slice
    if n_bands == 1:

        window_slice = tuple([slice(0, 1)] + list(window_slice))
        indexes = 1

    else:

        window_slice = tuple([slice(0, n_bands)] + list(window_slice))
        indexes = indexes_multi

    return window_slice, indexes


def _window_worker_time(w, n_bands, indexes_multi):

    """
    Helper to return window slice
    """

    window_slice = (slice(w.row_off, w.row_off + w.height), slice(w.col_off, w.col_off + w.width))

    # Prepend the band position index to the window slice
    if n_bands == 1:

        window_slice = tuple([slice(tidx, tidx + 1)] + [slice(0, 1)] + list(window_slice))
        indexes = 1

    else:

        window_slice = tuple([slice(tidx, tidx + 1)] + [slice(0, n_bands)] + list(window_slice))
        indexes = indexes_multi

    return window_slice, indexes


# def _old():
#
# Concurrent.futures environment context
# with SCHEDULERS[scheduler](max_workers=n_jobs) as executor:
#
#     def write(wr, window_slicer):
#
#         # Prepend the band position index to the window slice
#         if n_bands == 1:
#
#             window_slice_ = tuple([slice(0, 1)] + list(window_slicer))
#             indexes = 1
#
#         else:
#
#             window_slice_ = tuple([slice(0, n_bands)] + list(window_slicer))
#             indexes = indexes_multi
#
#         # Write the chunk to file
#         if isinstance(nodata, int) or isinstance(nodata, float):
#
#             dst.write(ds_data[window_slice_].squeeze().fillna(nodata).data.compute(num_workers=n_jobs),
#                       window=wr,
#                       indexes=indexes)
#
#         else:
#
#             dst.write(ds_data[window_slice_].squeeze().data.compute(num_workers=n_jobs),
#                       window=wr,
#                       indexes=indexes)
#
#     joblib.Parallel(n_jobs=n_jobs,
#                     max_nbytes=None)(joblib.delayed(write)(w, window_slice)
#                                      for w, window_slice in map(_window_worker, windows))
#
#     # Multiprocessing pool context
#     # This context is I/O bound, so use the default 'loky' scheduler
#     with multi.Pool(processes=n_jobs) as pool:
#
#         for w, window_slice in tqdm(pool.imap(_window_worker,
#                                               windows,
#                                               chunksize=pool_chunksize),
#                                     total=len(windows)):
#
#             write(wr, window_slice)


# def write_func(block, block_id=None):
#
#     # Current block upper left indices
#     if len(block_id) == 2:
#         i, j = block_id
#     else:
#         i, j = block_id[1:]
#
#     # Current block window
#     w = windows['{:d}{:d}'.format(i, j)]
#
#     if n_bands == 1:
#
#         dst.write(np.squeeze(block),
#                   window=w,
#                   indexes=1)
#
#     else:
#
#         dst.write(block,
#                   window=w,
#                   indexes=indexes)
#
#     return outd
#
# ds_data.data.map_blocks(write_func,
#                         dtype=ds_data.dtype,
#                         chunks=(1, 1, 1)).compute(num_workers=n_jobs)


def to_raster(ds_data,
              filename,
              crs,
              transform,
              driver='GTiff',
              n_jobs=1,
              gdal_cache=512,
              dtype='float64',
              time_chunks=None,
              band_chunks=None,
              row_chunks=None,
              col_chunks=None,
              verbose=0,
              overwrite=False,
              nodata=0,
              tags=None,
              **kwargs):

    """
    Writes an Xarray DataArray or Dataset to a raster file

    Args:
        ds_data (DataArray): The ``xarray.DataArray`` or ``xarray.Dataset`` to write.
        filename (str): The output file name to write to.
        crs (object): A ``rasterio.crs.CRS`` object.
        transform (object): An ``affine.Affine`` transform.
        n_jobs (Optional[str]): The number of parallel chunks to write.
        verbose (Optional[int]): The verbosity level.
        overwrite (Optional[bool]): Whether to overwrite an existing file.
        driver (Optional[str]): The raster driver.
        gdal_cache (Optional[int]): The ``GDAL`` cache size (in MB).
        dtype (Optional[int]): The output data type.
        time_chunks (Optional[int]): The processing time chunk size.
        band_chunks (Optional[int]): The processing band chunk size.
        row_chunks (Optional[int]): The processing row chunk size. In general, this should be left as None and
            the chunk size will be taken from the ``dask`` array.
        col_chunks (Optional[int]): The processing column chunk size. Same as ``row_chunks``.
        nodata (Optional[int]): A 'no data' value.
        tags (Optional[dict]): Image tags to write to file.
        kwargs (Optional[dict]): Additional keyword arguments to pass to ``rasterio.write``.

    Returns:
        None
    """

    # if scheduler not in ['threads', 'processes']:
    #     logger.exception("  The scheduler must be 'threads' or 'processes'.")

    if MKL_LIB:
        __ = MKL_LIB.MKL_Set_Num_Threads(n_jobs)

    d_name = os.path.dirname(filename)

    if d_name:

        if not os.path.isdir(d_name):
            os.makedirs(d_name)

    n_time = ds_data.gw.ntime
    n_bands = ds_data.gw.nbands
    n_rows = ds_data.gw.nrows
    n_cols = ds_data.gw.ncols

    if not isinstance(time_chunks, int):
        time_chunks = ds_data.gw.time_chunks

    if not isinstance(band_chunks, int):
        band_chunks = ds_data.gw.band_chunks

    if not isinstance(row_chunks, int):
        row_chunks = ds_data.gw.row_chunks

    if not isinstance(col_chunks, int):
        col_chunks = ds_data.gw.col_chunks

    if 'blockysize' not in kwargs:
        kwargs['blockysize'] = row_chunks

    if 'blockxsize' not in kwargs:
        kwargs['blockxsize'] = col_chunks

    if isinstance(dtype, str):

        if ds_data.dtype != dtype:
            ds_data = ds_data.astype(dtype)

    else:
        dtype = ds_data.dtype

    if verbose > 0:
        logger.info('  Creating and writing to the file ...')

    # Setup the windows
    windows = get_window_offsets(n_rows, n_cols, row_chunks, col_chunks)

    if n_bands > 1:
        indexes_multi = list(range(1, n_bands + 1))
    else:
        indexes_multi = None

    # outd = np.array([0], dtype='uint8')[None, None]

    # Rasterio environment context
    with rio.Env(GDAL_CACHEMAX=gdal_cache):

        if n_time > 1:

            d_name, f_name = os.path.split(filename)
            f_base, f_ext = os.path.splitext(filename)

            # Write each temporal layer separately
            for tidx in range(0, n_time):

                time_value = ds_data.coords['time'].values[tidx]

                if isinstance(time_value, np.datetime64):

                    time_value_dt = datetime.utcfromtimestamp(int(time_value) * 1e-9)
                    time_value = time_value_dt.strftime('%Y-%m-%d')

                else:
                    time_value = str(time_value)

                filename_time = os.path.join(d_name, '{BASE}_{INTV}{EXT}'.format(BASE=f_base,
                                                                                 INTV=time_value,
                                                                                 EXT=f_ext))

                if overwrite:

                    if os.path.isfile(filename_time):
                        os.remove(filename_time)

                # Open the output file for writing
                with rio.open(filename_time,
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

                        @dask.delayed
                        def write_func(output, out_window, out_indexes, last_write=None):

                            """
                            Writes a NumPy array to file

                            Reference:
                                https://github.com/dask/dask/issues/3600
                            """

                            del last_write
                            dst.write(output, window=out_window, indexes=out_indexes)

                        # Create the Dask.delayed writers
                        writer = None
                        for w in windows:

                            window_slice, indexes = _window_worker_time(w, n_bands, indexes_multi)

                            if isinstance(nodata, int) or isinstance(nodata, float):
                                writer = write_func(ds_data[window_slice].squeeze().fillna(nodata).data, w, indexes,writer)
                            else:
                                writer = write_func(ds_data[window_slice].squeeze().data, w, indexes, writer)

                        # Write the data to file
                        with ProgressBar():
                            writer.compute(num_workers=n_jobs)

        else:

            if overwrite:

                if os.path.isfile(filename):
                    os.remove(filename)

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

                    @dask.delayed
                    def write_func(output, out_window, out_indexes, last_write=None):

                        """
                        Writes a NumPy array to file

                        Reference:
                            https://github.com/dask/dask/issues/3600
                        """

                        del last_write
                        dst.write(output, window=out_window, indexes=out_indexes)

                    # Create the Dask.delayed writers
                    writer = None
                    for w in windows:

                        window_slice, indexes = _window_worker(w, n_bands, None)

                        if isinstance(nodata, int) or isinstance(nodata, float):
                            writer = write_func(ds_data[window_slice].squeeze().fillna(nodata).data, w, indexes, writer)
                        else:
                            writer = write_func(ds_data[window_slice].squeeze().data, w, indexes, writer)

                    # Write the data to file
                    with ProgressBar():
                        writer.compute(num_workers=n_jobs)

    if verbose > 0:
        logger.info('  Finished writing')


def apply(infile,
          outfile,
          block_func,
          gdal_cache=512,
          count=1,
          dtype='float64',
          nodata=0,
          compress='lzw',
          tiled=True,
          blockxsize=512,
          blockysize=512,
          n_jobs=4,
          overwrite=False):

    """
    Applies a function and writes results to file

    Args:
        infile (str)
        outfile (str)
        block_func (func)
        gdal_cache (Optional[int])
        count (Optional[int])
        dtype (Optional[str])
        nodata (Optional[int or float])
        compress (Optional[str])
        tiled (Optional[bool])
        blockxsize (Optional[int])
        blockysize (Optional[int])
        n_jobs (Optional[int])
        overwrite (Optional[bool])

    Examples:
        >>> import geowombat as gw
        >>>
        >>> gw.apply('input.tif', 'output.tif', my_func, n_jobs=8)

    Returns:
        None
    """

    if overwrite:

        if os.path.isfile(outfile):
            os.remove(outfile)

    with rasterio.Env(gdal_cache=gdal_cache):

        with rasterio.open(infile) as src:

            # Create a destination dataset based on source params. The
            # destination will be tiled, and we'll process the tiles
            # concurrently.
            profile = src.profile

            profile.update(count=count,
                           blockxsize=blockxsize,
                           blockysize=blockysize,
                           dtype=dtype,
                           nodata=nodata,
                           compress=compress,
                           tiled=tiled)

            with rasterio.open(outfile, 'w', **profile) as dst:

                # Materialize a list of destination block windows
                # that we will use in several statements below.
                windows = get_window_offsets(src.height, src.width, blockysize, blockxsize, return_as='list')

                # This generator comprehension gives us raster data
                # arrays for each window. Later we will zip a mapping
                # of it with the windows list to get (window, result)
                # pairs.
                data_gen = (src.read(window=window, out_dtype='float64') for window in windows)

                # scales_ = (scales for window in windows)

                with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:

                    # We map the compute() function over the raster
                    # data generator, zip the resulting iterator with
                    # the windows list, and as pairs come back we
                    # write data to the destination dataset.
                    for window, result in tqdm(zip(windows, executor.map(block_func, data_gen)), total=len(windows)):
                        dst.write(result, window=window, indexes=count)
