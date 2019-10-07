# import time
import os
import fnmatch
import ctypes
from datetime import datetime
import multiprocessing as multi
# import concurrent.futures

from ..errors import logger
from .windows import get_window_offsets

import numpy as np
import rasterio as rio
from tqdm import tqdm


try:
    MKL_LIB = ctypes.CDLL('libmkl_rt.so')
except:
    MKL_LIB = None


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

    matches = fnmatch.filter(d_name, wildcard)

    if matches:
        matches = [os.path.join(d_name, fn) for fn in matches]

    return matches


def _window_worker(w):
    """Helper to return window slice"""
    # time.sleep(0.001)
    return w, (slice(w.row_off, w.row_off+w.height), slice(w.col_off, w.col_off+w.width))


def to_raster(ds_data,
              filename,
              crs,
              transform,
              driver='GTiff',
              n_jobs=1,
              gdal_cache=512,
              dtype='float64',
              time_chunks=1,
              band_chunks=1,
              row_chunks=512,
              col_chunks=512,
              pool_chunksize=1000,
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
        row_chunks (Optional[int]): The processing row chunk size.
        col_chunks (Optional[int]): The processing column chunk size.
        pool_chunksize (Optional[int]): The `multiprocessing.Pool` chunk size.
        nodata (Optional[int]): A 'no data' value.
        tags (Optional[dict]): Image tags to write to file.
        kwargs (Optional[dict]): Additional keyword arguments to pass to ``rasterio.write``.

    Returns:
        None
    """

    if MKL_LIB:
        __ = MKL_LIB.MKL_Set_Num_Threads(n_jobs)

    d_name = os.path.dirname(filename)

    if d_name:

        if not os.path.isdir(d_name):
            os.makedirs(d_name)

    n_time = ds_data.gw.tdims
    n_bands = ds_data.gw.bands
    n_rows = ds_data.gw.rows
    n_cols = ds_data.gw.cols

    if not isinstance(time_chunks, int):
        time_chunks = ds_data.gw.time_chunks

    if not isinstance(band_chunks, int):
        band_chunks = ds_data.gw.band_chunks

    if not isinstance(row_chunks, int):
        row_chunks = ds_data.gw.row_chunks

    if not isinstance(col_chunks, int):
        col_chunks = ds_data.gw.col_chunks

    if isinstance(dtype, str):

        if ds_data.dtype != dtype:
            ds_data = ds_data.astype(dtype)

    else:
        dtype = ds_data.dtype

    # Setup the windows
    windows = get_window_offsets(n_rows, n_cols, row_chunks, col_chunks)
    # windows = get_window_offsets(n_rows, n_cols, row_chunks, col_chunks, return_as='dict')

    if n_bands > 1:
        indexes_multi = list(range(1, n_bands + 1))

    # outd = np.array([0], dtype='uint8')[None, None]

    if verbose > 0:
        logger.info('  Creating and writing to the file ...')

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

                        # Multiprocessing pool context
                        # This context is I/O bound, so use the default 'loky' scheduler
                        with multi.Pool(processes=n_jobs) as pool:

                            # Iterate over each window
                            for w, window_slice in tqdm(pool.imap_unordered(_window_worker,
                                                                            windows,
                                                                            chunksize=pool_chunksize),
                                                        total=len(windows)):

                                # Prepend the band position index to the window slice
                                if n_bands == 1:

                                    window_slice_ = tuple([slice(tidx, tidx+1)] + [slice(0, 1)] + list(window_slice))
                                    indexes = 1

                                else:

                                    window_slice_ = tuple([slice(tidx, tidx+1)] + [slice(0, n_bands)] + list(window_slice))
                                    indexes = indexes_multi

                                # Write the chunk to file
                                if isinstance(nodata, int) or isinstance(nodata, float):

                                    dst.write(ds_data[window_slice_].squeeze().fillna(nodata).load().data,
                                              window=w,
                                              indexes=indexes)

                                else:

                                    dst.write(ds_data[window_slice_].squeeze().load().data,
                                              window=w,
                                              indexes=indexes)

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

                    # Multiprocessing pool context
                    # This context is I/O bound, so use the default 'loky' scheduler
                    with multi.Pool(processes=n_jobs) as pool:

                        # Iterate over each window
                        for w, window_slice in tqdm(pool.imap_unordered(_window_worker,
                                                                        windows,
                                                                        chunksize=pool_chunksize),
                                                    total=len(windows)):

                    # with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:

                        # for w, window_slice in tqdm(executor.map(_window_worker, windows), total=len(windows)):

                            # Prepend the band position index to the window slice
                            if n_bands == 1:

                                window_slice_ = tuple([slice(0, 1)] + list(window_slice))
                                indexes = 1

                            else:

                                window_slice_ = tuple([slice(0, n_bands)] + list(window_slice))
                                indexes = indexes_multi

                            # Write the chunk to file
                            if isinstance(nodata, int) or isinstance(nodata, float):

                                dst.write(ds_data[window_slice_].squeeze().fillna(nodata).load().data,
                                          window=w,
                                          indexes=indexes)

                            else:

                                dst.write(ds_data[window_slice_].squeeze().load().data,
                                          window=w,
                                          indexes=indexes)

    if verbose > 0:
        logger.info('  Finished writing')
