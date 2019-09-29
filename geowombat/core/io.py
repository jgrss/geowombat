# import time
import os
import ctypes
import multiprocessing as multi
# import concurrent.futures

from ..errors import logger
from .windows import get_window_offsets

import rasterio as rio
from tqdm import tqdm


try:
    MKL_LIB = ctypes.CDLL('libmkl_rt.so')
except:
    MKL_LIB = None


def _window_worker(w):
    """Helper to return window slice"""
    # time.sleep(0.001)
    return w, (slice(w.row_off, w.row_off+w.height), slice(w.col_off, w.col_off+w.width))


def xarray_to_raster(ds_data,
                     filename,
                     crs,
                     transform,
                     driver,
                     n_jobs,
                     gdal_cache,
                     dtype,
                     row_chunks,
                     col_chunks,
                     pool_chunksize,
                     verbose,
                     overwrite,
                     nodata,
                     tags,
                     **kwargs):

    """
    Writes an Xarray DataArray to a raster file

    Args:
        ds_data (DataArray)
        filename (str)
        crs (object)
        transform (object)
        driver (str)
        n_jobs (int)
        gdal_cache (int)
        dtype (float)
        row_chunks (int)
        col_chunks (int)
        pool_chunksize (int)
        verbose (int)
        overwrite (bool)
        nodata (int or float)
        tags (dict)
        kwargs (dict)

    Returns:
        None
    """

    if MKL_LIB:
        __ = MKL_LIB.MKL_Set_Num_Threads(n_jobs)

    if overwrite:

        if os.path.isfile(filename):
            os.remove(filename)

    d_name = os.path.dirname(filename)

    if d_name:

        if not os.path.isdir(d_name):
            os.makedirs(d_name)

    data_shape = ds_data.shape

    if len(data_shape) == 2:

        n_bands = 1
        n_rows = data_shape[0]
        n_cols = data_shape[1]

        if not isinstance(row_chunks, int):
            row_chunks = ds_data.data.chunksize[0]

        if not isinstance(col_chunks, int):
            col_chunks = ds_data.data.chunksize[1]

    else:

        n_bands = data_shape[0]
        n_rows = data_shape[1]
        n_cols = data_shape[2]

        if not isinstance(row_chunks, int):
            row_chunks = ds_data.data.chunksize[1]

        if not isinstance(col_chunks, int):
            col_chunks = ds_data.data.chunksize[2]

    if isinstance(dtype, str):

        if ds_data.dtype != dtype:
            ds_data = ds_data.astype(dtype)

    else:
        dtype = ds_data.dtype

    # Setup the windows
    windows = get_window_offsets(n_rows, n_cols, row_chunks, col_chunks)
    # windows = get_window_offsets(n_rows, n_cols, row_chunks, col_chunks, return_as='dict')

    if n_bands > 1:
        indexes = list(range(1, n_bands + 1))

    # outd = np.array([0], dtype='uint8')[None, None]

    if verbose > 0:
        logger.info('  Creating and writing to the file ...')

    # Rasterio environment context
    with rio.Env(GDAL_CACHEMAX=gdal_cache):

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
                            indexes = list(range(1, n_bands+1))

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
