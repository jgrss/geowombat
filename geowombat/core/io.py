import os
from pathlib import Path
import shutil
import itertools
import fnmatch
import ctypes
import concurrent.futures
from contextlib import contextmanager
import multiprocessing as multi
import threading
import random
import string

from ..errors import logger
from ..backends.rasterio_ import WriteDaskArray
from ..backends.zarr_ import to_zarr
from .windows import get_window_offsets

import numpy as np
import geopandas as gpd
import xarray as xr

import dask.array as da
from dask import is_dask_collection
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

import rasterio as rio
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.warp import aligned_target

import shapely
from shapely.geometry import Polygon
from affine import Affine

import zarr
from tqdm import tqdm

try:
    MKL_LIB = ctypes.CDLL('libmkl_rt.so')
except:
    MKL_LIB = None


shapely.speedups.enable()


def get_norm_indices(n_bands, window_slice, indexes_multi):

    # Prepend the band position index to the window slice
    if n_bands == 1:

        window_slice = tuple([slice(0, 1)] + list(window_slice))
        indexes = 1

    else:

        window_slice = tuple([slice(0, n_bands)] + list(window_slice))
        indexes = indexes_multi

    return window_slice, indexes


def _window_worker(w):

    """
    Helper to return window slice
    """

    return slice(w.row_off, w.row_off + w.height), slice(w.col_off, w.col_off + w.width)


def _window_worker_time(w, n_bands, tidx, n_time):

    """
    Helper to return window slice
    """

    window_slice = (slice(w.row_off, w.row_off + w.height), slice(w.col_off, w.col_off + w.width))

    # Prepend the band position index to the window slice
    if n_bands == 1:
        window_slice = tuple([slice(tidx, n_time)] + [slice(0, 1)] + list(window_slice))
    else:
        window_slice = tuple([slice(tidx, n_time)] + [slice(0, n_bands)] + list(window_slice))

    return window_slice


@contextmanager
def _cluster_dummy(**kwargs):
    yield None


@contextmanager
def _client_dummy(**kwargs):
    yield None


def _compressor(*args):

    w_, b_, f_, o_ = list(itertools.chain(*args))

    with rio.open(f_, mode='r+', sharing=False) as dst_:

        dst_.write(np.squeeze(b_),
                   window=w_,
                   indexes=o_)


def _block_write_func(*args):

    ofn_, fn_, g_, t_ = list(itertools.chain(*args))

    if t_ == 'zarr':

        group_node = zarr.open(fn_, mode='r')[g_]

        w_ = Window(row_off=group_node.attrs['row_off'],
                    col_off=group_node.attrs['col_off'],
                    height=group_node.attrs['height'],
                    width=group_node.attrs['width'])

        out_data_ = np.squeeze(group_node['data'][:])

    else:

        w_ = Window(row_off=int(os.path.splitext(os.path.basename(fn_))[0].split('_')[-4][1:]),
                    col_off=int(os.path.splitext(os.path.basename(fn_))[0].split('_')[-3][1:]),
                    height=int(os.path.splitext(os.path.basename(fn_))[0].split('_')[-2][1:]),
                    width=int(os.path.splitext(os.path.basename(fn_))[0].split('_')[-1][1:]))

        with rio.open(fn_) as src_:
            out_data_ = np.squeeze(src_.read(window=w_))

    out_indexes_ = 1 if len(out_data_.shape) == 2 else list(range(1, out_data_.shape[0]+1))

    with rio.open(ofn_, mode='r+', sharing=False) as dst_:

        dst_.write(out_data_,
                   window=w_,
                   indexes=out_indexes_)


def _block_read_func(fn_, g_, t_):

    """
    Function for block writing with ``concurrent.futures``
    """

    # fn_, g_, t_ = list(itertools.chain(*args))

    if t_ == 'zarr':

        group_node = zarr.open(fn_, mode='r')[g_]

        w_ = Window(row_off=group_node.attrs['row_off'],
                    col_off=group_node.attrs['col_off'],
                    height=group_node.attrs['height'],
                    width=group_node.attrs['width'])

        out_data_ = np.squeeze(group_node['data'][:])

    else:

        w_ = Window(row_off=int(os.path.splitext(os.path.basename(fn_))[0].split('_')[-4][1:]),
                    col_off=int(os.path.splitext(os.path.basename(fn_))[0].split('_')[-3][1:]),
                    height=int(os.path.splitext(os.path.basename(fn_))[0].split('_')[-2][1:]),
                    width=int(os.path.splitext(os.path.basename(fn_))[0].split('_')[-1][1:]))

        out_data_ = np.squeeze(rio.open(fn_).read(window=w_))

    out_indexes_ = 1 if len(out_data_.shape) == 2 else list(range(1, out_data_.shape[0]+1))

    return w_, out_indexes_, out_data_


def _return_window(window_, block, num_workers):

    with threading.Lock():
        out_data_ = block.data.compute(scheduler='threads', num_workers=num_workers)

    if 'apply' in block.attrs:

        if ('apply_args' in block.attrs) and ('apply_kwargs' in block.attrs):
            out_data_ = block.attrs['apply'](out_data_, *block.attrs['apply_args'], **block.attrs['apply_kwargs'])
        elif ('apply_args' in block.attrs) and ('apply_kwargs' not in block.attrs):
            out_data_ = block.attrs['apply'](out_data_, *block.attrs['apply_args'])
        elif ('apply_args' not in block.attrs) and ('apply_kwargs' in block.attrs):
            out_data_ = block.attrs['apply'](out_data_, **block.attrs['apply_kwargs'])
        else:
            out_data_ = block.attrs['apply'](out_data_)

    dshape = out_data_.shape

    if len(dshape) > 2:
        out_data_ = np.squeeze(out_data_)

    if len(dshape) == 2:
        indexes_ = 1
    else:
        indexes_ = 1 if dshape[0] == 1 else list(range(1, dshape[0]+1))

    return out_data_, window_, indexes_


def _write_xarray(*args):

    """
    Writes a NumPy array to file

    Reference:
        https://github.com/dask/dask/issues/3600
    """

    zarr_file = None

    block, filename, block_window, n_threads, separate, chunks, root = list(itertools.chain(*args))

    output, out_window, out_indexes = _return_window(block_window, block, n_threads)

    if separate:
        zarr_file = to_zarr(filename, output, out_window, chunks, root=root)
    else:

        with threading.Lock():

            with rio.open(filename,
                          mode='r+',
                          sharing=False) as dst_:

                dst_.write(output,
                           window=out_window,
                           indexes=out_indexes)

    return zarr_file


def to_vrt(data,
           filename,
           resampling=None,
           nodata=None,
           init_dest_nodata=True,
           warp_mem_limit=128):

    """
    Writes a file to a VRT file

    Args:
        data (DataArray): The ``xarray.DataArray`` to write.
        filename (str): The output file name to write to.
        resampling (Optional[object]): The resampling algorithm for ``rasterio.vrt.WarpedVRT``. Default is 'nearest'.
        nodata (Optional[float or int]): The 'no data' value for ``rasterio.vrt.WarpedVRT``.
        init_dest_nodata (Optional[bool]): Whether or not to initialize output to ``nodata`` for ``rasterio.vrt.WarpedVRT``.
        warp_mem_limit (Optional[int]): The GDAL memory limit for ``rasterio.vrt.WarpedVRT``.

    Example:
        >>> import geowombat as gw
        >>> from rasterio.enums import Resampling
        >>>
        >>> with gw.config.update(ref_crs=102033):
        >>>
        >>>     with gw.open('image.tif') as ds:
        >>>
        >>>         gw.to_vrt(ds,
        >>>                   'image.vrt',
        >>>                   resampling=Resampling.cubic,
        >>>                   warp_mem_limit=256)
    """

    if not resampling:
        resampling = Resampling.nearest

    # Open the input file on disk
    with rio.open(data.filename) as src:

        with WarpedVRT(src,
                       src_crs=src.crs,                         # the original CRS
                       crs=data.crs,                            # the transformed CRS
                       src_transform=src.transform,             # the original transform
                       transform=data.transform,                # the new transform
                       dtype=data.gw.dtype,
                       resampling=resampling,
                       nodata=nodata,
                       init_dest_nodata=init_dest_nodata,
                       warp_mem_limit=warp_mem_limit) as vrt:

            rio_shutil.copy(vrt, filename, driver='VRT')


def to_raster(data,
              filename,
              readxsize=None,
              readysize=None,
              use_dask_store=False,
              separate=False,
              out_block_type='zarr',
              keep_blocks=False,
              verbose=0,
              overwrite=False,
              gdal_cache=512,
              scheduler='mpool',
              n_jobs=1,
              n_workers=None,
              n_threads=None,
              n_chunks=None,
              overviews=False,
              resampling='nearest',
              use_client=False,
              address=None,
              total_memory=48,
              **kwargs):

    """
    Writes a ``dask`` array to a raster file

    Args:
        data (DataArray): The ``xarray.DataArray`` to write.
        filename (str): The output file name to write to.
        readxsize (Optional[int]): The size of column chunks to read. If not given, ``readxsize`` defaults to Dask
            chunk size.
        readysize (Optional[int]): The size of row chunks to read. If not given, ``readysize`` defaults to Dask
            chunk size.
        separate (Optional[bool]): Whether to write blocks as separate files. Otherwise, write to a single file.
        use_dask_store (Optional[bool]): Whether to use ``dask.array.store`` to save with Dask task graphs.
        out_block_type (Optional[str]): The output block type. Choices are ['gtiff', 'zarr'].
            Only used if ``separate`` = ``True``.
        keep_blocks (Optional[bool]): Whether to keep the blocks stored on disk. Only used if ``separate`` = ``True``.
        verbose (Optional[int]): The verbosity level.
        overwrite (Optional[bool]): Whether to overwrite an existing file.
        gdal_cache (Optional[int]): The ``GDAL`` cache size (in MB).
        scheduler (Optional[str]): The ``concurrent.futures`` scheduler to use. Choices are ['processes', 'threads', 'mpool'].

            mpool: process pool of workers using ``multiprocessing.Pool``
            processes: process pool of workers using ``concurrent.futures``
            threads: thread pool of workers using ``concurrent.futures``

        n_jobs (Optional[int]): The total number of parallel jobs.
        n_workers (Optional[int]): The number of processes.
        n_threads (Optional[int]): The number of threads.
        n_chunks (Optional[int]): The chunk size of windows. If not given, equal to ``n_workers`` x 50.
        overviews (Optional[bool or list]): Whether to build overview layers.
        resampling (Optional[str]): The resampling method for overviews when ``overviews`` is ``True`` or a ``list``.
            Choices are ['average', 'bilinear', 'cubic', 'cubic_spline', 'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest'].
        use_client (Optional[bool]): Whether to use a ``dask`` client.
        address (Optional[str]): A cluster address to pass to client. Only used when ``use_client`` = ``True``.
        total_memory (Optional[int]): The total memory (in GB) required when ``use_client`` = ``True``.
        kwargs (Optional[dict]): Additional keyword arguments to pass to ``rasterio.write``.

    Returns:
        ``dask.delayed`` object

    Examples:
        >>> import geowombat as gw
        >>>
        >>> # Use 8 parallel workers
        >>> with gw.open('input.tif') as ds:
        >>>     gw.to_raster(ds, 'output.tif', n_jobs=8)
        >>>
        >>> # Use 4 process workers and 2 thread workers
        >>> with gw.open('input.tif') as ds:
        >>>     gw.to_raster(ds, 'output.tif', n_workers=4, n_threads=2)
        >>>
        >>> # Control the window chunks passed to concurrent.futures
        >>> with gw.open('input.tif') as ds:
        >>>     gw.to_raster(ds, 'output.tif', n_workers=4, n_threads=2, n_chunks=16)
        >>>
        >>> # Compress the output and build overviews
        >>> with gw.open('input.tif') as ds:
        >>>     gw.to_raster(ds, 'output.tif', n_jobs=8, overviews=True, compress='lzw')
    """

    if MKL_LIB:
        __ = MKL_LIB.MKL_Set_Num_Threads(n_threads)

    pfile = Path(filename)

    if scheduler.lower() == 'mpool':
        pool_executor = multi.Pool
    else:
        pool_executor = concurrent.futures.ProcessPoolExecutor if scheduler.lower() == 'processes' else concurrent.futures.ThreadPoolExecutor

    if overwrite:

        if pfile.is_file():
            pfile.unlink()

    if pfile.is_file():
        logger.warning('  The output file already exists.')
        return

    if not is_dask_collection(data.data):
        logger.exception('  The data should be a dask array.')

    if use_client:

        if address:
            cluster_object = _cluster_dummy
        else:
            cluster_object = LocalCluster

        client_object = Client

    else:

        cluster_object = _cluster_dummy
        client_object = _client_dummy

    if isinstance(n_workers, int) and isinstance(n_threads, int):
        n_jobs = n_workers * n_threads
    else:

        n_workers = n_jobs
        n_threads = 1

    mem_per_core = int(total_memory / n_workers)

    if not isinstance(n_chunks, int):
        n_chunks = n_workers * 50

    if not isinstance(readxsize, int):
        readxsize = data.gw.col_chunks

    if not isinstance(readysize, int):
        readysize = data.gw.row_chunks

    chunksize = (data.gw.row_chunks, data.gw.col_chunks)

    # Force tiled outputs with no file sharing
    kwargs['sharing'] = False

    if data.gw.tiled:
        kwargs['tiled'] = True

    if 'compress' in kwargs:

        # Store the compression type because
        #   it is removed in concurrent writing
        compress = True
        compress_type = kwargs['compress']
        del kwargs['compress']

    elif isinstance(data.gw.compress, str) and data.gw.compress.lower() in ['lzw', 'deflate']:

        compress = True
        compress_type = data.gw.compress

    else:
        compress = False

    if 'nodata' not in kwargs:

        if isinstance(data.gw.nodata, int) or isinstance(data.gw.nodata, float):
            kwargs['nodata'] = data.gw.nodata

    if 'blockxsize' not in kwargs:
        kwargs['blockxsize'] = data.gw.col_chunks

    if 'blockysize' not in kwargs:
        kwargs['blockysize'] = data.gw.row_chunks

    if 'bigtiff' not in kwargs:
        kwargs['bigtiff'] = data.gw.bigtiff

    if 'driver' not in kwargs:
        kwargs['driver'] = data.gw.driver

    if 'count' not in kwargs:
        kwargs['count'] = data.gw.nbands

    if 'width' not in kwargs:
        kwargs['width'] = data.gw.ncols

    if 'height' not in kwargs:
        kwargs['height'] = data.gw.nrows

    if separate:

        d_name = pfile.parent
        sub_dir = d_name.joinpath('sub_tmp_')
        zarr_file = sub_dir.joinpath('data.zarr').as_posix()

        sub_dir.mkdir(parents=True, exist_ok=True)

        root = zarr.open(zarr_file, mode='w')

    else:

        root = None

        if verbose > 0:
            logger.info('  Creating the file ...\n')

        with rio.open(filename, mode='w', **kwargs) as rio_dst:
            pass

    if verbose > 0:
        logger.info('  Writing data to file ...\n')

    with rio.Env(GDAL_CACHEMAX=gdal_cache):

        if not use_dask_store:

            windows = get_window_offsets(data.gw.nrows,
                                         data.gw.ncols,
                                         readysize,
                                         readxsize,
                                         return_as='list')

            n_windows = len(windows)

            # Iterate over the windows in chunks
            for wchunk in range(0, n_windows, n_chunks):

                window_slice = windows[wchunk:wchunk+n_chunks]
                n_windows_slice = len(window_slice)

                if verbose > 0:

                    logger.info('  Windows {:,d}--{:,d} of {:,d} ...'.format(wchunk+1,
                                                                             wchunk+n_windows_slice,
                                                                             n_windows))

                if len(data.shape) == 2:
                    data_gen = ((data[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width], filename, w, n_threads, separate, chunksize, root) for w in window_slice)
                elif len(data.shape) == 3:
                    data_gen = ((data[:, w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width], filename, w, n_threads, separate, chunksize, root) for w in window_slice)
                else:
                    data_gen = ((data[:, :, w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width], filename, w, n_threads, separate, chunksize, root) for w in window_slice)

                with pool_executor(n_workers) as executor:

                    if scheduler == 'mpool':

                        for zarr_file in tqdm(executor.imap_unordered(_write_xarray,
                                                                      data_gen),
                                              total=n_windows_slice):
                            pass

                    else:

                        for zarr_file in tqdm(executor.map(_write_xarray, data_gen), total=n_windows_slice):
                            pass

            # if overviews:
            #
            #     if not isinstance(overviews, list):
            #         overviews = [2, 4, 8, 16]
            #
            #     if resampling not in ['average', 'bilinear', 'cubic', 'cubic_spline',
            #                           'gauss', 'lanczos', 'max', 'med', 'min', 'mode', 'nearest']:
            #
            #         logger.warning("  The resampling method is not supported by rasterio. Setting to 'nearest'")
            #
            #         resampling = 'nearest'
            #
            #     if verbose > 0:
            #         logger.info('  Building pyramid overviews ...')
            #
            #     rio_dst.build_overviews(overviews, getattr(Resampling, resampling))
            #     rio_dst.update_tags(ns='overviews', resampling=resampling)

        else:

            with cluster_object(n_workers=n_workers,
                                threads_per_worker=n_threads,
                                scheduler_port=0,
                                processes=False,
                                memory_limit='{:d}GB'.format(mem_per_core)) as cluster:

                cluster_address = address if address else cluster

                with client_object(address=cluster_address) as client:

                    with WriteDaskArray(filename,
                                        overwrite=overwrite,
                                        separate=separate,
                                        out_block_type=out_block_type,
                                        keep_blocks=keep_blocks,
                                        gdal_cache=gdal_cache,
                                        **kwargs) as dst:

                        # Store the data and return a lazy evaluator
                        res = da.store(da.squeeze(data.data),
                                       dst,
                                       lock=False,
                                       compute=False)

                        if verbose > 0:
                            logger.info('  Writing data to file ...')

                        # Send the data to file
                        #
                        # *Note that the progress bar will
                        #   not work with a client.
                        if use_client:
                            res.compute(num_workers=n_jobs)
                        else:

                            with ProgressBar():
                                res.compute(num_workers=n_jobs)

                        if verbose > 0:
                            logger.info('  Finished writing data to file.')

                        out_block_type = dst.out_block_type
                        keep_blocks = dst.keep_blocks
                        zarr_file = dst.zarr_file
                        sub_dir = dst.sub_dir

        if compress:

            if verbose > 0:
                logger.info('  Compressing output file ...')

            if separate:

                group_keys = list(root.group_keys())
                n_groups = len(group_keys   )

                if out_block_type.lower() == 'zarr':
                    # root = zarr.open(zarr_file, mode='r')
                    open_file = zarr_file
                else:

                    outfiles = sorted(fnmatch.filter(os.listdir(sub_dir), '*.tif'))
                    outfiles = [os.path.join(sub_dir, fn) for fn in outfiles]

                    # data_gen = ((fn, None, 'gtiff') for fn in outfiles)

                kwargs['compress'] = compress_type

                n_windows = len(group_keys)

                # Compress into one file
                with rio.open(filename, mode='w', **kwargs) as dst_:

                    # Iterate over the windows in chunks
                    for wchunk in range(0, n_groups, n_chunks):

                        group_keys_slice = group_keys[wchunk:wchunk + n_chunks]
                        n_windows_slice = len(group_keys_slice)

                        if verbose > 0:

                            logger.info('  Windows {:,d}--{:,d} of {:,d} ...'.format(wchunk + 1,
                                                                                     wchunk + n_windows_slice,
                                                                                     n_windows))

                        ################################################
                        data_gen = ((open_file, group, 'zarr') for group in group_keys_slice)

                        # for f in tqdm(executor.map(_compressor, data_gen), total=n_windows_slice):
                        #     pass
                        #
                        # futures = [executor.submit(_compress_dummy, iter_[0], iter_[1], None) for iter_ in data_gen]
                        #
                        # for f in tqdm(concurrent.futures.as_completed(futures), total=n_windows_slice):
                        #
                        #     out_window, out_block = f.result()
                        #
                        #     dst_.write(np.squeeze(out_block),
                        #                window=out_window,
                        #                indexes=out_indexes_)
                        ################################################

                        # data_gen = ((root, group, 'zarr') for group in group_keys_slice)

                        # for f, g, t in tqdm(data_gen, total=n_windows_slice):
                        #
                        #     out_window, out_indexes, out_block = _block_read_func(f, g, t)

                        # executor.map(_block_write_func, data_gen)

                        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:

                            # Submit all of the tasks as futures
                            futures = [executor.submit(_block_read_func, f, g, t) for f, g, t in data_gen]

                            for f in tqdm(concurrent.futures.as_completed(futures), total=n_windows_slice):

                                out_window, out_indexes, out_block = f.result()

                                dst_.write(out_block,
                                           window=out_window,
                                           indexes=out_indexes)

                        futures = None

                if not keep_blocks:
                    shutil.rmtree(sub_dir)

            else:

                p = Path(filename)

                d_name = p.parent
                f_base, f_ext = os.path.splitext(p.name)

                ld = string.ascii_letters + string.digits
                rstr = ''.join(random.choice(ld) for i in range(0, 9))

                temp_file = d_name.joinpath('{}_temp_{}{}'.format(f_base, rstr, f_ext))

                compress_raster(filename,
                                temp_file.as_posix(),
                                n_jobs=n_jobs,
                                gdal_cache=gdal_cache,
                                compress=compress_type)

                temp_file.rename(filename)

            if verbose > 0:
                logger.info('  Finished compressing')

    if verbose > 0:
        logger.info('\nFinished writing the data.')


def _arg_gen(arg_, iter_):
    for i_ in iter_:
        yield arg_


def apply(infile,
          outfile,
          block_func,
          args=None,
          count=1,
          scheduler='processes',
          gdal_cache=512,
          n_jobs=4,
          overwrite=False,
          **kwargs):

    """
    Applies a function and writes results to file

    Args:
        infile (str): The input file to process.
        outfile (str): The output file.
        block_func (func): The user function to apply to each block. The function should always return the window,
            the data, and at least one argument. The block data inside the function will be a 2d array if the
            input image has 1 band, otherwise a 3d array.
        args (Optional[tuple]): Additional arguments to pass to ``block_func``.
        count (Optional[int]): The band count for the output file.
        scheduler (Optional[str]): The ``concurrent.futures`` scheduler to use. Choices are ['threads', 'processes'].
        gdal_cache (Optional[int]): The ``GDAL`` cache size (in MB).
        n_jobs (Optional[int]): The number of blocks to process in parallel.
        overwrite (Optional[bool]): Whether to overwrite an existing output file.
        kwargs (Optional[dict]): Additional keyword arguments to pass to ``rasterio.open``.

    Returns:
        None

    Examples:
        >>> import geowombat as gw
        >>>
        >>> # Here is a function with no arguments
        >>> def my_func0(w, block, arg):
        >>>     return w, block
        >>>
        >>> gw.apply('input.tif',
        >>>          'output.tif',
        >>>           my_func0,
        >>>           n_jobs=8)
        >>>
        >>> # Here is a function with 1 argument
        >>> def my_func1(w, block, arg):
        >>>     return w, block * arg
        >>>
        >>> gw.apply('input.tif',
        >>>          'output.tif',
        >>>           my_func1,
        >>>           args=(10.0,),
        >>>           n_jobs=8)
    """

    if not args:
        args = (None,)

    if overwrite:

        if os.path.isfile(outfile):
            os.remove(outfile)

    kwargs['sharing'] = False
    kwargs['tiled'] = True

    io_mode = 'r+' if os.path.isfile(outfile) else 'w'

    out_indexes = 1 if count == 1 else list(range(1, count+1))

    futures_executor = concurrent.futures.ThreadPoolExecutor if scheduler == 'threads' else concurrent.futures.ProcessPoolExecutor

    with rio.Env(GDAL_CACHEMAX=gdal_cache):

        with rio.open(infile) as src:

            profile = src.profile.copy()

            if 'dtype' not in kwargs:
                kwargs['dtype'] = profile['dtype']

            if 'nodata' not in kwargs:
                kwargs['nodata'] = profile['nodata']

            if 'blockxsize' not in kwargs:
                kwargs['blockxsize'] = profile['blockxsize']

            if 'blockxsize' not in kwargs:
                kwargs['blockysize'] = profile['blockysize']

            # Create a destination dataset based on source params. The
            # destination will be tiled, and we'll process the tiles
            # concurrently.
            profile.update(count=count,
                           **kwargs)

            with rio.open(outfile, io_mode, **profile) as dst:

                # Materialize a list of destination block windows
                # that we will use in several statements below.
                # windows = get_window_offsets(src.height,
                #                              src.width,
                #                              blockysize,
                #                              blockxsize, return_as='list')

                # This generator comprehension gives us raster data
                # arrays for each window. Later we will zip a mapping
                # of it with the windows list to get (window, result)
                # pairs.
                # if nbands == 1:
                data_gen = (src.read(window=w, out_dtype=profile['dtype']) for ij, w in src.block_windows(1))
                # else:
                #
                #     data_gen = (np.array([rio.open(fn).read(window=w,
                #                                             out_dtype=dtype) for fn in infile], dtype=dtype)
                #                 for ij, w in src.block_windows(1))

                if args:
                    args = [_arg_gen(arg, src.block_windows(1)) for arg in args]

                with futures_executor(max_workers=n_jobs) as executor:

                    # Submit all of the tasks as futures
                    futures = [executor.submit(block_func,
                                               iter_[0][1],    # window object
                                               *iter_[1:])     # other arguments
                               for iter_ in zip(list(src.block_windows(1)), data_gen, *args)]

                    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):

                        out_window, out_block = f.result()

                        dst.write(np.squeeze(out_block),
                                  window=out_window,
                                  indexes=out_indexes)

                    # We map the block_func() function over the raster
                    # data generator, zip the resulting iterator with
                    # the windows list, and as pairs come back we
                    # write data to the destination dataset.
                    # for window_tuple, result in tqdm(zip(list(src.block_windows(1)),
                    #                                      executor.map(block_func,
                    #                                                   data_gen,
                    #                                                   *args)),
                    #                                  total=n_windows):
                    #
                    #     dst.write(result,
                    #               window=window_tuple[1],
                    #               indexes=out_indexes)


def _compress_dummy(w, block, dummy):

    """
    Dummy function to pass to concurrent writing
    """

    return w, block


def compress_raster(infile, outfile, n_jobs=1, gdal_cache=512, compress='lzw'):

    """
    Compresses a raster file

    Args:
        infile (str): The file to compress.
        outfile (str): The output file.
        n_jobs (Optional[int]): The number of concurrent blocks to write.
        gdal_cache (Optional[int]): The ``GDAL`` cache size (in MB).
        compress (Optional[str]): The compression method.

    Returns:
        None
    """

    with rio.open(infile) as src:

        profile = src.profile.copy()

        profile.update(compress=compress)

        apply(infile,
              outfile,
              _compress_dummy,
              scheduler='processes',
              args=(None,),
              gdal_cache=gdal_cache,
              n_jobs=n_jobs,
              count=src.count,
              dtype=src.profile['dtype'],
              nodata=src.profile['nodata'],
              tiled=src.profile['tiled'],
              blockxsize=src.profile['blockxsize'],
              blockysize=src.profile['blockysize'],
              compress=compress)


def to_geodataframe(data, mask=None, connectivity=4, num_workers=1):

    """
    Converts a ``dask`` array to a ``GeoDataFrame``

    Args:
        data (DataArray): The ``xarray.DataArray`` to convert.
        mask (Optional[str, numpy ndarray, or rasterio Band object]): Must evaluate to bool (rasterio.bool_ or rasterio.uint8).
            Values of False or 0 will be excluded from feature generation. Note well that this is the inverse sense from
            Numpy’s, where a mask value of True indicates invalid data in an array. If source is a Numpy masked array
            and mask is None, the source’s mask will be inverted and used in place of mask. if ``mask`` is equal to
            'source', then ``data`` is used as the mask.
        connectivity (Optional[int]): Use 4 or 8 pixel connectivity for grouping pixels into features.
        num_workers (Optional[int]): The number of parallel workers to send to ``dask.compute``.

    Returns:
        ``GeoDataFrame``

    Example:
        >>> import geowombat as gw
        >>>
        >>> with gw.open('image.tif') as src:
        >>>
        >>>     # Convert the input image to a GeoDataFrame
        >>>     df = gw.to_geodataframe(src,
        >>>                             mask='source',
        >>>                             num_workers=8)
    """

    if not hasattr(data, 'transform'):
        logger.exception("  The data should have a 'transform' object.")

    if not hasattr(data, 'crs'):
        logger.exception("  The data should have a 'crs' object.")

    if isinstance(mask, str):

        if mask == 'source':
            mask = data.astype('uint8').data.compute(num_workers=num_workers)

    poly_objects = shapes(data.data.compute(num_workers=num_workers),
                          mask=mask,
                          connectivity=connectivity,
                          transform=data.transform)

    poly_geom = [Polygon(p[0]['coordinates'][0]) for p in poly_objects]

    return gpd.GeoDataFrame(data=np.ones(len(poly_geom), dtype='uint8'),
                            geometry=poly_geom,
                            crs=data.crs)


def geodataframe_to_array(dataframe,
                          cellx,
                          celly,
                          band_name=None,
                          row_chunks=512,
                          col_chunks=512,
                          src_res=None,
                          fill=0,
                          default_value=1,
                          all_touched=True,
                          dtype='uint8'):

    """
    Converts a polygon ``geopandas.GeoDataFrame`` to an ``xarray.DataArray``.

    Args:
        dataframe (GeoDataFrame): The ``geopandas.DataFrame`` with polygon geometries.
        cellx (float): The output cell x size.
        celly (float): The output cell y size.
        band_name (Optional[list]): The ``xarray.DataArray`` band name.
        row_chunks (Optional[int]): The ``dask`` row chunk size.
        col_chunks (Optional[int]): The ``dask`` column chunk size.
        src_res (Optional[tuple]: A source resolution to align to.
        fill (Optional[int]): The output fill value for ``rasterio.features.rasterize``.
        default_value (Optional[int]): The output default value for ``rasterio.features.rasterize``.
        all_touched (Optional[int]): The ``all_touched`` value for ``rasterio.features.rasterize``.
        dtype (Optional[int]): The output data type for ``rasterio.features.rasterize``.

    Returns:
        ``xarray.DataArray``

    Example:
        >>> import geowombat as gw
        >>> import geopandas as gpd
        >>>
        >>> df = gpd.read_file('polygons.gpkg')
        >>>
        >>> # 100x100 cell size
        >>> data = gw.geodataframe_to_array(df, 100.0, 100.0)
        >>>
        >>> # Align to an existing image
        >>> with gw.open('image.tif') as src:
        >>>
        >>>     data = gw.geodataframe_to_array(df,
        >>>                                     src.gw.cellx,
        >>>                                     src.gw.celly,
        >>>                                     row_chunks=src.gw.row_chunks,
        >>>                                     col_chunks=src.gw.col_chunks,
        >>>                                     src_res=src_res)
    """

    if not band_name:
        band_name = [1]

    left, bottom, right, top = dataframe.bounds.values.flatten().tolist()

    dst_height = int((top - bottom) / celly)
    dst_width = int((right - left) / cellx)

    dst_transform = Affine(cellx, 0.0, left, 0.0, -celly, top)

    if src_res:

        dst_transform = aligned_target(dst_transform,
                                       dst_width,
                                       dst_height,
                                       src_res)[0]

        dst_transform = Affine(cellx, 0.0, dst_transform[2], 0.0, -celly, dst_transform[5])

    data = rasterize(dataframe.geometry.values,
                     out_shape=(dst_height, dst_width),
                     transform=dst_transform,
                     fill=fill,
                     default_value=default_value,
                     all_touched=all_touched,
                     dtype=dtype)

    xcoords = np.arange(left, left + dst_width * cellx, cellx)
    ycoords = np.arange(top, top - dst_height * celly, -celly)

    attrs = {'transform': dst_transform[:6],
             'crs': dataframe.crs,
             'res': (cellx, celly),
             'is_tiled': 1}

    return xr.DataArray(data=da.from_array(data[np.newaxis, :, :],
                                           chunks=(1, row_chunks, col_chunks)),
                        coords={'band': band_name,
                                'y': ycoords,
                                'x': xcoords},
                        dims=('band', 'y', 'x'),
                        attrs=attrs)
