import multiprocessing as multi
import concurrent.futures

from .base import _executor_dummy
from .windows import get_window_offsets

import rasterio as rio
import xarray as xr
from tqdm import trange, tqdm


_EXEC_DICT = {
    'mpool': multi.Pool,
    'ray': None,
    'processes': concurrent.futures.ProcessPoolExecutor,
    'threads': concurrent.futures.ThreadPoolExecutor,
}


class ParallelTask(object):
    """A class for parallel tasks over a ``xarray.DataArray`` with returned
    results for each chunk.

    Args:
        data (DataArray): The ``xarray.DataArray`` to process.
        row_chunks (Optional[int]): The row chunk size to process in parallel.
        col_chunks (Optional[int]): The column chunk size to process in parallel.
        padding (Optional[tuple]): Padding for each window. ``padding`` should be given as a tuple
            of (left pad, bottom pad, right pad, top pad). If ``padding`` is given, the returned list will contain
            a tuple of ``rasterio.windows.Window`` objects as (w1, w2), where w1 contains the normal window offsets
            and w2 contains the padded window offsets.
        scheduler (Optional[str]): The parallel task scheduler to use. Choices are ['processes', 'threads', 'mpool'].

            mpool: process pool of workers using ``multiprocessing.Pool``
            ray: process pool of workers using ``ray.remote``.
            processes: process pool of workers using ``concurrent.futures``
            threads: thread pool of workers using ``concurrent.futures``

        get_ray (Optional[bool]): Whether to get results from ``ray`` futures.
        n_workers (Optional[int]): The number of parallel workers for ``scheduler``.
        n_chunks (Optional[int]): The chunk size of windows. If not given, equal to ``n_workers`` x 50.

    Examples:
        >>> import geowombat as gw
        >>> from geowombat.core.parallel import ParallelTask
        >>>
        >>> ########################
        >>> # Use concurrent threads
        >>> ########################
        >>>
        >>> def user_func_threads(*args):
        >>>     data, window_id, num_workers = list(itertools.chain(*args))
        >>>     return data.data.sum().compute(scheduler='threads', num_workers=num_workers)
        >>>
        >>> # Process 8 windows in parallel using threads
        >>> # Process 4 dask chunks in parallel using threads
        >>> # 32 total workers are needed
        >>> with gw.open('image.tif') as src:
        >>>     pt = ParallelTask(src, scheduler='threads', n_workers=8)
        >>>     res = pt.map(user_func_threads, 4)
        >>>
        >>> #########
        >>> # Use Ray
        >>> #########
        >>>
        >>> import ray
        >>>
        >>> @ray.remote
        >>> def user_func_ray(data_block_id, data_slice, window_id, num_workers):
        >>>     return (
        >>>         data_block_id[data_slice].data.sum()
        >>>         .compute(scheduler='threads', num_workers=num_workers)
        >>>     )
        >>>
        >>> ray.init(num_cpus=8)
        >>>
        >>> with gw.open('image.tif', chunks=512) as src:
        >>>     pt = ParallelTask(
        >>>         src, row_chunks=1024, col_chunks=1024, scheduler='ray', n_workers=8
        >>>     )
        >>>     res = ray.get(pt.map(user_func_ray, 4))
        >>>
        >>> ray.shutdown()
        >>>
        >>> #####################################
        >>> # Use with a dask.distributed cluster
        >>> #####################################
        >>>
        >>> from dask.distributed import LocalCluster
        >>>
        >>> with LocalCluster(
        >>>     n_workers=4,
        >>>     threads_per_worker=2,
        >>>     scheduler_port=0,
        >>>     processes=False,
        >>>     memory_limit='4GB'
        >>> ) as cluster:
        >>>
        >>>     with gw.open('image.tif') as src:
        >>>         pt = ParallelTask(src, scheduler='threads', n_workers=4, n_chunks=50)
        >>>         res = pt.map(user_func_threads, 2)
        >>>
        >>> # Map over multiple rasters
        >>> for pt in ParallelTask(
        >>>     ['image1.tif', 'image2.tif'], scheduler='threads', n_workers=4, n_chunks=500
        >>> ):
        >>>     res = pt.map(user_func_threads, 2)
    """

    def __init__(
        self,
        data=None,
        chunks=None,
        row_chunks=None,
        col_chunks=None,
        padding=None,
        scheduler='threads',
        get_ray=False,
        n_workers=1,
        n_chunks=None,
    ):

        self.chunks = 512 if not isinstance(chunks, int) else chunks

        if isinstance(data, list):
            self.data_list = data
            self.data = xr.open_rasterio(self.data_list[0])
        else:
            self.data = data

        self.row_chunks = row_chunks
        self.col_chunks = col_chunks
        self.padding = padding
        self.n = None
        self.scheduler = scheduler
        self.get_ray = get_ray
        self.executor = _EXEC_DICT[scheduler]
        self.n_workers = n_workers
        self.n_chunks = n_chunks
        self._in_session = False

        self.windows = None
        self.slices = None
        self.n_windows = None

        if not isinstance(self.n_chunks, int):
            self.n_chunks = self.n_workers * 50

        if not isinstance(data, list):
            self._setup()

    def __iter__(self):
        self.i_ = 0
        return self

    def __next__(self):

        self.data.close()

        if self.i_ >= len(self.data_list):
            raise StopIteration

        self.data = xr.open_rasterio(self.data_list[self.i_])
        self._setup()
        self.i_ += 1

        return self

    def __enter__(self):
        self._in_session = True
        return self

    def __exit__(self, *args, **kwargs):
        self._in_session = False

    def _setup(self):

        default_rchunks = (
            self.data.block_window(1, 0, 0).height
            if isinstance(self.data, rio.io.DatasetReader)
            else self.data.gw.row_chunks
        )
        default_cchunks = (
            self.data.block_window(1, 0, 0).width
            if isinstance(self.data, rio.io.DatasetReader)
            else self.data.gw.col_chunks
        )

        rchunksize = (
            self.row_chunks
            if isinstance(self.row_chunks, int)
            else default_rchunks
        )
        cchunksize = (
            self.col_chunks
            if isinstance(self.col_chunks, int)
            else default_cchunks
        )

        self.windows = get_window_offsets(
            self.data.height
            if isinstance(self.data, rio.io.DatasetReader)
            else self.data.gw.nrows,
            self.data.width
            if isinstance(self.data, rio.io.DatasetReader)
            else self.data.gw.ncols,
            rchunksize,
            cchunksize,
            return_as='list',
            padding=self.padding,
        )

        # Convert windows into slices
        if len(self.data.shape) == 2:
            self.slices = [
                (
                    slice(w.row_off, w.row_off + w.height),
                    slice(w.col_off, w.col_off + w.width),
                )
                for w in self.windows
            ]
        else:
            self.slices = [
                tuple([slice(0, None)] * (len(self.data.shape) - 2))
                + (
                    slice(w.row_off, w.row_off + w.height),
                    slice(w.col_off, w.col_off + w.width),
                )
                for w in self.windows
            ]

        self.n_windows = len(self.windows)

    def map(self, func, *args, **kwargs):

        """Maps a function over a DataArray.

        Args:
            func (func): The function to apply to the ``data`` chunks.

                When using any scheduler other than 'ray' (i.e., 'mpool', 'threads', 'processes'), the function
                should always be defined with ``*args``. With these schedulers, the function will always return
                the ``DataArray`` window and window id as the first two arguments. If no user arguments are
                passed to ``map`` , the function will look like:

                    def my_func(*args):
                        data, window_id = list(itertools.chain(*args))
                        # do something
                        return results

                If user arguments are passed, e.g., ``map(my_func, arg1, arg2)``, the function will look like:

                    def my_func(*args):
                        data, window_id, arg1, arg2 = list(itertools.chain(*args))
                        # do something
                        return results

                When ``scheduler`` = 'ray', the user function requires an additional slice argument that looks like:

                    @ray.remote
                    def my_ray_func(data_block_id, data_slice, window_id):
                        # do something
                        return results

                Note the addition of the ``@ray.remote`` decorator, as well as the explicit arguments in the function
                call. Extra user arguments would look like:

                    @ray.remote
                    def my_ray_func(data_block_id, data_slice, window_id, arg1, arg2):
                        # do something
                        return results

                Other ``ray`` classes can also be used in place of a function.

            args (items): Function arguments.
            kwargs (Optional[dict]): Keyword arguments passed to ``multiprocessing.Pool().imap``.

        Returns:
            ``list``: Results for each data chunk.

        Examples:
            >>> import geowombat as gw
            >>> from geowombat.core.parallel import ParallelTask
            >>> from geowombat.data import l8_224078_20200518_points, l8_224078_20200518
            >>> import geopandas as gpd
            >>> import rasterio as rio
            >>> import ray
            >>> from ray.util import ActorPool
            >>>
            >>> @ray.remote
            >>> class Actor(object):
            >>>
            >>>     def __init__(self, aoi_id=None, id_column=None, band_names=None):
            >>>
            >>>         self.aoi_id = aoi_id
            >>>         self.id_column = id_column
            >>>         self.band_names = band_names
            >>>
            >>>     # While the names can differ, these three arguments are required.
            >>>     # For ``ParallelTask``, the callable function within an ``Actor`` must be named exec_task.
            >>>     def exec_task(self, data_block_id, data_slice, window_id):
            >>>
            >>>         data_block = data_block_id[data_slice]
            >>>         left, bottom, right, top = data_block.gw.bounds
            >>>         aoi_sub = self.aoi_id.cx[left:right, bottom:top]
            >>>
            >>>         if aoi_sub.empty:
            >>>             return aoi_sub
            >>>
            >>>         # Return a GeoDataFrame for each actor
            >>>         return gw.extract(data_block,
            >>>                           aoi_sub,
            >>>                           id_column=self.id_column,
            >>>                           band_names=self.band_names)
            >>>
            >>> ray.init(num_cpus=8)
            >>>
            >>> band_names = [1, 2, 3]
            >>> df_id = ray.put(gpd.read_file(l8_224078_20200518_points))
            >>>
            >>> with rio.Env(GDAL_CACHEMAX=256*1e6) as env:
            >>>
            >>>     # Since we are iterating over the image block by block, we do not need to load
            >>>     # a lazy dask array (i.e., chunked).
            >>>     with gw.open(l8_224078_20200518, band_names=band_names, chunks=None) as src:
            >>>
            >>>         # Setup the pool of actors, one for each resource available to ``ray``.
            >>>         actor_pool = ActorPool([Actor.remote(aoi_id=df_id, id_column='id', band_names=band_names)
            >>>                                 for n in range(0, int(ray.cluster_resources()['CPU']))])
            >>>
            >>>         # Setup the task object
            >>>         pt = ParallelTask(src, row_chunks=4096, col_chunks=4096, scheduler='ray', n_chunks=1000)
            >>>         results = pt.map(actor_pool)
            >>>
            >>> del df_id, actor_pool
            >>>
            >>> ray.shutdown()
        """

        if (self.n_workers == 1) or (self.scheduler == 'ray'):
            executor_pool = _executor_dummy
            ranger = range
        else:
            executor_pool = self.executor
            ranger = trange

        if self.scheduler == 'ray':

            if self.padding:
                raise SyntaxError('Ray cannot be used with array padding.')

            import ray

            if isinstance(self.data, rio.io.DatasetReader):
                data_id = self.data.name
            else:
                data_id = ray.put(self.data)

        results = []

        with executor_pool(self.n_workers) as executor:

            # Iterate over the windows in chunks
            for wchunk in ranger(0, self.n_windows, self.n_chunks):

                if self.padding:

                    window_slice = self.windows[
                        wchunk : wchunk + self.n_chunks
                    ]

                    # Read the padded window
                    if len(self.data.shape) == 2:
                        data_gen = (
                            (
                                self.data[
                                    w[1].row_off : w[1].row_off + w[1].height,
                                    w[1].col_off : w[1].col_off + w[1].width,
                                ],
                                widx + wchunk,
                                *args,
                            )
                            for widx, w in enumerate(window_slice)
                        )
                    elif len(self.data.shape) == 3:
                        data_gen = (
                            (
                                self.data[
                                    :,
                                    w[1].row_off : w[1].row_off + w[1].height,
                                    w[1].col_off : w[1].col_off + w[1].width,
                                ],
                                widx + wchunk,
                                *args,
                            )
                            for widx, w in enumerate(window_slice)
                        )
                    else:
                        data_gen = (
                            (
                                self.data[
                                    :,
                                    :,
                                    w[1].row_off : w[1].row_off + w[1].height,
                                    w[1].col_off : w[1].col_off + w[1].width,
                                ],
                                widx + wchunk,
                                *args,
                            )
                            for widx, w in enumerate(window_slice)
                        )

                else:

                    window_slice = self.slices[wchunk : wchunk + self.n_chunks]

                    if self.scheduler == 'ray':
                        data_gen = (
                            (data_id, slice_, widx + wchunk, *args)
                            for widx, slice_ in enumerate(window_slice)
                        )
                    else:
                        data_gen = (
                            (self.data[slice_], widx + wchunk, *args)
                            for widx, slice_ in enumerate(window_slice)
                        )

                if (self.n_workers == 1) and (self.scheduler != 'ray'):

                    for result in map(func, data_gen):
                        results.append(result)

                else:

                    if self.scheduler == 'mpool':

                        for result in executor.imap(func, data_gen, **kwargs):
                            results.append(result)

                    elif self.scheduler == 'ray':

                        if isinstance(func, ray.util.actor_pool.ActorPool):

                            for result in tqdm(
                                func.map(
                                    lambda a, v: a.exec_task.remote(*v),
                                    data_gen,
                                ),
                                total=len(window_slice),
                            ):
                                results.append(result)

                        else:

                            if isinstance(func, ray.actor.ActorHandle):
                                futures = [
                                    func.exec_task.remote(*dargs)
                                    for dargs in data_gen
                                ]
                            else:
                                futures = [
                                    func.remote(*dargs) for dargs in data_gen
                                ]

                            if self.get_ray:

                                with tqdm(total=len(futures)) as pbar:

                                    results_ = []
                                    while len(futures):

                                        done_id, futures = ray.wait(futures)
                                        results_.append(ray.get(done_id[0]))

                                        pbar.update(1)

                                results += results_

                            else:
                                results += futures

                    else:

                        for result in executor.map(func, data_gen):
                            results.append(result)

        if self.scheduler == 'ray':
            del data_id

        return results
