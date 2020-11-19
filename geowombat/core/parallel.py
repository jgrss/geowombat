import multiprocessing as multi
import concurrent.futures

from .base import _executor_dummy
from .windows import get_window_offsets

from tqdm import trange


_EXEC_DICT = {'mpool': multi.Pool,
              'ray': None,
              'processes': concurrent.futures.ProcessPoolExecutor,
              'threads': concurrent.futures.ThreadPoolExecutor}


class ParallelTask(object):

    """
    A class for parallel tasks over a DataArray with returned results for each chunk

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

        n_workers (Optional[int]): The number of parallel workers for ``scheduler``.
        n_chunks (Optional[int]): The chunk size of windows. If not given, equal to ``n_workers`` x 50.

    Example:
        >>> import geowombat as gw
        >>> from geowombat.core.parallel import ParallelTask
        >>>
        >>> def user_func_threads(*args):
        >>>     data, window_id, num_workers = list(itertools.chain(*args))
        >>>     return data.data.sum().compute(scheduler='threads', num_workers=num_workers)
        >>>
        >>> # Process 8 windows in parallel using threads
        >>> # Process 4 dask chunks in parallel using threads
        >>> # 32 total workers are needed
        >>> with gw.open('image.tif') as src:
        >>>     pt = ParallelTask(src, n_workers=8)
        >>>     res = pt.map(user_func_threads, 4)
        >>>
        >>> import ray
        >>>
        >>> @ray.remote
        >>> def user_func_ray(data, window_id, num_workers):
        >>>     return data.data.sum().compute(scheduler='threads', num_workers=num_workers)
        >>>
        >>> ray.init(num_cpus=8)
        >>>
        >>> with gw.open('image.tif') as src:
        >>>     pt = ParallelTask(src, n_workers=8)
        >>>     res = ray.get(pt.map(user_func_ray, 4))
        >>>
        >>> ray.shutdown()
    """

    def __init__(self,
                 data,
                 row_chunks=None,
                 col_chunks=None,
                 padding=None,
                 scheduler='threads',
                 n_workers=1,
                 n_chunks=None):

        self.data = data
        self.padding = padding
        self.n = None
        self.scheduler = scheduler
        self.executor = _EXEC_DICT[scheduler]
        self.n_workers = n_workers
        self.n_chunks = n_chunks

        self.windows = None
        self.slices = None
        self.n_windows = None

        if not isinstance(self.n_chunks, int):
            self.n_chunks = self.n_workers * 50

        self._setup(row_chunks, col_chunks)

    def _setup(self, row_chunks, col_chunks):

        rchunksize = row_chunks if isinstance(row_chunks, int) else self.data.gw.row_chunks
        cchunksize = col_chunks if isinstance(col_chunks, int) else self.data.gw.col_chunks

        self.windows = get_window_offsets(self.data.gw.nrows,
                                          self.data.gw.ncols,
                                          rchunksize,
                                          cchunksize,
                                          return_as='list',
                                          padding=self.padding)

        # Convert windows into slices
        if len(self.data.shape) == 2:
            self.slices = [(slice(w.row_off, w.row_off+w.height), slice(w.col_off, w.col_off+w.width)) for w in self.windows]
        else:
            self.slices = [tuple([slice(0, None)] * (len(self.data.shape)-2)) + (slice(w.row_off, w.row_off+w.height), slice(w.col_off, w.col_off+w.width)) for w in self.windows]

        self.n_windows = len(self.windows)

    def map(self, func, *args, **kwargs):

        """
        Maps a function over a DataArray

        Args:
            func (func): The function to apply to the ``data`` chunks.
            args (items): Function arguments.
            kwargs (Optional[dict]): Keyword arguments passed to ``multiprocessing.Pool().imap``.

        Returns:
            ``list``: Results for each data chunk.
        """

        executor_pool = _executor_dummy if (self.n_workers == 1) or (self.scheduler == 'ray') else self.executor

        results = []

        with executor_pool(self.n_workers) as executor:

            # Iterate over the windows in chunks
            for wchunk in trange(0, self.n_windows, self.n_chunks):

                if self.padding:

                    window_slice = self.windows[wchunk:wchunk+self.n_chunks]
                    n_windows_slice = len(window_slice)

                    # Read the padded window
                    if len(self.data.shape) == 2:
                        data_gen = ((self.data[w[1].row_off:w[1].row_off + w[1].height, w[1].col_off:w[1].col_off + w[1].width], widx+wchunk, *args) for widx, w in enumerate(window_slice))
                    elif len(self.data.shape) == 3:
                        data_gen = ((self.data[:, w[1].row_off:w[1].row_off + w[1].height, w[1].col_off:w[1].col_off + w[1].width], widx+wchunk, *args) for widx, w in enumerate(window_slice))
                    else:
                        data_gen = ((self.data[:, :, w[1].row_off:w[1].row_off + w[1].height, w[1].col_off:w[1].col_off + w[1].width], widx+wchunk, *args) for widx, w in enumerate(window_slice))

                else:

                    window_slice = self.slices[wchunk:wchunk+self.n_chunks]
                    n_windows_slice = len(window_slice)

                    data_gen = ((self.data[slice_], widx+wchunk, *args) for widx, slice_ in enumerate(window_slice))

                if self.n_workers == 1:

                    for result in map(func, data_gen):
                        results.append(result)

                else:

                    if self.scheduler == 'mpool':

                        for result in executor.imap(func, data_gen, **kwargs):
                            results.append(result)

                    elif self.scheduler == 'ray':

                        for dargs in data_gen:
                            results.append(func.remote(*dargs))

                    else:

                        for result in executor.map(func, data_gen):
                            results.append(result)

        return results
