import multiprocessing as multi
import concurrent.futures

from .windows import get_window_offsets

from tqdm import tqdm


_EXEC_DICT = {'mpool': multi.Pool,
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
            processes: process pool of workers using ``concurrent.futures``
            threads: thread pool of workers using ``concurrent.futures``
        n_workers (Optional[int]): The number of parallel workers for ``scheduler``.
        n_chunks (Optional[int]): The chunk size of windows. If not given, equal to ``n_workers`` x 50.

    Example:
        >>> import geowombat as gw
        >>> from geowombat.core.parallel import ParallelTask
        >>>
        >>> def user_func(*args):
        >>>     data, num_workers = list(itertools.chain(*args))
        >>>     results = data.data.sum().compute(scheduler='threads', num_workers=num_workers)
        >>>     return results
        >>>
        >>> # Process 8 windows in parallel using threads
        >>> # Process 4 dask chunks in parallel using threads
        >>> # 32 total workers are needed
        >>> with gw.open('image.tif') as src:
        >>>     pt = ParallelTask(src, n_workers=8)
        >>>     res = pt.map(user_func, 4)
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

    def map(self, func, *args):

        """
        Maps a function over a DataArray

        Args:
            func (func): The function to apply to the ``data`` chunks.

        Returns:
            ``list``: Results for each data chunk.
        """

        results = []

        # Iterate over the windows in chunks
        for wchunk in range(0, self.n_windows, self.n_chunks):

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

                for result in tqdm(map(func, data_gen), total=n_windows_slice):
                    results.append(result)

            else:

                with self.executor(self.n_workers) as executor:

                    if self.scheduler == 'mpool':

                        for result in tqdm(executor.imap_unordered(func, data_gen), total=n_windows_slice):
                            results.append(result)

                    else:

                        for result in tqdm(executor.map(func, data_gen), total=n_windows_slice):
                            results.append(result)

        return results
