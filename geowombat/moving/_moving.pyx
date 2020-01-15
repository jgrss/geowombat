# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np

from cython.parallel import prange
from cython.parallel import parallel
from libc.stdlib cimport malloc, free, qsort

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


cdef extern from 'math.h':
   double sqrt(double val) nogil


# Define a function pointer to a metric.
ctypedef double (*metric_ptr)(double[:, :, ::1], Py_ssize_t, Py_ssize_t, Py_ssize_t, unsigned int, double, double) nogil


cdef double _get_var(double[:, :, ::1] input_view,
                     Py_ssize_t z,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     unsigned int w,
                     double w_samples,
                     double perc) nogil:

    cdef:
        Py_ssize_t m, n
        double window_mean = 0.0
        double window_var = 0.0

    # Mean
    for m in range(0, w):
        for n in range(0, w):
            window_mean += input_view[z, i+m, j+n]

    window_mean /= w_samples

    # Deviation from the mean
    for m in range(0, w):
        for n in range(0, w):
            window_var += (input_view[z, i+m, j+n] - window_mean)**2

    return window_var / w_samples


cdef double _get_std(double[:, :, ::1] input_view,
                     Py_ssize_t z,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     unsigned int w,
                     double w_samples,
                     double perc) nogil:

    cdef:
        Py_ssize_t m, n
        double window_mean = 0.0
        double window_var = 0.0

    # Mean
    for m in range(0, w):
        for n in range(0, w):
            window_mean += input_view[z, i+m, j+n]

    window_mean /= w_samples

    # Deviation from the mean
    for m in range(0, w):
        for n in range(0, w):
            window_var += (input_view[z, i+m, j+n] - window_mean)**2

    window_var /= w_samples

    return sqrt(window_var)


cdef double _get_mean(double[:, :, ::1] input_view,
                      Py_ssize_t z,
                      Py_ssize_t i,
                      Py_ssize_t j,
                      unsigned int w,
                      double w_samples,
                      double perc) nogil:

    cdef:
        Py_ssize_t m, n
        double window_mean = 0.0

    for m in range(0, w):
        for n in range(0, w):
            window_mean += input_view[z, i+m, j+n]

    return window_mean / w_samples


cdef double _get_min(double[:, :, ::1] input_view,
                     Py_ssize_t z,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     unsigned int w,
                     double w_samples,
                     double perc) nogil:

    cdef:
        Py_ssize_t m, n
        double window_min = 1e9

    for m in range(0, w):
        for n in range(0, w):

            if input_view[z, i+m, j+n] < window_min:
                window_min = input_view[z, i+m, j+n]

    return window_min


cdef double _get_max(double[:, :, ::1] input_view,
                     Py_ssize_t z,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     unsigned int w,
                     double w_samples,
                     double perc) nogil:

    cdef:
        Py_ssize_t m, n
        double window_max = -1e9

    for m in range(0, w):
        for n in range(0, w):

            if input_view[z, i+m, j+n] > window_max:
                window_max = input_view[z, i+m, j+n]

    return window_max


cdef int _cmp(const void * pa, const void * pb) nogil:

    cdef double a = (<double *>pa)[0]
    cdef double b = (<double *>pb)[0]

    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


cdef double _get_perc(double[:, :, ::1] input_view,
                      Py_ssize_t z,
                      Py_ssize_t i,
                      Py_ssize_t j,
                      unsigned int w,
                      double w_samples,
                      double perc) nogil:

    cdef:
        Py_ssize_t m, n, y
        int perc_index
        double *new_buffer = <double *>malloc((w * w) * sizeof(double))
        double perc_result

    y = 0

    for m in range(0, w):
        for n in range(0, w):

            new_buffer[y] = input_view[z, i+m, j+n]

            y += 1

    # Sort the buffer
    qsort(new_buffer, w, sizeof(double), _cmp)

    # cut of the percentile
    perc_index = <int>((<double>(w*w)) * perc / 100.0)
    perc_result = new_buffer[perc_index-1]

    # Deallocate the buffer
    free(new_buffer)

    return perc_result


cdef _moving_window3d(double[:, :, ::1] input,
                      str stat,
                      double perc,
                      unsigned int window_size,
                      unsigned int n_jobs):

    cdef:
        Py_ssize_t z, i, j
        unsigned int dims = input.shape[0]
        unsigned int rows = input.shape[1]
        unsigned int cols = input.shape[2]
        double w_samples = window_size * 2.0
        unsigned int hw = <int>(window_size / 2.0)
        unsigned int row_dims = rows - <int>(hw*2.0)
        unsigned int col_dims = cols - <int>(hw*2.0)
        double[:, :, ::1] output = np.zeros((dims, rows, cols), dtype='float64')
        double[:, :, ::1] output_view = output

        metric_ptr window_function

    if stat == 'mean':
        window_function = &_get_mean
    elif stat == 'std':
        window_function = &_get_std
    elif stat == 'var':
        window_function = &_get_var
    elif stat == 'min':
        window_function = &_get_min
    elif stat == 'max':
        window_function = &_get_max
    elif stat == 'perc':
        window_function = &_get_perc

    with nogil, parallel(num_threads=n_jobs):

        for z in prange(0, dims, schedule='static'):

            for i in range(0, row_dims):
                for j in range(0, col_dims):
                    output_view[z, i+hw, j+hw] = window_function(input, z, i, j, window_size, w_samples, perc)

    return np.float64(output)


def moving_window(np.ndarray input not None,
                  stat='mean',
                  perc=50.0,
                  w=3,
                  n_jobs=1):

    """
    Applies a moving window function over a NumPy array

    Args:
        input (NumPy array): The array to process.
        stat (Optional[str]): The statistic to compute. Choices are ['mean', 'std', 'var', 'min', 'max', 'perc'].
        perc (Optional[float]): The percentile to return if ``stat`` = 'perc'.
        w (Optional[int]): The moving window size (in pixels).
        n_jobs (Optional[int]): The number of bands to process in parallel.

    Returns:
        ``numpy.array``
    """

    return _moving_window3d(input, stat, perc, w, n_jobs)
