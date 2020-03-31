# distutils: language=c++
# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython
cimport cython

import numpy as np
cimport numpy as np

from ..util cimport percentiles

from cython.parallel import prange
from cython.parallel import parallel

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


cdef extern from 'math.h':
   double sqrt(double val) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(double val) nogil


# Define a function pointer to a metric.
ctypedef double (*metric_ptr)(double[:, ::1], Py_ssize_t, Py_ssize_t, unsigned int, double, double) nogil


cdef double _get_var(double[:, ::1] input_view,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     unsigned int w,
                     double w_samples,
                     double nodata) nogil:

    cdef:
        Py_ssize_t m, n
        double window_mean = 0.0
        double window_var = 0.0
        double center_value
        unsigned int count = 0
        double res

    # Mean
    for m in range(0, w):
        for n in range(0, w):

            center_value = input_view[i+m, j+n]

            if nodata == 1e9:

                window_mean += input_view[i+m, j+n]
                count += 1

            else:

                if center_value != nodata:

                    window_mean += input_view[i+m, j+n]
                    count += 1

    window_mean /= <double>count

    count = 0

    # Deviation from the mean
    for m in range(0, w):
        for n in range(0, w):

            center_value = input_view[i+m, j+n]

            if nodata == 1e9:

                window_var += (input_view[i+m, j+n] - window_mean)**2
                count += 1

            else:

                if center_value != nodata:

                    window_var += (input_view[i+m, j+n] - window_mean)**2
                    count += 1

    res = window_var / <double>count

    if nodata == 1e9:
        return res
    else:

        if npy_isnan(res):
            return nodata
        else:
            return res


cdef double _get_std(double[:, ::1] input_view,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     unsigned int w,
                     double w_samples,
                     double nodata) nogil:

    cdef:
        Py_ssize_t m, n
        double window_mean = 0.0
        double window_var = 0.0
        double center_value
        unsigned int count = 0
        double res

    # Mean
    for m in range(0, w):
        for n in range(0, w):

            center_value = input_view[i+m, j+n]

            if nodata == 1e9:

                window_mean += input_view[i+m, j+n]
                count += 1

            else:

                if center_value != nodata:

                    window_mean += input_view[i+m, j+n]
                    count += 1

    window_mean /= <double>count

    count = 0

    # Deviation from the mean
    for m in range(0, w):
        for n in range(0, w):

            center_value = input_view[i+m, j+n]

            if nodata == 1e9:

                window_var += (input_view[i+m, j+n] - window_mean)**2
                count += 1

            else:

                if center_value != nodata:

                    window_var += (input_view[i+m, j+n] - window_mean)**2
                    count += 1

    window_var /= <double>count

    res = sqrt(window_var)

    if nodata == 1e9:
        return res
    else:

        if npy_isnan(res):
            return nodata
        else:
            return res


cdef double _get_mean(double[:, ::1] input_view,
                      Py_ssize_t i,
                      Py_ssize_t j,
                      unsigned int w,
                      double w_samples,
                      double nodata) nogil:

    cdef:
        Py_ssize_t m, n
        double window_mean = 0.0
        double center_value
        unsigned int count = 0
        double res

    for m in range(0, w):
        for n in range(0, w):

            center_value = input_view[i+m, j+n]

            if nodata == 1e9:

                window_mean += input_view[i+m, j+n]
                count += 1

            else:

                if center_value != nodata:

                    window_mean += input_view[i+m, j+n]
                    count += 1

    res = window_mean / <double>count

    if nodata == 1e9:
        return res
    else:

        if npy_isnan(res):
            return nodata
        else:
            return res


cdef double _get_min(double[:, ::1] input_view,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     unsigned int w,
                     double w_samples,
                     double nodata) nogil:

    cdef:
        Py_ssize_t m, n
        double window_min = 1e9
        double center_value

    for m in range(0, w):
        for n in range(0, w):

            center_value = input_view[i+m, j+n]

            if nodata == 1e9:

                if center_value < window_min:
                    window_ = center_value

            else:

                if (center_value < window_min) and (center_value != nodata):
                    window_min = center_value

    if nodata == 1e9:
        return window_min
    else:

        if npy_isnan(window_min):
            return nodata
        else:
            return window_min


cdef double _get_max(double[:, ::1] input_view,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     unsigned int w,
                     double w_samples,
                     double nodata) nogil:

    cdef:
        Py_ssize_t m, n
        double window_max = -1e9
        double center_value

    for m in range(0, w):
        for n in range(0, w):

            center_value = input_view[i+m, j+n]

            if nodata == 1e9:

                if center_value > window_max:
                    window_max = center_value

            else:

                if (center_value > window_max) and (center_value != nodata):
                    window_max = center_value

    if nodata == 1e9:
        return window_max
    else:

        if npy_isnan(window_max):
            return nodata
        else:
            return window_max


cdef double[:, ::1] _moving_window(double[:, ::1] indata,
                                   double[:, ::1] output,
                                   str stat,
                                   unsigned int perc,
                                   unsigned int window_size,
                                   double nodata,
                                   unsigned int n_jobs):

    cdef:
        Py_ssize_t c, i, j, f
        unsigned int rows = indata.shape[0]
        unsigned int cols = indata.shape[1]
        double w_samples = window_size * 2.0
        unsigned int hw = <int>(window_size / 2.0)
        unsigned int row_dims = rows - <int>(hw*2.0)
        unsigned int col_dims = cols - <int>(hw*2.0)
        double percf = <double>perc

        unsigned int nsamples = <int>(row_dims * col_dims)
        long[::1] array_index_rows = np.zeros(nsamples, dtype='int64')
        long[::1] array_index_cols = np.zeros(nsamples, dtype='int64')

        metric_ptr window_function

    with nogil:

        c = 0
        for i in range(0, row_dims):
            for j in range(0, row_dims):
                array_index_rows[c] = i
                array_index_cols[c] = j
                c += 1

    if stat == 'perc':

        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                output[array_index_rows[f]+hw, array_index_cols[f]+hw] = percentiles.get_perc2d(indata,
                                                                                                array_index_rows[f],
                                                                                                array_index_cols[f],
                                                                                                window_size,
                                                                                                nodata,
                                                                                                percf)

    else:

        if stat == 'mean':
            window_function = & _get_mean
        elif stat == 'std':
            window_function = & _get_std
        elif stat == 'var':
            window_function = & _get_var
        elif stat == 'min':
            window_function = & _get_min
        elif stat == 'max':
            window_function = & _get_max

        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                output[array_index_rows[f]+hw, array_index_cols[f]+hw] = window_function(indata,
                                                                                         array_index_rows[f],
                                                                                         array_index_cols[f],
                                                                                         window_size,
                                                                                         w_samples,
                                                                                         nodata)

    return output


def moving_window(np.ndarray indata not None,
                  stat='mean',
                  perc=50,
                  w=3,
                  nodata=1e9,
                  n_jobs=1):

    """
    Applies a moving window function over a NumPy array

    Args:
        indata (2d NumPy array): The array to process.
        stat (Optional[str]): The statistic to compute. Choices are ['mean', 'std', 'var', 'min', 'max', 'perc'].
        perc (Optional[int]): The percentile to return if ``stat`` = 'perc'.
        w (Optional[int]): The moving window size (in pixels).
        nodata (Optional[int or float]): A 'no data' value to ignore.
        n_jobs (Optional[int]): The number of bands to process in parallel.

    Returns:
        2d ``numpy.array``
    """

    cdef:
        double[:, ::1] output = np.float64(indata).copy()

    return np.float64(_moving_window(indata, output, stat, perc, w, nodata, n_jobs))
