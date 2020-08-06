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

from cython.parallel import prange
from cython.parallel import parallel


cdef extern from 'math.h':
   double floor(double val) nogil


cdef extern from 'math.h':
   double sqrt(double val) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(double val) nogil


# Define a function pointer to a metric.
ctypedef double (*metric_ptr)(double[:, ::1], Py_ssize_t, Py_ssize_t, int, double, double, double[:, ::1]) nogil


cdef inline int _get_rindex(int col_dims, Py_ssize_t index) nogil:
    return <int>floor(<double>index / <double>col_dims)


cdef inline int _get_cindex(int col_dims, Py_ssize_t index, int row_index) nogil:
    return <int>(index - <double>col_dims * row_index)


cdef inline double _pow2(double value) nogil:
    return value*value


cdef inline double _edist(double xloc, double yloc, double hw) nogil:
    return sqrt(_pow2(xloc - hw) + _pow2(yloc - hw))


cdef double _get_var(double[:, ::1] input_view,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     int w,
                     double w_samples,
                     double nodata,
                     double[:, ::1] window_weights) nogil:

    cdef:
        Py_ssize_t m, n
        double window_mean = 0.0
        double window_var = 0.0
        double center_value, weight_value
        double wsum = 0.0
        double res

    # Mean
    for m in range(0, w):
        for n in range(0, w):

            weight_value = window_weights[m, n]
            center_value = input_view[i+m, j+n]

            if nodata == 1e9:

                window_mean += (input_view[i+m, j+n] * weight_value)
                wsum += weight_value

            else:

                if center_value != nodata:

                    window_mean += (input_view[i+m, j+n] * weight_value)
                    wsum += weight_value

    window_mean /= wsum

    # Deviation from the mean
    for m in range(0, w):
        for n in range(0, w):

            weight_value = window_weights[m, n]
            center_value = input_view[i+m, j+n]

            if nodata == 1e9:
                window_var += _pow2(input_view[i+m, j+n] * weight_value - window_mean)
            else:

                if center_value != nodata:
                    window_var += _pow2(input_view[i+m, j+n] * weight_value - window_mean)

    res = window_var / wsum

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
                     int w,
                     double w_samples,
                     double nodata,
                     double[:, ::1] window_weights) nogil:

    cdef:
        Py_ssize_t m, n
        double window_mean = 0.0
        double window_var = 0.0
        double center_value, weight_value
        double wsum = 0.0
        double res

    # Mean
    for m in range(0, w):
        for n in range(0, w):

            weight_value = window_weights[m, n]
            center_value = input_view[i+m, j+n]

            if nodata == 1e9:

                window_mean += (input_view[i+m, j+n] * weight_value)
                wsum += weight_value

            else:

                if center_value != nodata:

                    window_mean += (input_view[i+m, j+n] * weight_value)
                    wsum += weight_value

    window_mean /= wsum

    # Deviation from the mean
    for m in range(0, w):
        for n in range(0, w):

            weight_value = window_weights[m, n]
            center_value = input_view[i+m, j+n]

            if nodata == 1e9:
                window_var += _pow2(input_view[i+m, j+n] * weight_value - window_mean)
            else:

                if center_value != nodata:
                    window_var += _pow2(input_view[i+m, j+n] * weight_value - window_mean)

    window_var /= wsum

    res = sqrt(window_var)

    if nodata == 1e9:
        return res
    else:

        if npy_isnan(res):
            return nodata
        else:
            return res


cdef double[:, ::1] _moving_window(double[:, ::1] indata,
                                   double[:, ::1] output,
                                   int window_size,
                                   double nodata,
                                   bint weights,
                                   unsigned int n_jobs,
                                   int method):

    cdef:
        Py_ssize_t f, wi, wj
        int i, j
        unsigned int rows = indata.shape[0]
        unsigned int cols = indata.shape[1]
        double w_samples = window_size * 2.0
        int hw = <int>(window_size / 2.0)
        unsigned int row_dims = rows - window_size
        unsigned int col_dims = cols - window_size

        unsigned int nsamples = <int>(row_dims * col_dims)

        double[:, ::1] window_weights = np.ones((window_size, window_size), dtype='float64')
        double max_dist

        metric_ptr window_function

    if weights:

        with nogil:

            for wi in range(0, window_size):
                for wj in range(0, window_size):
                    window_weights[wi, wj] = _edist(<double>wj, <double>wi, <double>hw)

            max_dist = _edist(0.0, 0.0, <double>hw)

            for wi in range(0, window_size):
                for wj in range(0, window_size):
                    window_weights[wi, wj] = 1.0 - (window_weights[wi, wj] / max_dist)

    if method == 1:

        # normal moving window without a metric pointer

        with nogil:

            for i in range(0, row_dims):
                for j in range(0, col_dims):

                    output[i+hw, j+hw] = _get_std(indata,
                                                  i, j,
                                                  window_size,
                                                  w_samples,
                                                  nodata,
                                                  window_weights)

    elif method == 2:

        # normal moving window with a metric pointer

        window_function = & _get_std

        with nogil:

            for i in range(0, row_dims):
                for j in range(0, col_dims):

                    output[i+hw, j+hw] = window_function(indata,
                                                         i, j,
                                                         window_size,
                                                         w_samples,
                                                         nodata,
                                                         window_weights)

    elif method == 3:

        # untraditional moving window without parallelism

        window_function = & _get_std

        with nogil:

            for f in range(0, nsamples):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                output[i+hw, j+hw] = window_function(indata,
                                                     i, j,
                                                     window_size,
                                                     w_samples,
                                                     nodata,
                                                     window_weights)

    elif method == 4:

        # untraditional moving window with parallelism

        window_function = &_get_std

        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                output[i+hw, j+hw] = window_function(indata,
                                                     i, j,
                                                     window_size,
                                                     w_samples,
                                                     nodata,
                                                     window_weights)

    return output


def moving_window(np.ndarray indata not None,
                  w=3,
                  nodata=1e9,
                  weights=False,
                  n_jobs=1,
                  method=1):
    
    """
    Applies a moving window function over a NumPy array

    Args:
        indata (2d NumPy array): The array to process.
        w (Optional[int]): The moving window size (in pixels).
        nodata (Optional[int or float]): A 'no data' value to ignore.
        weights (Optional[bool]): Whether to weight values by distance from window center.
        n_jobs (Optional[int]): The number of bands to process in parallel.

    Returns:
        2d ``numpy.array``
    """

    cdef:
        double[:, ::1] output = np.float64(indata).copy()

    return np.float64(_moving_window(indata, output, w, nodata, weights, n_jobs, method))
