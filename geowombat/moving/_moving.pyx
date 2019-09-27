# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np


DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


# Define a function pointer to a metric.
ctypedef double (*metric_ptr)(double[:, ::1], Py_ssize_t, Py_ssize_t, unsigned int, double) nogil


cdef double _get_mean(double[:, ::1] input_view, Py_ssize_t i, Py_ssize_t j, unsigned int w, double w_samples) nogil:

    cdef:
        Py_ssize_t m, n
        double window_mean = 0.0

    for m in range(0, w):
        for n in range(0, w):
            window_mean += input_view[i+m, j+n]

    return window_mean / w_samples


cdef _moving_window2d(double[:, ::1] input,
                      str stat,
                      unsigned int window_size):

    cdef:
        Py_ssize_t i, j
        unsigned int rows = input.shape[0]
        unsigned int cols = input.shape[1]
        double w_samples = window_size * 2.0
        unsigned int hw = <int>(window_size / 2.0)
        unsigned int row_dims = rows - <int>(hw*2.0)
        unsigned int col_dims = cols - <int>(hw*2.0)
        double[:, ::1] output = np.zeros((rows, cols), dtype='float64')
        metric_ptr window_function

    if stat == 'mean':
        window_function = &_get_mean

    with nogil:

        for i in range(0, row_dims):
            for j in range(0, col_dims):
                output[i+hw, j+hw] = window_function(input, i, j, window_size, w_samples)

    return np.float64(output)


def moving_window(np.ndarray input not None, str stat='mean', w=3):
    return _moving_window2d(input, stat, w)