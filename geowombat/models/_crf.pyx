# distutils: language = c++
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np

from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector
from libcpp.string cimport string

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


cdef cpp_map[string, double] _sample_to_dict(double[::1] tsamp):

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[string, double] features_map

    for t in range(1, tsamp_len+1):
        features_map[str(t).encode('utf-8')] = tsamp[t-1] * 0.0001

    return features_map


def time_to_crffeas(double[:, :, ::1] data,
                    unsigned int ntime,
                    unsigned int nrows,
                    unsigned int ncols):

    """
    Converts a time-shaped array to CRF features

    Args:
        data (list): Vector of length 'time'.

    Returns:
        ``list`` of feature dictionaries
    """

    cdef:
        Py_ssize_t i, j
        double[:, ::1] tdata
        double[::1] tsample
        vector[cpp_map[string, double]] samples
        vector[vector[cpp_map[string, double]]] samples_full

    for i in range(0, nrows*ncols):

        for j in range(0, ntime):

            tdata = data[j]
            tsample = tdata[i, :]

            samples.push_back(_sample_to_dict(tsample))

        samples_full.push_back(samples)

        samples.clear()

    return samples_full
