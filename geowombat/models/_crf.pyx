# distutils: language = c++
# cython: profile=False
# cython: cdivision=True
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


cdef cpp_map[string, double] _sample_to_dict(double[::1] array_sample,
                                             unsigned int nfeas,
                                             vector[string] band_names_):

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t k
        cpp_map[string, double] features_map_sub

    for k in range(0, nfeas):
        features_map_sub[band_names_[k]] = array_sample[k]

    return features_map_sub


def time_to_crffeas(cpp_map[int, double[:, ::1]] features_map,
                    vector[char] band_names,
                    unsigned int nfeas):

    """
    Converts a time-shaped array to CRF features

    Args:
        data (4d array): time x bands x rows x columns.

    Example:
        time_to_features(array, ['swir1', 'swir2', 'date-diff', 'b-b'])

    Returns:
        ``list`` of feature dictionaries
    """

    # cdef:
    #     Py_ssize_t ti, si, i, j, m
        # unsigned int ntime = data.shape[0]
        # unsigned int nbands = data.shape[1]
        # unsigned int nrows = data.shape[2]
        # unsigned int ncols = data.shape[3]
        # unsigned int nsamps = <int>(nrows * ncols)
        # np.ndarray[double, ndim=3] tlayer
        # cpp_map[int, double[:, ::1]] features_map
        # vector[double[::1]] feature_vector_temp
        # vector[cpp_map[string, double]] feature_vector
        # unsigned int nfeas = len(band_names)
        # vector[char] band_names_

    # for si in range(0, nfeas):
    #     band_names_.push_back(band_names[si].encode('utf8'))

    print(band_names)

    # for ti in range(0, ntime):
    #
    #     tlayer = data[<int>ti]
    #     features_map[<int>ti] = np.ascontiguousarray(tlayer.transpose(1, 2, 0).reshape(nsamps, nbands), dtype='float64')
    print(features_map)
    # for i in range(0, nsamps):
    #
    #     for j in range(0, ntime):
    #         feature_vector_temp.push_back(features_map[j][i])
    #
    # for m in range(0, nsamps*ntime):
    #
    #     feature_vector.push_back(_sample_to_dict(feature_vector_temp[m],
    #                                              nfeas,
    #                                              band_names_))
    #
    # return feature_vector
