# distutils: language = c++
# cython: cdivision=True
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


cdef inline double _ndvi(double red, double nir):
    return (nir - red) / (nir + red)


cdef unicode _text(s):

    if type(s) is unicode:
        # Fast path for most common case(s).
        return <unicode>s

    elif isinstance(s, unicode):
        # We know from the fast path above that 's' can only be a subtype here.
        # An evil cast to <unicode> might still work in some(!) cases,
        # depending on what the further processing does.  To be safe,
        # we can always create a copy instead.
        return unicode(s)

    else:
        raise TypeError('Could not convert to unicode.')


cdef cpp_map[string, double] _sample_to_dict_pan(double[::1] tsamp):

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[string, double] features_map

    for t in range(1, tsamp_len+1):
        features_map[<bytes>str(t).encode('utf-8')] = tsamp[t-1] * 0.0001

    return features_map


cdef cpp_map[string, double] _sample_to_dict(double[::1] tsamp,
                                             double ndvi,
                                             bytes ndvi_string):

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[string, double] features_map

    for t in range(1, tsamp_len+1):
        features_map[<bytes>str(t).encode('utf-8')] = tsamp[t-1] * 0.0001

    features_map[ndvi_string] = ndvi

    return features_map


cdef vector[double] _push_class(vector[double] v,
                                cpp_map[string, double] ps,
                                bytes label):

    if ps.count(label) > 0:
        v.push_back(ps[label])
    else:
        v.push_back(0.0)

    return v


def cloud_crf_to_probas(vector[vector[cpp_map[string, double]]] pred,
                        unsigned int ntime,
                        unsigned int nrows,
                        unsigned int ncols):

    """
    Converts CRF cloud predictions to class labels

    Args:
        pred (vector): The CRF predictions.

    Returns:

    """

    cdef:
        vector[cpp_map[string, double]] pr
        cpp_map[string, double] ps
        vector[double] v1
        vector[vector[double]] v2
        vector[vector[vector[double]]] v3
        unsigned int n_classes = 5

        bytes land = 'l'.encode('utf-8')
        bytes urban = 'u'.encode('utf-8')
        bytes water = 'w'.encode('utf-8')
        bytes cloud = 'c'.encode('utf-8')
        bytes shadow = 's'.encode('utf-8')

    for pr in pred:

        for ps in pr:

            v1 = _push_class(v1, ps, land)      # land
            v1 = _push_class(v1, ps, urban)     # urban
            v1 = _push_class(v1, ps, water)     # water
            v1 = _push_class(v1, ps, cloud)     # cloud
            v1 = _push_class(v1, ps, shadow)    # shadow

            v2.push_back(v1)

            v1.clear()

        v3.push_back(v2)

        v2.clear()

    return np.array(v3, dtype='float64').transpose(1, 2, 0).reshape(ntime,
                                                                    n_classes,
                                                                    nrows,
                                                                    ncols)


def time_to_crffeas(double[:, :, ::1] data,
                    str sensor,
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
        unsigned int red_idx
        unsigned int nir_idx
        double ndvi
        bytes ndvi_string = str('ndvi').encode('utf-8')

    SENSOR_BANDS = dict(l7=dict(blue=0,
                                green=1,
                                red=2,
                                nir=3,
                                swir1=4,
                                swir2=5),
                        s2l7=dict(blue=0,
                                  green=1,
                                  red=2,
                                  nir=3,
                                  swir1=4,
                                  swir2=5))

    if sensor != 'pan':

        red_idx = SENSOR_BANDS[sensor]['red']
        nir_idx = SENSOR_BANDS[sensor]['nir']

    for i in range(0, nrows*ncols):

        for j in range(0, ntime):

            tdata = data[j]
            tsample = tdata[i, :]

            if sensor == 'pan':
                samples.push_back(_sample_to_dict_pan(tsample))
            else:

                ndvi = _ndvi(tsample[red_idx]*0.0001, tsample[nir_idx]*0.0001)

                samples.push_back(_sample_to_dict(tsample,
                                                  ndvi,
                                                  ndvi_string))

        samples_full.push_back(samples)

        samples.clear()

    return samples_full
