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
from libcpp.vector cimport vector as cpp_vector
from libcpp.string cimport string as cpp_string

ctypedef char* char_ptr


cdef inline double _ndvi(double red, double nir) nogil:
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


cdef cpp_map[cpp_string, double] _sample_to_dict_pan(double[::1] tsamp,
                                                     cpp_vector[cpp_string] string_ints) nogil:

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[cpp_string, double] features_map

    for t in range(0, tsamp_len):
        features_map[string_ints[t]] = tsamp[t] * 0.0001

    return features_map


cdef cpp_map[cpp_string, double] _sample_to_dict(double[::1] tsamp,
                                                 cpp_vector[cpp_string] string_ints,
                                                 double ndvi,
                                                 cpp_string ndvi_string) nogil:

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[cpp_string, double] features_map

    for t in range(0, tsamp_len):
        features_map[string_ints[t]] = tsamp[t] * 0.0001

    features_map[ndvi_string] = ndvi

    return features_map


cdef cpp_vector[double] _push_classes(cpp_vector[double] vct,
                                      cpp_map[cpp_string, double] ps,
                                      cpp_vector[cpp_string] labels_bytes,
                                      unsigned int n_classes) nogil:

    cdef:
        Py_ssize_t m

    for m in range(0, n_classes):

        if ps.count(labels_bytes[m]) > 0:
            vct.push_back(ps[labels_bytes[m]])
        else:
            vct.push_back(0.0)

    return vct


def probas_to_labels(cpp_vector[cpp_vector[cpp_map[cpp_string, double]]] pred,
                     cpp_vector[cpp_string] labels,
                     unsigned int n_classes,
                     unsigned int ntime,
                     unsigned int nrows,
                     unsigned int ncols):

    """
    Converts CRF predictions to class labels

    Returns:

    """

    cdef:
        Py_ssize_t m
        cpp_vector[cpp_map[cpp_string, double]] pr
        cpp_map[cpp_string, double] ps
        cpp_vector[double] v1
        cpp_vector[cpp_vector[double]] v2
        cpp_vector[cpp_vector[cpp_vector[double]]] v3

    with nogil:

        for pr in pred:

            for ps in pr:

                v1 = _push_classes(v1, ps, labels, n_classes)
                v2.push_back(v1)
                v1.clear()

            v3.push_back(v2)

            v2.clear()

    return np.array(v3, dtype='float64').transpose(1, 2, 0).reshape(ntime,
                                                                    n_classes,
                                                                    nrows,
                                                                    ncols)


def time_to_crffeas(double[:, :, ::1] data,
                    cpp_string sensor,
                    unsigned int ntime,
                    unsigned int nrows,
                    unsigned int ncols):

    """
    Converts a time-shaped array to CRF features

    Returns:
        ``list`` of feature dictionaries
    """

    cdef:
        Py_ssize_t i, j, v
        double[:, ::1] tdata
        double[::1] tsample
        cpp_vector[cpp_map[cpp_string, double]] samples
        cpp_vector[cpp_vector[cpp_map[cpp_string, double]]] samples_full
        unsigned int red_idx
        unsigned int nir_idx
        double ndvi
        cpp_string ndvi_string = <cpp_string>'ndvi'.encode('utf-8')
        cpp_vector[cpp_string] string_ints
        cpp_map[cpp_string, cpp_map[cpp_string, int]] sensor_bands
        cpp_map[cpp_string, int] l7_like
        cpp_map[cpp_string, int] l8
        cpp_map[cpp_string, int] s210
        cpp_map[cpp_string, int] s2

    for v in range(1, ncols+1):
        string_ints.push_back(<cpp_string>str(v).encode('utf-8'))

    l7_like[b'blue'] = 0
    l7_like[b'green'] = 1
    l7_like[b'red'] = 2
    l7_like[b'nir'] = 3
    l7_like[b'swir1'] = 4
    l7_like[b'swir2'] = 5

    l8[b'coastal'] = 0
    l8[b'blue'] = 1
    l8[b'green'] = 2
    l8[b'red'] = 3
    l8[b'nir'] = 4
    l8[b'swir1'] = 5
    l8[b'swir2'] = 6

    s210[b'blue'] = 0
    s210[b'green'] = 1
    s210[b'red'] = 2
    s210[b'nir'] = 3

    s2[b'blue'] = 0
    s2[b'green'] = 1
    s2[b'red'] = 2
    s2[b'nir1'] = 3
    s2[b'nir2'] = 4
    s2[b'nir3'] = 5
    s2[b'nir'] = 6
    s2[b'rededge'] = 7
    s2[b'swir1'] = 8
    s2[b'swir2'] = 8

    sensor_bands[b'l7'] = l7_like
    sensor_bands[b'l8'] = l8
    sensor_bands[b's210'] = s210
    sensor_bands[b's2'] = s2
    sensor_bands[b's2l7'] = l7_like

    if sensor != b'pan':

        red_idx = sensor_bands[sensor]['red']
        nir_idx = sensor_bands[sensor]['nir']

    with nogil:

        for i in range(0, nrows*ncols):

            for j in range(0, ntime):

                tdata = data[j]
                tsample = tdata[i, :]

                if sensor == b'pan':

                    samples.push_back(_sample_to_dict_pan(tsample,
                                                          string_ints))

                else:

                    ndvi = _ndvi(tsample[red_idx]*0.0001, tsample[nir_idx]*0.0001)

                    samples.push_back(_sample_to_dict(tsample,
                                                      string_ints,
                                                      ndvi,
                                                      ndvi_string))

            samples_full.push_back(samples)

            samples.clear()

    return samples_full
