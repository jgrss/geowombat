# distutils: language = c++
# cython: language_level=3
# cython: cdivision=True
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython
cimport cython

import numpy as np
cimport numpy as np

from libcpp.map cimport map as cpp_map
# from libcpp.vector cimport vector as cpp_vector
from libcpp.string cimport string as cpp_string

ctypedef char* char_ptr


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil
        size_t size() nogil
        T& operator[](size_t) nogil
        void clear() nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(double value) nogil


cdef inline double _nan_check(double value) nogil:
    return 0.0 if npy_isnan(value) else value


cdef inline double _clip_low(double value) nogil:
    return 0.0 if value < 0 else value


cdef inline double _clip_high(double value) nogil:
    return 1.0 if value > 1 else value


cdef inline double _scale_min_max(double xv, double mno, double mxo, double mni, double mxi) nogil:
    return (((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno


cdef inline double _clip(double value) nogil:
    return _clip_low(value) if value < 1 else _clip_high(value)


cdef inline double _evi(double blue, double red, double nir) nogil:
    """Enhanced Vegetation Index"""
    return 2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0))


cdef inline double _evi2(double red, double nir) nogil:
    """Two-band Enhanced Vegetation Index"""
    return 2.5 * ((nir - red) / (nir + 1.0 + (2.4 * red)))


# cdef inline double _bsi(double blue, double red, double nir, double swir2):
#     """Bare Soil Index"""
#     return ((swir2 + red) - (nir - blue)) / ((swir2 + red) + (nir - blue))


cdef inline double _brightness(double green, double red, double nir) nogil:
    """Brightness Index"""
    return (green**2 + red**2 + nir**2)**0.5


cdef inline double _brightness_swir(double green, double red, double nir, double swir1) nogil:
    """Brightness Index"""
    return (green**2 + red**2 + nir**2 + swir1**2)**0.5


# cdef inline double _dbsi(double green, double red, double nir, double swir1):
#     """Dry Bare Soil Index"""
#     return ((swir1 - green) / (swir1 + green)) - _ndvi(red, nir)


cdef inline double _gndvi(double green, double nir) nogil:
    """Green Normalized Difference Vegetation Index"""
    return (nir - green) / (nir + green)


cdef inline double _nbr(double nir, double swir2) nogil:
    """Normalized Burn Ratio"""
    return (nir - swir2) / (nir + swir2)


cdef inline double _ndmi(double nir, double swir1) nogil:
    """Normalized Difference Moisture Index"""
    return (nir - swir1) / (nir + swir1)


cdef inline double _ndvi(double red, double nir) nogil:
    """Normalized Difference Vegetation Index"""
    return (nir - red) / (nir + red)


cdef inline double _wi(double red, double swir1) nogil:
    """Woody Index"""
    return 0.0 if red + swir1 > 0.5 else 1.0 - ((red + swir1) / 0.5)


# cdef unicode _text(s):
#
#     if type(s) is unicode:
#         # Fast path for most common case(s).
#         return <unicode>s
#
#     elif isinstance(s, unicode):
#         # We know from the fast path above that 's' can only be a subtype here.
#         # An evil cast to <unicode> might still work in some(!) cases,
#         # depending on what the further processing does.  To be safe,
#         # we can always create a copy instead.
#         return unicode(s)
#
#     else:
#         raise TypeError('Could not convert to unicode.')


cdef cpp_map[cpp_string, double] _sample_to_dict_pan(double[::1] tsamp,
                                                     vector[cpp_string] string_ints,
                                                     double scale_factor) nogil:

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[cpp_string, double] features_map

    for t in range(0, tsamp_len):
        features_map[string_ints[t]] = tsamp[t] * scale_factor

    return features_map


cdef cpp_map[cpp_string, double] _sample_to_dict_bgrn(double[::1] tsamp,
                                                      vector[cpp_string] string_ints,
                                                      double brightness,
                                                      double evi,
                                                      double evi2,
                                                      double gndvi,
                                                      double ndvi,
                                                      cpp_string brightness_string,
                                                      cpp_string evi_string,
                                                      cpp_string evi2_string,
                                                      cpp_string gndvi_string,
                                                      cpp_string ndvi_string,
                                                      double scale_factor) nogil:

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[cpp_string, double] features_map

    for t in range(0, tsamp_len):
        features_map[string_ints[t]] = tsamp[t] * scale_factor

    features_map[brightness_string] = _nan_check(brightness)
    features_map[evi_string] = _clip(_nan_check(evi))
    features_map[evi2_string] = _clip(_nan_check(evi2))
    features_map[gndvi_string] = _nan_check(gndvi)
    features_map[ndvi_string] = _nan_check(ndvi)

    return features_map


cdef cpp_map[cpp_string, double] _sample_to_dict_s220(double[::1] tsamp,
                                                      vector[cpp_string] string_ints,
                                                      double brightness,
                                                      double nbr,
                                                      double ndmi,
                                                      double ndvi,
                                                      double wi,
                                                      cpp_string brightness_string,
                                                      cpp_string nbr_string,
                                                      cpp_string ndmi_string,
                                                      cpp_string ndvi_string,
                                                      cpp_string wi_string,
                                                      double scale_factor) nogil:

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[cpp_string, double] features_map

    for t in range(0, tsamp_len):
        features_map[string_ints[t]] = tsamp[t] * scale_factor

    features_map[brightness_string] = _nan_check(brightness)
    features_map[nbr_string] = _nan_check(nbr)
    features_map[ndmi_string] = _nan_check(ndmi)
    features_map[ndvi_string] = _nan_check(ndvi)
    features_map[wi_string] = _clip(_nan_check(wi))

    return features_map


cdef cpp_map[cpp_string, double] _sample_to_dict(double[::1] tsamp,
                                                 vector[cpp_string] string_ints,
                                                 double brightness,
                                                 double evi,
                                                 double evi2,
                                                 double gndvi,
                                                 double nbr,
                                                 double ndmi,
                                                 double ndvi,
                                                 double wi,
                                                 cpp_string brightness_string,
                                                 cpp_string evi_string,
                                                 cpp_string evi2_string,
                                                 cpp_string gndvi_string,
                                                 cpp_string nbr_string,
                                                 cpp_string ndmi_string,
                                                 cpp_string ndvi_string,
                                                 cpp_string wi_string,
                                                 double scale_factor,
                                                 bint nodata) nogil:

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t t
        unsigned int tsamp_len = tsamp.shape[0]
        cpp_map[cpp_string, double] features_map

    if nodata:

        for t in range(0, tsamp_len):
            features_map[string_ints[t]] = 0.0

        features_map[brightness_string] = 0.0
        features_map[evi_string] = 0.0
        features_map[evi2_string] = 0.0
        features_map[gndvi_string] = 0.0
        features_map[nbr_string] = 0.0
        features_map[ndmi_string] = 0.0
        features_map[ndvi_string] = 0.0
        features_map[wi_string] = 0.0

    else:

        for t in range(0, tsamp_len):
            features_map[string_ints[t]] = tsamp[t] * scale_factor

        features_map[brightness_string] = _scale_min_max(_nan_check(brightness), 0.01, 1.0, 0.0, 1.0)
        features_map[evi_string] = _scale_min_max(_clip(_nan_check(evi)), 0.01, 1.0, 0.0, 1.0)
        features_map[evi2_string] = _scale_min_max(_clip(_nan_check(evi2)), 0.01, 1.0, 0.0, 1.0)
        features_map[gndvi_string] = _clip(_scale_min_max(_nan_check(gndvi), 0.01, 1.0, -1.0, 1.0))
        features_map[nbr_string] = _clip(_scale_min_max(_nan_check(nbr), 0.01, 1.0, -1.0, 1.0))
        features_map[ndmi_string] = _clip(_scale_min_max(_nan_check(ndmi), 0.01, 1.0, -1.0, 1.0))
        features_map[ndvi_string] = _clip(_scale_min_max(_nan_check(ndvi), 0.01, 1.0, -1.0, 1.0))
        features_map[wi_string] = _scale_min_max(_clip(_nan_check(wi)), 0.01, 1.0, 0.0, 1.0)

    return features_map


cdef cpp_map[cpp_string, double] _samples_to_dict(vector[char*] labels,
                                                  double[::1] tsamp,
                                                  unsigned int nvars) nogil:

    """
    Converts names and a 1d array to a dictionary
    """

    cdef:
        Py_ssize_t v
        cpp_map[cpp_string, double] features_map

    for v in range(0, nvars):
        features_map[labels[v]] = tsamp[v]

    return features_map


cdef vector[double] _push_classes(vector[double] vct,
                                  cpp_map[cpp_string, double] ps,
                                  vector[cpp_string] labels_bytes,
                                  unsigned int n_classes) nogil:

    cdef:
        Py_ssize_t m
        double ps_label_value
        cpp_string m_label

    for m in range(0, n_classes):

        m_label = labels_bytes[m]

        if ps.count(m_label) > 0:
            ps_label_value = ps[m_label]
        else:
            ps_label_value = 0.0

        vct.push_back(ps_label_value)

    return vct


def transform_probas(vector[vector[cpp_map[cpp_string, double]]] pred,
                     vector[cpp_string] labels,
                     unsigned int n_classes,
                     unsigned int ntime,
                     unsigned int nrows,
                     unsigned int ncols):

    """
    Transforms CRF probabilities in dictionary format to probabilities in array format
    """

    cdef:
        Py_ssize_t m
        vector[cpp_map[cpp_string, double]] pr
        cpp_map[cpp_string, double] ps
        vector[double] v1
        vector[vector[double]] v2
        vector[vector[vector[double]]] v3

        Py_ssize_t i, j
        unsigned int niters_i = pred.size()
        unsigned int niters_j

    with nogil:

        #for pr in pred:
        for i in range(0, niters_i):

            pr = pred[i]
            niters_j = pr.size()

            #for ps in pr:
            for j in range(0, niters_j):

                ps = pr[j]

                v1 = _push_classes(v1, ps, labels, n_classes)
                v2.push_back(v1)
                v1.clear()

            v3.push_back(v2)

            v2.clear()

    return np.array(v3, dtype='float64').transpose(1, 2, 0).reshape(ntime,
                                                                    n_classes,
                                                                    nrows,
                                                                    ncols)


def time_to_sensor_feas(double[:, :, ::1] data,
                        cpp_string sensor,
                        unsigned int ntime,
                        unsigned int nrows,
                        unsigned int ncols,
                        double scale_factor=0.0001):

    """
    Converts a time-shaped array to CRF features

    Returns:
        ``list`` of feature dictionaries
    """

    cdef:
        Py_ssize_t i, j, v
        double[:, ::1] tdata
        double[::1] tsample
        vector[cpp_map[cpp_string, double]] samples
        vector[vector[cpp_map[cpp_string, double]]] samples_full

        unsigned int blue_idx, green_idx, red_idx, nir_idx, swir1_idx, swir2_idx, nir1_idx, nir2_idx, nir3_idx, rededge_idx

        double brightness, evi, evi2, gndvi, nbr, ndmi, ndvi, wi

        cpp_string brightness_string = <cpp_string>'bri'.encode('utf-8')
        cpp_string evi_string = <cpp_string>'evi'.encode('utf-8')
        cpp_string evi2_string = <cpp_string>'evi2'.encode('utf-8')
        cpp_string gndvi_string = <cpp_string>'gndvi'.encode('utf-8')
        cpp_string nbr_string = <cpp_string>'nbr'.encode('utf-8')
        cpp_string ndmi_string = <cpp_string>'ndmi'.encode('utf-8')
        cpp_string ndvi_string = <cpp_string>'ndvi'.encode('utf-8')
        cpp_string wi_string = <cpp_string>'wi'.encode('utf-8')

        vector[cpp_string] string_ints
        cpp_map[cpp_string, cpp_map[cpp_string, int]] sensor_bands
        cpp_map[cpp_string, int] l7_like
        cpp_map[cpp_string, int] l8
        cpp_map[cpp_string, int] s210
        cpp_map[cpp_string, int] s220
        cpp_map[cpp_string, int] s2

        bint nodata

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

    s220[b'nir1'] = 0
    s220[b'nir2'] = 1
    s220[b'nir3'] = 2
    s220[b'rededge'] = 3
    s220[b'swir1'] = 4
    s220[b'swir2'] = 5

    s2[b'blue'] = 0
    s2[b'green'] = 1
    s2[b'red'] = 2
    s2[b'nir1'] = 3
    s2[b'nir2'] = 4
    s2[b'nir3'] = 5
    s2[b'nir'] = 6
    s2[b'rededge'] = 7
    s2[b'swir1'] = 8
    s2[b'swir2'] = 9

    sensor_bands[b'l7'] = l7_like
    sensor_bands[b'l8'] = l8
    sensor_bands[b'l5bgrn'] = s210
    sensor_bands[b'l7bgrn'] = s210
    sensor_bands[b'l8bgrn'] = s210
    sensor_bands[b'bgrn'] = s210
    sensor_bands[b'qb'] = s210
    sensor_bands[b'ps'] = s210
    sensor_bands[b's210'] = s210
    sensor_bands[b's220'] = s220
    sensor_bands[b's2'] = s2
    sensor_bands[b's2l7'] = l7_like

    if sensor != b'pan':

        if (sensor == b's210') or (sensor == b'l5bgrn') or (sensor == b'l7bgrn') or (sensor == b'l8bgrn') or (sensor == b'bgrn') or (sensor == b'qb') or (sensor == b'ps'):

            blue_idx = sensor_bands[sensor][b'blue']
            green_idx = sensor_bands[sensor][b'green']
            red_idx = sensor_bands[sensor][b'red']
            nir_idx = sensor_bands[sensor][b'nir']

        elif sensor == b's220':

            nir1_idx = sensor_bands[sensor][b'nir1']
            nir2_idx = sensor_bands[sensor][b'nir2']
            nir3_idx = sensor_bands[sensor][b'nir3']
            rededge_idx = sensor_bands[sensor][b'rededge']
            swir1_idx = sensor_bands[sensor][b'swir1']
            swir2_idx = sensor_bands[sensor][b'swir2']

        else:

            blue_idx = sensor_bands[sensor][b'blue']
            green_idx = sensor_bands[sensor][b'green']
            red_idx = sensor_bands[sensor][b'red']
            nir_idx = sensor_bands[sensor][b'nir']
            swir1_idx = sensor_bands[sensor][b'swir1']
            swir2_idx = sensor_bands[sensor][b'swir2']

    with nogil:

        for i in range(0, nrows*ncols):

            for j in range(0, ntime):

                tdata = data[j]
                tsample = tdata[i, :]

                if sensor == b'pan':

                    samples.push_back(_sample_to_dict_pan(tsample,
                                                          string_ints,
                                                          scale_factor))

                else:

                    if (sensor == b's210') or (sensor == b'l5bgrn') or (sensor == b'l7bgrn') or (sensor == b'l8bgrn') or (sensor == b'bgrn') or (sensor == b'qb') or (sensor == b'ps'):

                        if (tsample[blue_idx] * scale_factor < 0.01) and (tsample[green_idx] * scale_factor < 0.01) and (tsample[red_idx] * scale_factor < 0.01):

                            brightness = 0.0
                            evi = 0.0
                            evi2 = 0.0
                            gndvi = 0.0
                            ndvi = 0.0

                        else:

                            brightness = _brightness(tsample[green_idx]*scale_factor,
                                                     tsample[red_idx]*scale_factor,
                                                     tsample[nir_idx]*scale_factor)

                            evi = _evi(tsample[blue_idx]*scale_factor, tsample[red_idx]*scale_factor, tsample[nir_idx]*scale_factor)
                            evi2 = _evi2(tsample[red_idx]*scale_factor, tsample[nir_idx]*scale_factor)
                            gndvi = _gndvi(tsample[green_idx]*scale_factor, tsample[nir_idx]*scale_factor)
                            ndvi = _ndvi(tsample[red_idx]*scale_factor, tsample[nir_idx]*scale_factor)

                        samples.push_back(_sample_to_dict_bgrn(tsample,
                                                               string_ints,
                                                               brightness,
                                                               evi,
                                                               evi2,
                                                               gndvi,
                                                               ndvi,
                                                               brightness_string,
                                                               evi_string,
                                                               evi2_string,
                                                               gndvi_string,
                                                               ndvi_string,
                                                               scale_factor))

                    elif sensor == b's220':

                        brightness = _brightness(tsample[nir1_idx]*scale_factor,
                                                 tsample[rededge_idx]*scale_factor,
                                                 tsample[swir1_idx]*scale_factor)
                        nbr = _nbr(tsample[rededge_idx]*scale_factor, tsample[swir2_idx]*scale_factor)
                        ndmi = _ndmi(tsample[rededge_idx]*scale_factor, tsample[swir1_idx]*scale_factor)
                        ndvi = _ndvi(tsample[nir1_idx]*scale_factor, tsample[rededge_idx]*scale_factor)
                        wi = _wi(tsample[nir1_idx]*scale_factor, tsample[swir1_idx]*scale_factor)

                        samples.push_back(_sample_to_dict_s220(tsample,
                                                               string_ints,
                                                               brightness,
                                                               nbr,
                                                               ndmi,
                                                               ndvi,
                                                               wi,
                                                               brightness_string,
                                                               nbr_string,
                                                               ndmi_string,
                                                               ndvi_string,
                                                               wi_string,
                                                               scale_factor))

                    else:

                        if (tsample[blue_idx] * scale_factor < 0.01) and (tsample[green_idx] * scale_factor < 0.01) and (tsample[red_idx] * scale_factor < 0.01):

                            brightness = 0.0
                            evi = 0.0
                            evi2 = 0.0
                            gndvi = 0.0
                            nbr = 0.0
                            ndmi = 0.0
                            ndvi = 0.0
                            wi = 0.0

                            nodata = True

                        else:

                            brightness = _brightness_swir(tsample[green_idx] * scale_factor,
                                                          tsample[red_idx] * scale_factor,
                                                          tsample[nir_idx] * scale_factor,
                                                          tsample[swir1_idx] * scale_factor)

                            evi = _evi(tsample[blue_idx] * scale_factor,
                                       tsample[red_idx] * scale_factor,
                                       tsample[nir_idx] * scale_factor)

                            evi2 = _evi2(tsample[red_idx] * scale_factor, tsample[nir_idx] * scale_factor)
                            gndvi = _gndvi(tsample[green_idx] * scale_factor, tsample[nir_idx] * scale_factor)
                            nbr = _nbr(tsample[nir_idx] * scale_factor, tsample[swir2_idx] * scale_factor)
                            ndmi = _ndmi(tsample[nir_idx] * scale_factor, tsample[swir1_idx] * scale_factor)
                            ndvi = _ndvi(tsample[red_idx] * scale_factor, tsample[nir_idx] * scale_factor)
                            wi = _wi(tsample[red_idx] * scale_factor, tsample[swir1_idx] * scale_factor)

                            nodata = False

                        samples.push_back(_sample_to_dict(tsample,
                                                          string_ints,
                                                          brightness,
                                                          evi,
                                                          evi2,
                                                          gndvi,
                                                          nbr,
                                                          ndmi,
                                                          ndvi,
                                                          wi,
                                                          brightness_string,
                                                          evi_string,
                                                          evi2_string,
                                                          gndvi_string,
                                                          nbr_string,
                                                          ndmi_string,
                                                          ndvi_string,
                                                          wi_string,
                                                          scale_factor,
                                                          nodata))

            samples_full.push_back(samples)
            samples.clear()

    return samples_full


def time_to_feas(double[:, :, ::1] data,
                 vector[char*] labels):

    """
    Transforms time-formatted variables to CRF-formatted features

    Args:
        data (3d array): The data to transform. The shape should be [time x
        samples x predictors].
        labels (vector): The labels that correspond to each predictor.

    Returns:
        ``list``
    """

    cdef:
        Py_ssize_t s, t
        unsigned int ntime = data.shape[0]
        unsigned int nsamples = data.shape[1]
        unsigned int nvars = data.shape[2]
        vector[cpp_map[cpp_string, double]] samples
        vector[vector[cpp_map[cpp_string, double]]] samples_full

    with nogil:

        for s in range(0, nsamples):

            for t in range(0, ntime):
                samples.push_back(_samples_to_dict(labels, data[t, s, :], nvars))

            samples_full.push_back(samples)
            samples.clear()

    return samples_full


def labels_to_values(vector[vector[char_ptr]] labels,
                     cpp_map[cpp_string, int] label_mappings,
                     unsigned int nsamples,
                     unsigned int ntime):

    """
    Transforms CRF str/byte labels to values

    Args:
        labels (list)
        label_mappings (dict)
        nsamples (int)
        ntime (int)

    Returns:
        ``list``
    """

    cdef:
        Py_ssize_t s, t
        vector[char*] sample
        vector[int] out_sample
        vector[vector[int]] out_full

    with nogil:

        for s in range(0, nsamples):

            sample = labels[s]

            for t in range(0, ntime):
                out_sample.push_back(label_mappings[sample[t]])

            out_full.push_back(out_sample)
            out_sample.clear()

    return out_full
