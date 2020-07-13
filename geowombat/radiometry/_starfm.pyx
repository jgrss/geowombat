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
   double log(double val) nogil


cdef extern from 'math.h':
   double floor(double val) nogil


cdef extern from 'math.h':
   double sqrt(double val) nogil


cdef extern from 'math.h':
   double fabs(double val) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(double val) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(double val) nogil


cdef inline int _get_rindex(int col_dims, Py_ssize_t index) nogil:
    return <int>floor(<double>index / <double>col_dims)


cdef inline int _get_cindex(int col_dims, Py_ssize_t index, int row_index) nogil:
    return <int>(index - <double>col_dims * row_index)


cdef void _fit(double[:, ::1] band_weights,
               double[:, ::1] hres_k,
               double[:, ::1] mres_k,
               double[:, ::1] mres_0,
               double[:, ::1] dist_window,
               Py_ssize_t i,
               Py_ssize_t j,
               int w,
               int hw,
               double max_s,
               double max_t,
               double param_a,
               double param_b) nogil:

    cdef:
        Py_ssize_t m1, n1, m2, n2
        double sp_dist, tp_dist, sw_dist, weight, comb_score
        double comb_sum = 0.0

    for m1 in range(0, w):
        for n1 in range(0, w):

            if (hres_k[i+m1, j+n1] > 0) and (mres_k[i+m1, j+n1] > 0) and (mres_0[i+m1, j+n1] > 0):

                # Spectral distance
                sp_dist = fabs(hres_k[i+m1, j+n1] - mres_k[i+m1, j+n1]) + 1.0

                # Temporal distance
                tp_dist = fabs(mres_0[i+m1, j+n1] - mres_k[i+m1, j+n1]) + 1.0

                if (sp_dist < max_s) and (tp_dist < max_t):

                    # Spatial distance
                    sw_dist = 1.0 + dist_window[m1, n1] / param_a
                    # sw_dist = dist_window[m1, n1]

                    # Combine weights
                    # TODO: add log option
                    # comb_sum += (1.0 / (log(sp_dist*param_b) * log(tp_dist*param_b) * sw_dist))
                    comb_sum += (1.0 / (sp_dist * tp_dist * sw_dist))

    if comb_sum > 0:

        for m2 in range(0, w):
            for n2 in range(0, w):

                if (hres_k[i+m2, j+n2] > 0) and (mres_k[i+m2, j+n2] > 0) and (mres_0[i+m2, j+n2] > 0):

                    # Spectral distance
                    sp_dist = fabs(hres_k[i+m2, j+n2] - mres_k[i+m2, j+n2]) + 1.0

                    # Temporal distance
                    tp_dist = fabs(mres_0[i+m2, j+n2] - mres_k[i+m2, j+n2]) + 1.0

                    if (sp_dist < max_s) and (tp_dist < max_t):

                        # Spatial distance
                        sw_dist = 1.0 + dist_window[m2, n2] / param_a
                        # sw_dist = dist_window[m1, n1]

                        # Combine weights
                        # TODO: add log option
                        # comb_score = 1.0 / (log(sp_dist*param_b) * log(tp_dist*param_b) * sw_dist)
                        comb_score = 1.0 / (sp_dist * tp_dist * sw_dist)

                        weight = comb_score / comb_sum

                        band_weights[<int>(i*w)+m2, <int>(j*w)+n2] = weight


cdef double _transform(double[:, ::1] band_weights,
                       double[:, ::1] hres_k,
                       double[:, ::1] mres_k,
                       double[:, ::1] mres_0,
                       double[:, ::1] dist_window,
                       Py_ssize_t i,
                       Py_ssize_t j,
                       int w,
                       int hw,
                       double max_s,
                       double max_t,
                       double param_a,
                       double param_b) nogil:

    cdef:
        Py_ssize_t m2, n2
        double sp_dist, tp_dist, weight
        double pred = 0.0

    for m2 in range(0, w):
        for n2 in range(0, w):

            if (hres_k[i+m2, j+n2] > 0) and (mres_k[i+m2, j+n2] > 0) and (mres_0[i+m2, j+n2] > 0):

                # Spectral distance
                sp_dist = fabs(hres_k[i+m2, j+n2] - mres_k[i+m2, j+n2]) + 1.0

                # Temporal distance
                tp_dist = fabs(mres_0[i+m2, j+n2] - mres_k[i+m2, j+n2]) + 1.0

                if (sp_dist < max_s) and (tp_dist < max_t):

                    weight = band_weights[<int>(i*w)+m2, <int>(j*w)+n2]

                    pred += (weight * (mres_0[i+m2, j+n2] + hres_k[i+m2, j+n2] - mres_k[i+m2, j+n2]))

    return pred


cdef double _fit_transform(double[:, ::1] hres_k,
                           double[:, ::1] mres_k,
                           double[:, ::1] mres_0,
                           double[:, ::1] dist_window,
                           Py_ssize_t i,
                           Py_ssize_t j,
                           int w,
                           int hw,
                           double max_s,
                           double max_t,
                           double param_a,
                           double param_b) nogil:

    cdef:
        Py_ssize_t m1, n1, m2, n2
        double sp_dist, tp_dist, sw_dist, weight, comb_score
        double comb_sum = 0.0
        double pred = 0.0

    for m1 in range(0, w):
        for n1 in range(0, w):

            if (hres_k[i+m1, j+n1] > 0) and (mres_k[i+m1, j+n1] > 0) and (mres_0[i+m1, j+n1] > 0):

                # Spectral distance
                sp_dist = fabs(hres_k[i+m1, j+n1] - mres_k[i+m1, j+n1]) + 1.0

                # Temporal distance
                tp_dist = fabs(mres_0[i+m1, j+n1] - mres_k[i+m1, j+n1]) + 1.0

                if (sp_dist < max_s) and (tp_dist < max_t):

                    # Spatial distance
                    sw_dist = 1.0 + dist_window[m1, n1] / param_a
                    # sw_dist = dist_window[m1, n1]

                    # Combine weights
                    # TODO: add log option
                    # comb_sum += (1.0 / (log(sp_dist*param_b) * log(tp_dist*param_b) * sw_dist))
                    comb_sum += (1.0 / (sp_dist * tp_dist * sw_dist))

    if comb_sum > 0:

        for m2 in range(0, w):
            for n2 in range(0, w):

                if (hres_k[i+m2, j+n2] > 0) and (mres_k[i+m2, j+n2] > 0) and (mres_0[i+m2, j+n2] > 0):

                    # Spectral distance
                    sp_dist = fabs(hres_k[i+m2, j+n2] - mres_k[i+m2, j+n2]) + 1.0

                    # Temporal distance
                    tp_dist = fabs(mres_0[i+m2, j+n2] - mres_k[i+m2, j+n2]) + 1.0

                    if (sp_dist < max_s) and (tp_dist < max_t):

                        # Spatial distance
                        sw_dist = 1.0 + dist_window[m2, n2] / param_a
                        # sw_dist = dist_window[m1, n1]

                        # Combine weights
                        # TODO: add log option
                        # comb_score = 1.0 / (log(sp_dist*param_b) * log(tp_dist*param_b) * sw_dist)
                        comb_score = 1.0 / (sp_dist * tp_dist * sw_dist)

                        weight = comb_score / comb_sum

                        pred += (weight * (mres_0[i+m2, j+n2] + hres_k[i+m2, j+n2] - mres_k[i+m2, j+n2]))

    return pred


cdef class StarFM(object):
    
    cdef:
        unsigned int window_size_
        double hres_uncert_
        double mres_uncert_
        double param_a_
        double param_b_
        unsigned int n_jobs_

    def __init__(self, 
                 unsigned int window_size=25,
                 double hres_uncert=0.003,
                 double mres_uncert=0.003,
                 double param_a=1.0,
                 double param_b=1.0,
                 unsigned int n_jobs=1):

        self.window_size_ = window_size
        self.hres_uncert_ = hres_uncert
        self.mres_uncert_ = mres_uncert
        self.param_a_ = param_a
        self.param_b_ = param_b
        self.n_jobs_ = n_jobs
    
    def fit(self,
            double[:, ::1] hres_k not None,
            double[:, ::1] mres_k not None,
            double[:, ::1] mres_0 not None):
    
        cdef:

            unsigned int window_size = self.window_size_
            double hres_uncert = self.hres_uncert_
            double mres_uncert = self.mres_uncert_
            double param_a = self.param_a_
            double param_b = self.param_b_
            unsigned int n_jobs = self.n_jobs_

            Py_ssize_t f, m0, n0
            int i, j
            unsigned int rows = hres_k.shape[0]
            unsigned int cols = hres_k.shape[1]
            int hw = <int>(window_size / 2.0)
            unsigned int row_dims = rows - window_size
            unsigned int col_dims = cols - window_size
            unsigned int nsamples = <int>(row_dims * col_dims)
            double[:, ::1] dist_window = np.zeros((window_size, window_size), dtype='float64')
            double[:, ::1] output = np.zeros((int(rows*window_size), int(cols*window_size)), dtype='float64')
            double sensor_uncert, temp_uncert, max_s, max_t

        with nogil:
    
            sensor_uncert = sqrt(hres_uncert ** 2 + mres_uncert ** 2)
            temp_uncert = sqrt(2.0 * mres_uncert ** 2)
    
            for m0 in range(0, window_size):
                for n0 in range(0, window_size):
                    dist_window[m0, n0] = sqrt((<double>m0 - <double>hw)**2 + (<double>n0 - <double>hw)**2)
    
        max_s = (np.abs(np.float64(hres_k) - np.float64(mres_k)) + sensor_uncert).max() + 1.0
        max_t = (np.fabs(np.float64(mres_k) - np.float64(mres_0)) + temp_uncert).max() + 1.0
    
        with nogil, parallel(num_threads=n_jobs):
    
            for f in prange(0, nsamples, schedule='static'):
    
                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)
    
                _fit(output,
                     hres_k,
                     mres_k,
                     mres_0,
                     dist_window,
                     i,
                     j,
                     window_size,
                     hw,
                     max_s,
                     max_t,
                     param_a,
                     param_b)
    
        return np.float64(output)

    def transform(self,
                  double[:, ::1] band_weights not None,
                  double[:, ::1] hres_k not None,
                  double[:, ::1] mres_k not None,
                  double[:, ::1] mres_0 not None):

        cdef:

            unsigned int window_size = self.window_size_
            double hres_uncert = self.hres_uncert_
            double mres_uncert = self.mres_uncert_
            double param_a = self.param_a_
            double param_b = self.param_b_
            unsigned int n_jobs = self.n_jobs_

            Py_ssize_t f, m0, n0
            int i, j
            unsigned int rows = hres_k.shape[0]
            unsigned int cols = hres_k.shape[1]
            int hw = <int>(window_size / 2.0)
            unsigned int row_dims = rows - window_size
            unsigned int col_dims = cols - window_size
            unsigned int nsamples = <int>(row_dims * col_dims)
            double[:, ::1] dist_window = np.zeros((window_size, window_size), dtype='float64')
            double[:, ::1] output = np.zeros((rows, cols), dtype='float64')
            double sensor_uncert, temp_uncert, max_s, max_t

        with nogil:

            sensor_uncert = sqrt(hres_uncert ** 2 + mres_uncert ** 2)
            temp_uncert = sqrt(2.0 * mres_uncert ** 2)

            for m0 in range(0, window_size):
                for n0 in range(0, window_size):
                    dist_window[m0, n0] = sqrt((<double>m0 - <double>hw)**2 + (<double>n0 - <double>hw)**2)

        max_s = (np.abs(np.float64(hres_k) - np.float64(mres_k)) + sensor_uncert).max() + 1.0
        max_t = (np.fabs(np.float64(mres_k) - np.float64(mres_0)) + temp_uncert).max() + 1.0

        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                output[i+hw, j+hw] = _transform(band_weights,
                                                hres_k,
                                                mres_k,
                                                mres_0,
                                                dist_window,
                                                i,
                                                j,
                                                window_size,
                                                hw,
                                                max_s,
                                                max_t,
                                                param_a,
                                                param_b)

        return np.float64(output)

    def fit_transform(self,
                      double[:, ::1] hres_k not None,
                      double[:, ::1] mres_k not None,
                      double[:, ::1] mres_0 not None):

        cdef:

            unsigned int window_size = self.window_size_
            double hres_uncert = self.hres_uncert_
            double mres_uncert = self.mres_uncert_
            double param_a = self.param_a_
            double param_b = self.param_b_
            unsigned int n_jobs = self.n_jobs_

            Py_ssize_t f, m0, n0
            int i, j
            unsigned int rows = hres_k.shape[0]
            unsigned int cols = hres_k.shape[1]
            int hw = <int>(window_size / 2.0)
            unsigned int row_dims = rows - window_size
            unsigned int col_dims = cols - window_size
            unsigned int nsamples = <int>(row_dims * col_dims)
            double[:, ::1] dist_window = np.zeros((window_size, window_size), dtype='float64')
            double[:, ::1] output = np.zeros((rows, cols), dtype='float64')
            double sensor_uncert, temp_uncert, max_s, max_t

        with nogil:

            sensor_uncert = sqrt(hres_uncert ** 2 + mres_uncert ** 2)
            temp_uncert = sqrt(2.0 * mres_uncert ** 2)

            for m0 in range(0, window_size):
                for n0 in range(0, window_size):
                    dist_window[m0, n0] = sqrt((<double>m0 - <double>hw)**2 + (<double>n0 - <double>hw)**2)

        max_s = (np.abs(np.float64(hres_k) - np.float64(mres_k)) + sensor_uncert).max() + 1.0
        max_t = (np.fabs(np.float64(mres_k) - np.float64(mres_0)) + temp_uncert).max() + 1.0

        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                output[i+hw, j+hw] = _fit_transform(hres_k,
                                                    mres_k,
                                                    mres_0,
                                                    dist_window,
                                                    i,
                                                    j,
                                                    window_size,
                                                    hw,
                                                    max_s,
                                                    max_t,
                                                    param_a,
                                                    param_b)

        return np.float64(output)
