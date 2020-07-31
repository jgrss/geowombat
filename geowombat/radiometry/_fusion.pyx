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
   double exp(double val) nogil


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


cdef inline double _logistic(double x, double x0, double r) nogil:
    return 1.0 / (1.0 + exp(-r * (x - x0)))


cdef inline double _logistic_scaler(double x, double xmin, double xmax, double x0, double r) nogil:
    return 1.0 / (1.0 + exp(r * ((x - xmin) / (xmax - xmin)) - x0))


cdef void _std(double[:, ::1] data1,
               double[:, ::1] data2,
               double[:, ::1] data3,
               double[:, :, ::1] res_std,
               double[:, ::1] dist_window,
               Py_ssize_t i,
               Py_ssize_t j,
               unsigned int w,
               int hw,
               unsigned int nrows) nogil:

    """
    Calculates the local window standard deviation
    """

    cdef:
        Py_ssize_t i0, j0
        double v1, v2, v3, data_std1, data_std2, data_std3
        double data_mean1 = 0.0
        double data_mean2 = 0.0
        double data_mean3 = 0.0
        double data_var1 = 0.0
        double data_var2 = 0.0
        double data_var3 = 0.0
        unsigned int data_n1 = 0
        unsigned int data_n2 = 0
        unsigned int data_n3 = 0

    for i0 in range(0, w):
        for j0 in range(0, w):

            if dist_window[i0, j0] != -1:

                v1 = data1[i+i0, j+j0]
                v2 = data2[i+i0, j+j0]
                v3 = data3[i+i0, j+j0]

                if v1 != 0:
                    data_mean1 += v1
                    data_n1 += 1

                if v2 != 0:
                    data_mean2 += v2
                    data_n2 += 1

                if v3 != 0:
                    data_mean3 += v3
                    data_n3 += 1

    data_mean1 /= <double>data_n1
    data_mean2 /= <double>data_n2
    data_mean3 /= <double>data_n3

    for i0 in range(0, w):
        for j0 in range(0, w):

            if dist_window[i0, j0] != -1:

                v1 = data1[i+i0, j+j0]
                v2 = data2[i+i0, j+j0]
                v3 = data3[i+i0, j+j0]

                if v1 != 0:
                    data_var1 += (v1 - data_mean1)**2

                if v2 != 0:
                    data_var2 += (v2 - data_mean2)**2

                if v3 != 0:
                    data_var3 += (v3 - data_mean3)**2

    data_var1 /= <double>data_n1
    data_var2 /= <double>data_n2
    data_var3 /= <double>data_n3

    data_std1 = sqrt(data_var1)
    data_std2 = sqrt(data_var2)
    data_std3 = sqrt(data_var3)

    res_std[0, i+hw, j+hw] = data_std1
    res_std[1, i+hw, j+hw] = data_std2
    res_std[2, i+hw, j+hw] = data_std3

    # Minimum
    if data_std1 < res_std[0, nrows, 0]:
        res_std[0, nrows, 0] = data_std1

    if data_std2 < res_std[1, nrows, 0]:
        res_std[1, nrows, 0] = data_std2

    if data_std3 < res_std[2, nrows, 0]:
        res_std[2, nrows, 0] = data_std3

    # Maximum
    if data_std1 > res_std[0, nrows, 1]:
        res_std[0, nrows, 1] = data_std1

    if data_std2 > res_std[1, nrows, 1]:
        res_std[1, nrows, 1] = data_std2

    if data_std3 > res_std[2, nrows, 1]:
        res_std[2, nrows, 1] = data_std3


cdef void _fit_starfm(double[:, ::1] band_weights,
               double[:, ::1] hres_k,
               double[:, ::1] mres_k,
               double[:, ::1] mres_0,
               double[:, ::1] dist_window,
               Py_ssize_t i,
               Py_ssize_t j,
               int w,
               int hw,
               double sensor_uncert,
               double temp_uncert,
               double param_a,
               double param_b) nogil:

    cdef:
        Py_ssize_t m1, n1, m2, n2
        double sp_dist, tp_dist, sw_dist, weight, comb_score
        double comb_sum = 0.0
        double max_s, max_t

    max_s = fabs(hres_k[i+hw, j+hw] - mres_k[i+hw, j+hw]) + sensor_uncert + 1.0
    max_t = fabs(mres_k[i+hw, j+hw] - mres_0[i+hw, j+hw]) + temp_uncert + 1.0

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


cdef double _transform_starfm(double[:, ::1] band_weights,
                       double[:, ::1] hres_k,
                       double[:, ::1] mres_k,
                       double[:, ::1] mres_0,
                       double[:, ::1] dist_window,
                       Py_ssize_t i,
                       Py_ssize_t j,
                       int w,
                       int hw,
                       double sensor_uncert,
                       double temp_uncert,
                       double param_a,
                       double param_b) nogil:

    cdef:
        Py_ssize_t m2, n2
        double sp_dist, tp_dist, weight
        double pred = 0.0
        double max_s, max_t

    max_s = fabs(hres_k[i+hw, j+hw] - mres_k[i+hw, j+hw]) + sensor_uncert + 1.0
    max_t = fabs(mres_k[i+hw, j+hw] - mres_0[i+hw, j+hw]) + temp_uncert + 1.0

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


cdef double _fit_transform_starfm(double[:, ::1] hres_k,
                           double[:, ::1] mres_k,
                           double[:, ::1] mres_0,
                           double[:, ::1] hres_k_sim,
                           double[:, ::1] mres_k_sim,
                           double[:, ::1] dist_window,
                           double[:, :, ::1] std_stack,
                           double[::1] abs_dev_k,
                           double[::1] abs_dev_0,
                           Py_ssize_t i,
                           Py_ssize_t j,
                           unsigned int nrows,
                           unsigned int w,
                           int hw,
                           double sensor_uncert,
                           double temp_uncert,
                           double param_a,
                           double param_b,
                           int param_n) nogil:

    cdef:
        Py_ssize_t m1, n1, m2, n2
        double sp_dist, tp_dist, sw_dist
        double weight, score, fweight
        double weight_sum = 0.0
        double pred = 0.0
        double hres_k_std_dist, mres_k_std_dist, mres_0_std_dist
        # double hres_sim

        double hres_k_sim_thresh = 2.0 * std_stack[0, i+hw, j+hw] / <double>param_n

        # Maximum possible weight
        double max_weight = 6.0

    # if mres_0[i+hw, j+hw] == mres_k[i+hw, j+hw]:
    #     return hres_k[i+hw, j+hw]
    # elif hres_k[i+hw, j+hw] == mres_k[i+hw, j+hw]:
    #     return mres_0[i+hw, j+hw]
    # else:

    for m1 in range(0, w):
        for n1 in range(0, w):

            # Stay within the circle
            if dist_window[m1, n1] != -1:

                if (hres_k[i+m1, j+n1] > 0) and (mres_k[i+m1, j+n1] > 0) and (mres_0[i+m1, j+n1] > 0):

                    # Spectral distance
                    # sp_dist = _logistic_scaler(fabs(hres_k[i+m1, j+n1] - mres_0[i+m1, j+n1]), 0.0, 1.0, 5.0, 15.0)
                    sp_dist = fabs(hres_k[i+m1, j+n1] - mres_0[i+m1, j+n1])

                    # max_sp_dist = sp_diff + sensor_uncert + 1.0
                    # sp_dist_filter = 1 if sp_dist > 1.0 / max_sp_dist else 0

                    # Temporal distance
                    # tp_dist = _logistic_scaler(fabs(mres_k[i+m1, j+n1] - mres_0[i+m1, j+n1]), 0.0, 1.0, 5.0, 15.0)
                    tp_dist = fabs(mres_k[i+m1, j+n1] - mres_0[i+m1, j+n1])

                    # max_tp_dist = tp_diff + temp_uncert + 1.0
                    # tp_dist_filter = 1 if tp_dist > 1.0 / max_tp_dist else 0

                    # Similarity test (|current value - center value|)
                    # hres_sim = fabs(hres_k_sim[i+m1, j+n1] - hres_k_sim[i+hw, j+hw])

                    # Weight for local heterogeneity
                    hres_k_std_dist = _logistic_scaler(std_stack[0, i+m1, j+n1], std_stack[0, nrows, 0], std_stack[0, nrows, 1], 7.5, 25.0)
                    mres_k_std_dist = _logistic_scaler(std_stack[1, i+m1, j+n1], std_stack[1, nrows, 0], std_stack[1, nrows, 1], 7.5, 25.0)
                    mres_0_std_dist = _logistic_scaler(std_stack[2, i+m1, j+n1], std_stack[2, nrows, 0], std_stack[2, nrows, 1], 7.5, 25.0)

                    # if hres_sim <= hres_k_sim_thresh:

                    # Spatial distance
                    sw_dist = 1.0 + dist_window[m1, n1] / param_a
                    # sw_dist = _logistic_scaler(dist_window[m1, n1], 0.0, <double>hw, 7.5, 5.0)
                    # sw_dist = 1.0 / log(dist_window[m1, n1] + 1.0)

                    # High value = bad
                    # weight = (sp_dist + tp_dist + sw_dist + hres_k_std_dist*0.5 + mres_k_std_dist*0.5 + mres_0_std_dist*0.5) / 4.5
                    weight = sp_dist * tp_dist * sw_dist #* hres_k_std_dist * mres_k_std_dist * mres_0_std_dist

                    # Combine weights
                    # TODO: add log option
                    # weight_sum += (1.0 / (log(sp_dist*param_b) * log(tp_dist*param_b) * sw_dist))
                    # if (sp_dist == 0) or (tp_dist == 0):
                    #     weight_sum += 10000.0
                    # else:
                    weight_sum += (1.0 / weight)

    if weight_sum > 0:

        for m2 in range(0, w):
            for n2 in range(0, w):

                if dist_window[m2, n2] != -1:

                    if (hres_k[i+m2, j+n2] > 0) and (mres_k[i+m2, j+n2] > 0) and (mres_0[i+m2, j+n2] > 0):

                        # Spectral distance
                        # sp_dist = _logistic_scaler(fabs(hres_k[i+m2, j+n2] - mres_0[i+m2, j+n2]), 0.0, 1.0, 5.0, 15.0)
                        sp_dist = fabs(hres_k[i+m2, j+n2] - mres_0[i+m2, j+n2])

                        # max_sp_dist = sp_diff + sensor_uncert + 1.0
                        # sp_dist_filter = 1 if sp_dist > 1.0 / max_sp_dist else 0

                        # Temporal distance
                        # tp_dist = _logistic_scaler(fabs(mres_k[i+m2, j+n2] - mres_0[i+m2, j+n2]), 0.0, 1.0, 5.0, 15.0)
                        tp_dist = fabs(mres_k[i+m2, j+n2] - mres_0[i+m2, j+n2])

                        # max_tp_dist = tp_diff + temp_uncert + 1.0
                        # tp_dist_filter = 1 if tp_dist > 1.0 / max_tp_dist else 0

                        # Similarity test (|current value - center value|)
                        # hres_sim = fabs(hres_k_sim[i+m2, j+n2] - hres_k_sim[i+hw, j+hw])

                        # Weight for local heterogeneity
                        hres_k_std_dist = _logistic_scaler(std_stack[0, i+m2, j+n2], std_stack[0, nrows, 0], std_stack[0, nrows, 1], 7.5, 25.0)
                        mres_k_std_dist = _logistic_scaler(std_stack[1, i+m2, j+n2], std_stack[1, nrows, 0], std_stack[1, nrows, 1], 7.5, 25.0)
                        mres_0_std_dist = _logistic_scaler(std_stack[2, i+m2, j+n2], std_stack[2, nrows, 0], std_stack[2, nrows, 1], 7.5, 25.0)

                        # if hres_sim <= hres_k_sim_thresh:

                        # Spatial distance
                        sw_dist = 1.0 + dist_window[m2, n2] / param_a
                        # sw_dist = _logistic_scaler(dist_window[m2, n2], 0.0, <double>hw, 7.5, 5.0)
                        # sw_dist = 1.0 / log(dist_window[m2, n2] + 1.0)

                        # weight = (sp_dist + tp_dist + sw_dist + hres_k_std_dist*0.5 + mres_k_std_dist*0.5 + mres_0_std_dist*0.5) / 4.5
                        weight = sp_dist * tp_dist * sw_dist #* hres_k_std_dist * mres_k_std_dist * mres_0_std_dist

                        # Combine weights
                        # TODO: add log option
                        # comb_score = 1.0 / (log(sp_dist*param_b) * log(tp_dist*param_b) * sw_dist)
                        # if (sp_dist == 0) or (tp_dist == 0):
                        #     score = 10000.0
                        # else:
                        #     score = 1.0 / (1.0 - weight)

                        pred += ((1.0 / weight) * (mres_0[i+m2, j+n2] + hres_k[i+m2, j+n2] - mres_k[i+m2, j+n2]))

        pred /= weight_sum

        return pred


cdef double _fit_transform_force(double[:, ::1] hres_k,
                                 double[:, ::1] mres_0,
                                 double[:, ::1] dist_window,
                                 double[:, :, ::1] std_stack,
                                 Py_ssize_t i,
                                 Py_ssize_t j,
                                 unsigned int nrows,
                                 unsigned int w,
                                 int hw) nogil:

    cdef:
        Py_ssize_t m1, n1, m2, n2
        double sp_dist, hres_k_std_dist, mres_0_std_dist, sw_dist
        double weight
        double weight_sum = 0.0
        double pred = 0.0

    for m1 in range(0, w):
        for n1 in range(0, w):

            # Stay within the circle
            if dist_window[m1, n1] != -1:

                # Score spectral difference [0,1], where 1 is smaller spectral difference
                sp_dist = _logistic_scaler(fabs(hres_k[i+m1, j+n1] - mres_0[i+m1, j+n1]), 0.0, 1.0, 5.0, 15.0)

                # Weight for local heterogeneity
                # L(\std, min, max)
                hres_k_std_dist = _logistic_scaler(std_stack[0, i+m1, j+n1], std_stack[0, nrows, 0], std_stack[0, nrows, 1], 7.5, 25.0)
                mres_0_std_dist = _logistic_scaler(std_stack[2, i+m1, j+n1], std_stack[2, nrows, 0], std_stack[2, nrows, 1], 7.5, 25.0)

                # sw_dist = _logistic_scaler(_logistic(dist_window[m1, n1], 0.5, -10.0), 0.0, 1.0)
                sw_dist = _logistic_scaler(dist_window[m1, n1], 0.0, <double>hw, 5.0, 10.0)

                weight = sp_dist * hres_k_std_dist * mres_0_std_dist * sw_dist

                weight_sum += weight

    if weight_sum > 0:

        for m2 in range(0, w):
            for n2 in range(0, w):

                if dist_window[m2, n2] != -1:

                    # Score spectral difference [0,1], where 1 is smaller spectral difference
                    sp_dist = _logistic_scaler(fabs(hres_k[i+m2, j+n2] - mres_0[i+m2, j+n2]), 0.0, 1.0, 5.0, 15.0)

                    # Weight for local heterogeneity
                    hres_k_std_dist = _logistic_scaler(std_stack[0, i+m2, j+n2], std_stack[0, nrows, 0], std_stack[0, nrows, 1], 7.5, 25.0)
                    mres_0_std_dist = _logistic_scaler(std_stack[2, i+m2, j+n2], std_stack[2, nrows, 0], std_stack[2, nrows, 1], 7.5, 25.0)

                    # sw_dist = _logistic_scaler(_logistic(dist_window[m2, n2], 0.5, -10.0), 0.0, 1.0)
                    sw_dist = _logistic_scaler(dist_window[m2, n2], 0.0, <double>hw, 5.0, 10.0)

                    weight = sp_dist * hres_k_std_dist * mres_0_std_dist * sw_dist

                    pred += (mres_0[i+m2, j+n2] * weight)

        return pred / weight_sum

    else:
        return 0.0


cdef void _create_dist_window(double[:, ::1] dist_window_,
                              unsigned int window_size,
                              unsigned int hw,
                              bint mask_corners=False,
                              bint norm=False) nogil:

    cdef:
        Py_ssize_t m0, n0

    for m0 in range(0, window_size):
        for n0 in range(0, window_size):
            dist_window_[m0, n0] = sqrt((<double>m0 - <double>hw)**2 + (<double>n0 - <double>hw)**2)

    # Create circular window (-1 are outside the radius)
    for m0 in range(0, window_size):
        for n0 in range(0, window_size):

            if mask_corners and window_size > 5:
                if dist_window_[m0, n0] > hw:
                    dist_window_[m0, n0] = -1

            if norm and dist_window_[m0, n0] != -1:
                dist_window_[m0, n0] /= <double>hw


cdef class ImproPhe(object):

    cdef:
        unsigned int window_size_
        unsigned int n_jobs_

    def __init__(self,
                 unsigned int window_size=25,
                 unsigned int n_jobs=1):

        self.window_size_ = window_size
        self.n_jobs_ = n_jobs

    def fit_transform(self,
                      double[:, ::1] hres_k not None,
                      double[:, ::1] mres_0 not None):

        cdef:

            unsigned int window_size = self.window_size_
            unsigned int n_jobs = self.n_jobs_

            Py_ssize_t f
            int i, j
            unsigned int rows = hres_k.shape[0]
            unsigned int cols = hres_k.shape[1]
            int hw = <int>(window_size / 2.0)
            unsigned int row_dims = rows - window_size
            unsigned int col_dims = cols - window_size
            unsigned int nsamples = <int>(row_dims * col_dims)
            double[:, ::1] dist_window = np.zeros((window_size, window_size), dtype='float64')
            double[:, ::1] output = np.zeros((rows, cols), dtype='float64')

            double[:, :, ::1] std_stack = np.zeros((3, rows+1, cols), dtype='float64')

        # Create the distance window
        with nogil:

            _create_dist_window(dist_window,
                                window_size,
                                hw)

        # Set the minimum
        std_stack[0, rows, 0] = 1e9
        std_stack[1, rows, 0] = 1e9
        std_stack[2, rows, 0] = 1e9

        # Standard deviation
        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                _std(hres_k, mres_0, mres_0, std_stack, dist_window, i, j, window_size, hw, rows)

        # Fusion
        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                output[i+hw, j+hw] = _fit_transform_force(hres_k,
                                                          mres_0,
                                                          dist_window,
                                                          std_stack,
                                                          i,
                                                          j,
                                                          rows,
                                                          window_size,
                                                          hw)

        return np.float64(output)


cdef class StarFM(object):
    
    cdef:
        unsigned int window_size_
        double hres_uncert_
        double mres_uncert_
        double param_a_
        double param_b_
        int param_n_
        unsigned int n_jobs_

    def __init__(self, 
                 unsigned int window_size=25,
                 double hres_uncert=0.003,
                 double mres_uncert=0.003,
                 double param_a=1.0,
                 double param_b=1.0,
                 int param_n=2,
                 unsigned int n_jobs=1):

        self.window_size_ = window_size
        self.hres_uncert_ = hres_uncert
        self.mres_uncert_ = mres_uncert
        self.param_a_ = param_a
        self.param_b_ = param_b
        self.param_n_ = param_n
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
            double sensor_uncert, temp_uncert

        with nogil:
    
            sensor_uncert = sqrt(hres_uncert ** 2 + mres_uncert ** 2)
            temp_uncert = sqrt(2.0 * mres_uncert ** 2)
    
            for m0 in range(0, window_size):
                for n0 in range(0, window_size):
                    dist_window[m0, n0] = sqrt((<double>m0 - <double>hw)**2 + (<double>n0 - <double>hw)**2)

        with nogil, parallel(num_threads=n_jobs):
    
            for f in prange(0, nsamples, schedule='static'):
    
                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)
    
                _fit_starfm(output,
                     hres_k,
                     mres_k,
                     mres_0,
                     dist_window,
                     i,
                     j,
                     window_size,
                     hw,
                     sensor_uncert,
                     temp_uncert,
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
            double sensor_uncert, temp_uncert

        with nogil:

            sensor_uncert = sqrt(hres_uncert ** 2 + mres_uncert ** 2)
            temp_uncert = sqrt(2.0 * mres_uncert ** 2)

            for m0 in range(0, window_size):
                for n0 in range(0, window_size):
                    dist_window[m0, n0] = sqrt((<double>m0 - <double>hw)**2 + (<double>n0 - <double>hw)**2)

        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                output[i+hw, j+hw] = _transform_starfm(band_weights,
                                                hres_k,
                                                mres_k,
                                                mres_0,
                                                dist_window,
                                                i,
                                                j,
                                                window_size,
                                                hw,
                                                sensor_uncert,
                                                temp_uncert,
                                                param_a,
                                                param_b)

        return np.float64(output)

    def fit_transform(self,
                      double[:, ::1] hres_k not None,
                      double[:, ::1] mres_k not None,
                      double[:, ::1] mres_0 not None,
                      double[:, ::1] hres_k_sim not None,
                      double[:, ::1] mres_k_sim not None):

        cdef:

            unsigned int window_size = self.window_size_
            double hres_uncert = self.hres_uncert_
            double mres_uncert = self.mres_uncert_
            double param_a = self.param_a_
            double param_b = self.param_b_
            int param_n = self.param_n_
            unsigned int n_jobs = self.n_jobs_

            Py_ssize_t f
            int i, j
            unsigned int rows = hres_k.shape[0]
            unsigned int cols = hres_k.shape[1]
            int hw = <int>(window_size / 2.0)
            unsigned int row_dims = rows - window_size
            unsigned int col_dims = cols - window_size
            unsigned int nsamples = <int>(row_dims * col_dims)
            double[:, ::1] dist_window = np.zeros((window_size, window_size), dtype='float64')
            double[:, ::1] output = np.zeros((rows, cols), dtype='float64')
            double sensor_uncert, temp_uncert

            double[::1] abs_dev_k = np.zeros(2, dtype='float64')
            double[::1] abs_dev_0 = np.zeros(2, dtype='float64')

            double[:, :, ::1] std_stack = np.zeros((3, rows+1, cols), dtype='float64')

        # Calculate the similarity threshold
        # hres_k_sim_thresh = 2.0 * (np.array(hres_k_sim).std() / param_n)
        # mres_k_sim_thresh = 2.0 * (np.array(mres_k_sim).std() / param_n)

        # Calculate the maximum deviations
        abs_dev_k[0] = np.abs(np.array(hres_k) - np.array(mres_k)).min()
        abs_dev_k[1] = np.abs(np.array(hres_k) - np.array(mres_k)).max()
        abs_dev_0[0] = np.abs(np.array(mres_0) - np.array(mres_k)).min()
        abs_dev_0[1] = np.abs(np.array(mres_0) - np.array(mres_k)).max()

        # Create the distance window
        with nogil:

            sensor_uncert = sqrt(hres_uncert**2 + mres_uncert**2)
            temp_uncert = sqrt(2.0 * mres_uncert**2)

            _create_dist_window(dist_window,
                                window_size,
                                hw)

        # Set the minimum
        std_stack[0, rows, 0] = 1e9
        std_stack[1, rows, 0] = 1e9
        std_stack[2, rows, 0] = 1e9

        # Standard deviation
        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                _std(hres_k, mres_k, mres_0, std_stack, dist_window, i, j, window_size, hw, rows)

        # StarFM
        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static'):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                output[i+hw, j+hw] = _fit_transform_starfm(hres_k,
                                                    mres_k,
                                                    mres_0,
                                                    hres_k_sim,
                                                    mres_k_sim,
                                                    dist_window,
                                                    std_stack,
                                                    abs_dev_k,
                                                    abs_dev_0,
                                                    i,
                                                    j,
                                                    rows,
                                                    window_size,
                                                    hw,
                                                    sensor_uncert,
                                                    temp_uncert,
                                                    param_a,
                                                    param_b,
                                                    param_n)

        return np.float64(output)
