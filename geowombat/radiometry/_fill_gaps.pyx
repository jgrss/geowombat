# distutils: language = c++
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

# from libc.stdlib cimport malloc, free

from cython.parallel import prange
from cython.parallel import parallel

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil
        size_t size() nogil
        T& operator[](size_t) nogil
        void clear() nogil


cdef extern from 'stdlib.h' nogil:
    double fabs(double val)


cdef extern from 'math.h':
   double sqrt(double val) nogil


cdef inline double _edist(double xloc, double yloc, double hw) nogil:
    return (xloc - hw)**2 + (yloc - hw)**2


cdef inline double _pow(double val) nogil:
    return val*val


cdef double _apply(double xavg,
                   double yavg,
                   double xstd,
                   double ystd,
                   double gain,
                   double bias,
                   double center_avg,
                   unsigned int count) nogil:

    """
    Applies the least squares solution to estimate the missing value
    """

    cdef:
        double estimate = gain * center_avg + bias

    if estimate > 1:

        gain = ystd / xstd
        bias = yavg - gain * xavg

        estimate = gain * center_avg + bias

    if estimate > 1:
        return 1.0
    elif estimate < 0:
        return 0
    else:
        return estimate


cdef void _calc_wlr(vector[double] xdata,
                    vector[double] ydata,
                    double xavg,
                    double yavg,
                    vector[double] weights,
                    double wsum,
                    unsigned int count,
                    double[::1] results) nogil:

    """
    Calculates the weighted least squares slope and intercept
    """

    cdef:
        Py_ssize_t s
        double xdev, ydev
        # double *results = <double *>malloc(4 * sizeof(double))
        double wxvar = 0.0   # x weighted variance
        double xvar = 0.0   # x variance
        double yvar = 0.0   # y variance
        double cvar = 0.0   # covariance
        double slope, intercept
        double w

    for s in range(0, count):

        w = weights[s] / wsum

        xdev = xdata[s] - xavg
        ydev = ydata[s] - yavg

        xvar += _pow(xdev)
        yvar += _pow(ydev)
        wxvar += _pow(xdev) * w

        cvar += xdev * ydev * w

    slope = cvar / wxvar
    intercept = yavg - slope * xavg

    # Standard deviation
    results[0] = sqrt(xvar / <double>count)
    results[1] = sqrt(yvar / <double>count)

    # beta (gain)
    results[2] = slope

    # alpha (bias)
    results[3] = intercept


cdef double _get_center_mean(double[:, :, :, ::] indata,
                             Py_ssize_t b,
                             Py_ssize_t i,
                             Py_ssize_t j,
                             unsigned int dims,
                             Py_ssize_t ci,
                             unsigned int hw,
                             double nodata) nogil:

    """
    Gets the center average of the reference data
    """

    cdef:
        Py_ssize_t m, n, d
        unsigned int offset = hw - <int>(ci / 2.0)
        double center_avg = 0.0
        double yvalue
        Py_ssize_t count = 0

    for m in range(0, ci):
        for n in range(0, ci):
            for d in range(1, dims):

                yvalue = indata[d, b, i+m+offset, j+n+offset]

                if yvalue != nodata:

                    center_avg += yvalue
                    count += 1

        if center_avg > 0:
            break

    return center_avg / <double>count


cdef double _estimate_gap(double[:, :, :, ::1] indata,
                          Py_ssize_t b,
                          Py_ssize_t i,
                          Py_ssize_t j,
                          unsigned int dims,
                          Py_ssize_t wi,
                          unsigned int hw,
                          double nodata,
                          unsigned int min_thresh,
                          double center_avg,
                          double[::1] stdv,
                          double[::1] results_zeros) nogil:

    cdef:
        Py_ssize_t m, n, d
        unsigned int offset = hw - <int>(wi / 2.0)
        double xvalue, yvalue
        vector[double] xdata, ydata
        Py_ssize_t count = 0

        double xavg = 0.0
        double yavg = 0.0
        # double *stdv = <double *>malloc(4 * sizeof(double))
        double estimate

        vector[double] weights
        double w
        double wsum = 0.0
        double alpha = 0.0001

    stdv[...] = results_zeros

    # Iterate over the window
    for m in range(0, wi):
        for n in range(0, wi):

            yvalue = indata[0, b, i+m+offset, j+n+offset]

            # Iterate over each reference file to fill the window
            for d in range(1, dims):

                xvalue = indata[d, b, i+m+offset, j+n+offset]

                if (xvalue != nodata) and (yvalue != nodata):

                    w = fabs(yvalue - xvalue + alpha) * _edist(<double>n, <double>m, <double>hw)

                    weights.push_back(w)

                    wsum += w

                    xdata.push_back(xvalue)
                    ydata.push_back(yvalue)

                    xavg += xvalue
                    yavg += yvalue

                    count += 1

    if count < min_thresh:
        return -999.0
    else:

        # Window average
        xavg /= <double>count
        yavg /= <double>count

        # Std. dev. of [x, y], slope, intercept
        _calc_wlr(xdata,
                  ydata,
                  xavg,
                  yavg,
                  weights,
                  wsum,
                  count,
                  stdv)

        # Calculate the least squares solution
        estimate = _apply(xavg,
                          yavg,
                          stdv[0],
                          stdv[1],
                          stdv[2],
                          stdv[3],
                          center_avg,
                          count)

        # free(stdv)

        return estimate


cdef double[:, :, ::1] _fill_gaps(double[:, :, :, ::1] indata,
                                  double[:, :, ::1] output,
                                  unsigned int wmax,
                                  unsigned int wmin,
                                  double nodata,
                                  double min_prop,
                                  unsigned int n_jobs):

    cdef:
        Py_ssize_t b, i, j, ci
        unsigned int dims = indata.shape[0]
        unsigned int bands = indata.shape[1]
        unsigned int rows = indata.shape[2]
        unsigned int cols = indata.shape[3]
        unsigned int hw = <int>(wmax / 2.0)
        unsigned int row_dims = rows - <int>(hw*2.0)
        unsigned int col_dims = cols - <int>(hw*2.0)

        double[::1] stdv_results = np.zeros(4, dtype='float64')
        double[::1] results_zeros = np.zeros(4, dtype='float64')

        double tar_center, center_avg
        Py_ssize_t wi
        double fill_value

        unsigned int min_thresh = <int>(min_prop * <double>(wmax*wmax))

    with nogil, parallel(num_threads=n_jobs):

        for i in prange(0, row_dims, schedule='static'):
            for j in range(0, col_dims):
                for b in range(0, bands):

                    # Center target sample
                    tar_center = indata[0, b, i+hw, j+hw]

                    if tar_center != nodata:
                        continue

                    # Get an average of the center value
                    for ci from 3 <= ci < 7 by 2:

                        center_avg = _get_center_mean(indata, b, i, j, dims, ci, hw, nodata)

                        if center_avg > 0:
                            break

                    if center_avg > 0:

                        # Search for data over varying-sized windows
                        for wi from wmin <= wi < wmax by 2:

                            fill_value = _estimate_gap(indata,
                                                       b,
                                                       i,
                                                       j,
                                                       dims,
                                                       wi,
                                                       hw,
                                                       nodata,
                                                       min_thresh,
                                                       center_avg,
                                                       stdv_results,
                                                       results_zeros)

                            if fill_value != -999.0:
                                output[b, i+hw, j+hw] = fill_value

    return output


def fill_gaps(np.ndarray[DTYPE_float64_t, ndim=4] indata not None,
              wmax=25,
              wmin=9,
              nodata=0,
              min_prop=0.15,
              n_jobs=1):

    """
    Fills data gaps using spatial-temporal weighted least squares linear regression

    Args:
        indata (4d array): Layers x bands x rows x columns. The first layer is the target and the remaining layers
            are the references. The reference layers should be sorted from the date closest to the target to the
            date furthest from the target date.
        wmax (Optional[int]): The maximum window size.
        wmin (Optional[int]): The minimum window size.
        nodata (Optional[int]): The 'no data' value to fill.
        min_prop (Optional[float]): The minimum required proportion of ``wmax`` x ``wmax`` in order to fill.
        n_jobs (Optional[int]): The number of bands to process in parallel.

    Returns:
        3d ``numpy.ndarray`` (bands x rows x columns)
    """

    cdef:
        double[:, :, ::1] output = indata[0].copy()

    return np.float64(_fill_gaps(indata, output, wmax, wmin, nodata, min_prop, n_jobs))
