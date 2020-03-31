# distutils: language=c++
# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

from libc.stdlib cimport malloc, free, qsort


cdef inline int _cmp(const void * pa, const void * pb) nogil:

    cdef:
        double a = (<double *>pa)[0]
        double b = (<double *>pb)[0]

    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


cdef inline double get_perc2d(double[:, ::1] input_view,
                              Py_ssize_t i,
                              Py_ssize_t j,
                              unsigned int w,
                              double nodata,
                              double perc) nogil:

    """
    Reference:
        https://gist.github.com/guillemborrell/2a11a258fbd9cfd8d8e0bc14eac83fa7
    """

    cdef:
        Py_ssize_t a, b, bidx, nvalid
        int perc_index
        double* perc_buffer
        double perc_result

    nvalid = 0
    for a in range(0, w):
        for b in range(0, w):

            if input_view[i+a, j+b] != nodata:
                nvalid += 1

    perc_buffer = <double *>malloc(nvalid * sizeof(double))

    bidx = 0
    for a in range(0, w):
        for b in range(0, w):

            if input_view[i+a, j+b] != nodata:

                perc_buffer[bidx] = input_view[i+a, j+b]
                bidx += 1

    # Sort the buffer
    qsort(perc_buffer, nvalid, sizeof(double), _cmp)

    # Get the percentile
    perc_index = <int>(<double>nvalid * (perc / 100.0))

    if perc_index - 1 < 0:
        perc_result = perc_buffer[0]
    else:
        perc_result = perc_buffer[perc_index-1]

    # Deallocate the buffer
    free(perc_buffer)

    return perc_result
