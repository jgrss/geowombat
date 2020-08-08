#ifndef IAS_MATH_H
#define IAS_MATH_H

#include "ias_types.h"
#include "ias_const.h"
#include "ias_structures.h"

/* Define names for the line intersection return values */
#define IAS_LINES_INTERSECT 1
#define IAS_LINES_PARALLEL 0
#define IAS_LINES_DO_NOT_INTERSECT -1

/* Useful macros */
#define IAS_MAX(A,B)    (((A) > (B)) ? (A) : (B))
#define IAS_MIN(A,B)    (((A) < (B)) ? (A) : (B))

typedef enum
{
    IAS_MATH_FFT = -1,    /* Forward code for FFT  */
    IAS_MATH_IFFT = 1     /* Inverse code for FFT  */
} IAS_FFT_DIRECTION_TYPE;


/* Define a structure for holding a line segment */
typedef struct ias_line_segment
{
    double x1;
    double y1;
    double x2;
    double y2;
} IAS_LINE_SEGMENT;

typedef struct ias_math_leap_seconds_data
{
    /* Leap seconds from the CPF Earth Constants group */
    int leap_seconds_count;         /* number leap seconds since 1972 */ 
    int *leap_years;                /* list of years of occurance */
    int *leap_months;               /* list of months of occurance (1-12) */
    int *leap_days;                 /* list of days of occurance (1-31) */
    int *num_leap_seconds;          /* number of leap seconds added at each
                                       time */
} IAS_MATH_LEAP_SECONDS_DATA;

void ias_math_add_seconds_to_year_doy_sod
(
    double seconds,     /* I: Seconds to add to date given */
    double *date        /* IO: Year, DOY, SOD */
);

int ias_math_convert_month_day_to_doy
(
    int month,  /* I: Month */
    int day,    /* I: Day of month */
    int year,   /* I: Year */
    int *doy    /* O: Day of year */
); 

int ias_math_convert_doy_to_month_day
(
    int doy,    /* I: Day of Year */  
    int year,   /* I: Year  */
    int *mon,   /* O: Month */
    int *day    /* O: Day of month */
);

int ias_math_matrix_QRfactorization
(
    double *A,     /* I/O: Matrix A (stored by columns)      */
    int m,         /* I: Number of rows in matrix A         */
    int n,         /* I: Number of columns in matrix A      */
    double *v,     /* O: Vector v (work vector)             */
    int flag       /* I: If true, order matrix A by columns */
);

int ias_math_get_time_difference
(
    const double *epoch_1, /* I: epoch year, DOY, SOD */
    const double *epoch_2, /* I: epoch year, DOY, SOD */
    double *seconds        /* O: the difference in seconds. */
);

int ias_math_matrix_QRsolve
(
    const double *A, /* I: Matrix A (stored by columns)*/
    int  m,          /* I: Number of rows in matrix A */
    int  n,          /* I: Number of columns in matrix A */
    const double *v, /* I: Vector v */
    double *b,       /* O: Vector b */
    int  iflag       /* I: flag (see comments above) */
);

void ias_math_compute_3dvec_cross
(
    const IAS_VECTOR *vec1,     /* I: Input vector number one       */
    const IAS_VECTOR *vec2,     /* I: Input vector number two       */
    IAS_VECTOR *vec3            /* O: Output vector (cross product) */
);

double ias_math_compute_3dvec_dot
(
    const IAS_VECTOR *vec1,     /* I: Vector one to be multiplied  */
    const IAS_VECTOR *vec2      /* I: Vector two to be multiplied  */
);

void ias_math_eval_poly
(
    int degree,      /* I: Degree of polynomial                      */
    const double *a, /* I: Array of polynomial coefficients          */
    double x,        /* I: X coordinates to be evaluated             */
    double y,        /* I: Y coordinates to be evaluated             */
    double *val      /* O: Value of the polynomial at the point      */
);

/* evalutate the polynomial with an x*y term */
double ias_math_eval_poly_xy
(
    int degree,     /* I: Degree of polynomial */
    const double *a,/* I: Array of polynomial coefficients */
    double x,       /* I: X coordinates to be evaluated */
    double y        /* I: Y coordinates to be evaluated */
);

double ias_math_compute_full_julian_date
( 
   int  yr,                /* I: year */ 
   int  mo,                /* I: month */ 
   int  day,               /* I: day */
   double seconds          /* I: seconds of day */
);

int ias_math_invert_3x3_matrix
(
    double inmatrix[3][3],     /* I Input matrix         */
    double outmatrix[3][3]     /* O Inverted matrix      */
);

int ias_math_solve_linear_equation
( 
    double **A,   /* I/O: double pointer to the n x n coefficient matrix which
                        will be inverted */
    int n,        /* I: the dimension of the matrix */
    double b[]    /* I/O: The n x 1 constant vector, with first m to be
                        solved, where m is the rank of A.  The elements of
                        this vector will be replaced by the solution vector */
);

void ias_math_multiply_3x3_matrix
(
    double matrix1[3][3],       /* I: Input matrix one */
    double matrix2[3][3],       /* I: Input matrix two */
    double outmatrix[3][3]      /* O: Multiplied matrix */
);

double ias_math_compute_vector_length
(
    const IAS_VECTOR *vec       /* I: Vector to find the length of */
);

int ias_math_compute_unit_vector
(
    const IAS_VECTOR *vec,      /* I: Input vector */
    IAS_VECTOR *unit_vector     /* O: Unit vector of the input vector */
);

double ias_math_compute_quaternion_magnitude
(
    const IAS_QUATERNION *quat  /* I: Quaternion to find the magnitude of */
);

void ias_math_multiply_quaternions
(
    const IAS_QUATERNION *q1,       /* I: first quaternion */
    const IAS_QUATERNION *q2,       /* I: second quaternion */
    IAS_QUATERNION *result          /* O: multiplication result */
);

void ias_math_convert_quaternion2rpy
(
     const IAS_QUATERNION *quat,    /* I: quaternion to convert */
     IAS_VECTOR *att                /* O: output roll-pitch-yaw */
);

double ias_math_eval_legendre
(
    double x,                   /* I: point at which to compute value */
    const double *coefficients, /* I: Array of Coefficients */
    int num_coefficients        /* I: Number of Coefficients in array */
);

void ias_math_transpose_3x3_matrix
(
    double A[3][3]        /* I/O: the matrix to be transposed          */
);

void ias_math_transform_3dvec
(
    const IAS_VECTOR *Xold, /* I: vector in old system */
    double Trans[3][3],     /* I:transformation matrix from old to new system*/
    IAS_VECTOR *Xnew        /* O: vector in the new system */
);

int ias_math_compute_mean
(
    const double *sample,       /* I: pointer to the data */
    int nsamps,                 /* I: number of samples */
    double *mean                /* O: mean of the data */
);

int ias_math_compute_rmse
(
    const double *sample,   /* I: pointer to the data */
    int  nsamps,            /* I: number of samples */
    double *rms_error       /* O: root mean square error of the data */
);

int ias_math_compute_stdev
(
    const double *sample,   /* I: pointer to the data */
    int nsamps,             /* I: number of samples */
    double mean,            /* I: mean of the samples */
    double *std_dev         /* O: standard deviation of the data */
);

double ias_math_compute_t_confidence
(
    double t,        /* I: value at which to evaluate t dist PDF */
    int dof          /* I: t distribution degrees of freedom */
);

void ias_math_transpose_matrix
( 
    const double *a,       /* I: matrix to be transposed */
    double *b,             /* O: transpose of matrix */
    int row,               /* I: number of rows in matrix */
    int col                /* I: number of columns in matrix */
);
int ias_math_multiply_matrix
( 
    const double *a,        /* I: ptr to matrix */
    const double *b,        /* I: ptr to matrix */
    double *m,              /* O: ptr to output matrix */
    int arow,               /* I: number of rows in a */
    int acol,               /* I: number of cols in a */
    int brow,               /* I: number of rows in b */
    int bcol                /* I: number of cols in b */
);

int ias_math_invert_matrix
(
    const double *a,    /* I: input array */
    double *y,          /* O: ouput array */
    int n               /* I: dimension of a (n y n) */
);

void ias_math_add_matrix
( 
    const double *a,        /* I: input matrix 1 */
    const double *b,        /* I: input matrix 2 */
    double *m,              /* O: output matrix */
    int row,                /* I: number of rows in matrices */
    int col                 /* I: number of cols in matrices */
);

void ias_math_subtract_matrix
(
    const double *a,         /* I: input matrix 1 */
    const double *b,         /* I: input matrix 2 */
    double *m,               /* O: output matrix */
    int row,                 /* I: number of rows in matrix */
    int col                  /* I: number of cols in matrix */
);

int ias_math_decompose_lu_matrix
(
    double *a,               /* I/O: input matrix, output LU decomp matrix */
    int n,                   /* I: size of matrix */
    int *indx,               /* O: index */
    double *d                /* O: flag */
);

int ias_math_check_if_singular_matrix
(
    double *matrix,  /* I: Input matrix */
    int matrix_size  /* I: Size of matrix */
);

void ias_math_back_substitute_lu_matrix
(
    double *a,               /* I/O: lu decomp matrix */
    int n,                   /* I: size of matrix */
    const int *indx,         /* I: flag */
    double b[]               /* O: solution vector */
);

/* ias_rotate.c functions */
void ias_math_rotate_3dvec_around_x
(
    const IAS_VECTOR *r_old, /* I: coordinates (x, y, z) in the old system */
    double angle,     /* I: angle to be rotated, in radian, positive
                         anticlockwise */
    IAS_VECTOR *r_new /* O: coordinates in the new system */
);

void ias_math_rotate_3dvec_around_y
(
    const IAS_VECTOR *r_old, /* I: coordinates (x, y, z) in the old system */
    double angle,        /* I: angle to be rotated, in radian, positive
                                            anticlockwise */
    IAS_VECTOR *r_new    /* O: coordinates in the new system */
);

void ias_math_rotate_3dvec_around_z
(
    const IAS_VECTOR *r_old, /* I: coordinates (x, y, z) in the old system */
    double angle,      /* I: angle to be rotated, in radian, positive
                             anticlockwise */
    IAS_VECTOR *r_new  /* O: coordinates in the new system */
);

int ias_math_point_in_polygon
(
    int vertice_count,      /* I: Number of vertices in x/y_vert */
    const double *x_vert,   /* I: Vertices of polygon */
    const double *y_vert,   /* I: Vertices of polygon */
    double x_coord,         /* I: X coordinate of point */
    double y_coord          /* I: Y coordinate of point */
);

int ias_math_calculate_polygon_centroid
(
    const double *x_points, /* I: Array of x points. Parallel to y_points. */
    const double *y_points, /* I: Array of y points. Parallel to x_points. */
    int num_points,   /* I: Number of provided points */
    double *x_center, /* O: X coordinate of polygon center */
    double *y_center  /* O: Y coordinate of polygon center */
);

int ias_math_predict_state
(
    const double *S,   /* I: State transition matrix */
    const double *Xk,  /* I: State at K */
    double *Xk1,       /* O: State at K+1 */
    int m              /* I: size of matrix */
);

int ias_math_compute_kalman_gain
(
    const double *Pn, /* I: Predicted error covariance matrix */
    const double *H,  /* I: size of state matrix */
    const double *R,  /* I: Covariance of measured noise matrix */
    double *K,        /* O: Kalman gain matrix */
    int m,            /* I: size in m direction */
    int n             /* I: size in n direction */
);

int ias_math_compute_predicted_error_covar
(
    const double *S,    /* I: State transition matrix */
    const double *Pn,   /* I: Filtered error covariance matrix */
    double *Pn1,        /* O: Predicted error covariance matrix at k+1 */
    const double *Q,    /* I: Process noise matrix */
    int m               /* I: size in m direction */
);

int ias_math_update_filter_state
(
    const double *Xk, /* I: State matrix at time K */
    double *Xk1,      /* O: State matrix at time K+1 */
    const double *K,  /* I: Kalman gain matrix */
    const double *z,  /* I: measure ment of gain matrix */
    const double *H,  /* I: size of state matrix */
    int m,            /* I: size in m direction */
    int n             /* I: size in n direction */
);

int ias_math_update_filter_error_covar
(
    const double *K,  /* I: Kalman gain matrix */
    const double *H,  /* I: size of state matrix */
    const double *Pn, /* I: Filtered error covariance matrix */
    double *Pn1,      /* O: predicted error covariance matrix */
    int m,            /* I: size in m direction */
    int n             /* I: size in n direction */
);

int ias_math_smooth_gain
(
    const double *P,   /* I: Filtered covariance matrix */
    const double *Pn,  /* I: predicted error covariance matrix at time K+1 */
    const double *S,   /* I: state transition matrix */
    double *A,         /* O: smoothing gain matrix */
    int m              /* I: size in m direction */
);

int ias_math_smooth_state
(
    const double *X,   /* I: state matrix */
    const double *Xk,  /* I: state matrix at time K */
    const double *XN,  /* I: estimate of state [X] up to N */
    const double *A,   /* I: smoothing gain matrix */
    double *XN1,       /* O: predicted error covariance at time K+1*/
    int m              /* I: size in m direction */
);

double ias_math_interpolate_lagrange
(
    const double *p_YY, /* I: Pointer to the array of input values */
    const double *p_XX, /* I: Pointer to the array of times closest to the
                              requested time */
    int n_pts,          /* I: Number of points for the interpolation */
    double in_time      /* I: Requested time for interpolation */
);

void ias_math_interpolate_lagrange_3dvec
( 
    const IAS_VECTOR *p_YY, /* I: Y-axis or values to be interpolated */
    const double *p_XX,     /* I: X-axis of value to be interpolated */
    int num_points,    /* I: Number of points to use in the interpolation */
    double in_time,    /* I: X-value for which Y value is to be calculated */
    IAS_VECTOR *output /* O: Output result */
);

void ias_math_heapsort_double_array
(
    int n,        /* I: number of elements        */
    double *ra    /* I/O: element values to sort  */
);

void ias_math_insertion_sort_integer_array
(
    int num_values,  /* I: Number of values to sort     */
    int sort_array[] /* I/O: Array of integers to sort  */
);

int ias_math_is_leap_year
(
    int year    /*I: Year to test */
);

int ias_math_check_pixels_in_range
(
    const float *window,         /* I: Image window                          */
    int  size,                   /* I: Size of the window (nlines * nsamps)  */
    float invalid_thresh,        /* I: Threshold for out of range data       */
    float valid_image_max,       /* I: Upper bound defining valid vs invalid */
    float valid_image_min        /* I: Lower bound defining valid vs invalid */
);

unsigned short int ias_math_find_median_unsigned
( 
    int N,                      /* I: Number of elements in the array */
    const unsigned short int z[]/* I: An integer array to find median from */
);

void ias_math_fft2d
(
    double *data,  /* I/O: array */
    int  nrows,    /* I: number of rows */
    int  ncols,    /* I: number of columns */
    IAS_FFT_DIRECTION_TYPE isign /* I: Flag where IAS_MATH_FFT = 2 dimensional 
                              discrete Fourier Transform
                         IAS_MATH_IFFT = inverse transform times the product of
                              the lengths of all dimensions */
);

int ias_math_compute_grey_cross_same_size
(
    const float *images,  /* I: Search subimage */
    const float *imager,  /* I: Reference subimage */
    const int *win_size,  /* I: Actual size of subimage windows: samps/lines */
    int max_off,          /* I: Maximum offset to search for */
    double *unormc        /* O: Array of unnormalized (raw) counts of edge 
                                coincidences for each alignment of reference
                                image relative to search image */
);

void ias_math_normalize_grey_cross_same_size
(
    const int *surf_size, /* I: Size of correlation surface (unormc and ccnorm)
                              samps,lines */
    const double *unormc, /* I: Array of raw cross product sums */
    double *ccnorm,       /* O: Array of norm xcorr coeffs */
    double *pkval,        /* O: Table of top NPEAK normalized values */
    int *ipkcol,          /* O: Table of column numbers for top NPEAK values */
    int *ipkrow,          /* O: Table of row numbers for top NPEAK values */
    double sums[2],       /* O: Sum of all normalized values, sum of squares */
    int abs_corr_coeff    /* I: the flag to use the abs correlation coeffs */
);

int ias_math_compute_grey_cross
(
    const float *images, /* I: Search subimage */
    const float *imager, /* I: Reference subimage */
    const int *srch_size,/* I: Actual size of search subimage:  samps/lines */
    const int *ref_size, /* I: Actual size of reference subimage: samps/lines */
    int ncol,           /* I: # of columns in cross-product sums array(unormc)*/
    int nrow,           /* I: # of rows in cross-product sums array (unormc) */
    double *unormc      /* O: Array of unnormalized (raw) counts of edge 
                              coincidences for each alignment of reference
                              image relative to search image */
);

int ias_math_normalize_grey_cross
(
    const float *imager, /* I: Reference subimage */
    const float *images, /* I: Search subimage */
    const int *ref_size, /* I: Actual size of reference subimage--samps/lines */
    const int *srch_size,/* I: Actual size of search subimage--samps/lines */
    int  ncol,          /* I: #cols in cross-product sum array            */
    int  nrow,          /* I: #rows in cross-product sum array            */
    const double *unormc,/* I: Array of raw cross product sums */
    double *ccnorm,     /* O: Array of norm xcorr coeffs */
    double *pkval,      /* O: Table of top 32 normalized values */
    int  *ipkcol,       /* O: Table of column numbers for top 32 values */
    int  *ipkrow,       /* O: Table of row numbers for top 32 values */
    double *sums,       /* O: Sum of all normalized values, & sum of squares*/
    int abs_corr_coeff  /* I: the flag to use the abs of the correlation 
                              coeffs*/
);

void ias_math_evaluate_grey
(
    int  ncol,      /* I: Number of columns in coincidence-count array */
    int  nrow,      /* I: Number of rows in coincidence-count array */
    const double *ccnorm,/* I: Array of norm xcorr coeffs */
    const double *pkval,/* I: Table of top 32 normalized values */
    const int  *ipkcol, /* I: Table of column numbers for top 32 values */
    const int  *ipkrow, /* I: Table of row numbers for top 32 values */
    double *sums,       /* O: Sum of all normalized values and sum of squares */
    double min_corr,    /* I: Minimum acceptable correlation strength */
    double *strength,   /* O: Strength of correlation */
    double *cpval,      /* O: 3 by 3 array of xcorr vals in std dev */
    int  *mult_peak_flag,/* O: subsidiary peak too near edge of search area */
    int  *edge_flag,    /* O: peak too near edge of search area */
    int  *low_peak_flag /* O: strength of peak below minimum */
);

int ias_math_fit_registration
(
    const double *cpval, /* I: 3 by 3 array of xcorr vals, in units of standard 
                               dev above background, centered on corr peak   */
    IAS_CORRELATION_FIT_TYPE fit_method, /* I: Method of surface fit: 
                                 FIT_ELLIP_PARA - Elliptical paraboloid
                                 FIT_ELLIP_GAUSS - Elliptical Gaussian
                                 FIT_RECIP_PARA - Reciprocal Paraboloid      */
    double *pkoffs,      /* O: Best-fit horiz and vertical offsets of 
                               correlation peak relative to center of 3 by 3 
                                array */
    double *est_err      /* O: Estimated horizontal error [0], vertical 
                               error [1] and h-v cross term [2] in best-fit 
                               offsets */
);

int ias_math_correlate_grey
(
    const float *images, /* I: Search subimage                              */
    const float *imager, /* I: Reference subimage                           */
    const int *srch_size,/* I: Actual size of search subimage:  samps,lines */
    const int *ref_size, /* I: Actual size of reference subimage: samps,lines */
    double min_corr,    /* I: Minimum acceptable correlation strength        */
    IAS_CORRELATION_FIT_TYPE fit_method, /* I: Surface Fit Method:
                                    FIT_ELLIP_PARA - Elliptical paraboloid 
                                    FIT_ELLIP_GAUSS - Elliptical Gaussian 
                                    FIT_RECIP_PARA - Reciprocal Paraboloid 
                                    FIT_ROUND - Round to nearest int    */
    double max_disp,    /* I: Maximum allowed diagonal displacement from nominal
                              tiepoint loc to location found by correlation  */
    const double *nom_off,/* I: Nominal horiz & vert offsets of UL corner of 
                              reference subimage relative to search subimage */
    double *strength,   /* O: Strength of correlation                        */
    double *fit_offset, /* O: Best-fit horiz & vert offsets of correlatn peak*/
    double *est_err,    /* O: Est horiz error, vert error, and h-v cross term in
                              best-fit offsets (3 values)                     */
    double *diag_disp,  /* O: Actual diagonal displacement from nominal tiepoint
                              location to location found by correlation      */
    int *mult_peak_flag,/* O: subsidiary peak too near edge of search area */
    int *edge_flag,     /* O: peak too near edge of search area */
    int *low_peak_flag, /* O: strength of peak below minimum */
    int *max_disp_flag, /* O: diag displacement from nom location exceeds max*/
    int abs_corr_coeff  /* I: flag to use the abs of the correlation coeffs */
);

int ias_math_correlate_fine
(
    const float *images, /* I: Search subimage */
    const float *imager, /* I: Reference subimage */
    const int *srch_size,/* I: Actual size of search subimage:  samps,lines */
    const int *ref_size, /* I: Actual size of reference subimage: samps,lines */
    double *fit_offset, /* O: Best-fit horizontal & vertical offsets of
                              correlation peak                               */
    double *diag_disp   /* O: Actual diagonal displacement from nominal
                              tiepoint location to location found by
                              correlation                                    */
);

/* ias_math_convert_j2000.c routines */
int ias_math_init_leap_seconds
(
    double j2secs,  /* I: seconds since J2000 to use for initializing the
                          number of leap seconds */
    const IAS_MATH_LEAP_SECONDS_DATA *cpf_leap_seconds
                    /* I: Leap seconds info from CPF */
);

int ias_math_init_leap_seconds_from_UTC_time
(
    const double *ref_time, /* I: Year, day of year, seconds of day to use to
                                 determine the number of leap seconds */
    const IAS_MATH_LEAP_SECONDS_DATA *cpf_leap_seconds
                            /* I: Leap seconds info from CPF */
);

void ias_math_clear_leap_seconds();

int ias_math_get_leap_second_adjustment
(
    int *seconds    /* O: the leap seconds adjustment previously initialized */
);

int ias_math_convert_year_doy_sod_to_j2000_seconds
(
    const double *time, /* I: Year, DOY, SOD to convert to J2000 epoch time */
    double *j2secs      /* O: Seconds from epoch */
);

int ias_math_convert_j2000_seconds_to_year_doy_sod
(
    double j2secs,  /* I: seconds since J2000 to convert to Year, DOY, SOD */
    double *time    /* O: Year, day of year, seconds of day output array */
);

double ias_math_cubic_convolution
(
   double alpha,   /* I: Cubic convolution alpha parameter     */
   double x        /* I: Value to perform cubic convolution on */
);

void ias_math_conjugate_quaternion
(
    const IAS_QUATERNION *quaternion, /* I: the quaternion to conjugate */
    IAS_QUATERNION *conjugated_quaternion /* O: the conjugated quaternion */
);

int ias_math_convert_euler_to_quaternion
(
    double tolerance,          /* I: error tolerance for the conversion */
    double matrix[3][3],       /* I: matrix to convert */
    IAS_QUATERNION *quaternion /* O: converted results */
);

void ias_math_convert_quaternion_to_euler
(
    const IAS_QUATERNION *quaternion, /* I: the quaternion to convert */
    double matrix[3][3]               /* O: the converted results */
);

int ias_math_find_line_segment_intersection
(
    const IAS_LINE_SEGMENT *line1,  /* I: First line segment */
    const IAS_LINE_SEGMENT *line2,  /* I: Second line segment */
    double *intersect_x,            /* O: X coordinate of intersection */
    double *intersect_y             /* O: Y coordinate of intersection */
);

/* Simulated annealing functions */
int ias_math_simulated_annealing
(
    int npts,                      /* I: Number of points in profile */
    double **p,                    /* I/O: Simplex array of parameter vectors */
    double *y,                     /* I/O: Objective function values for the
                                         simplex */
    int ndim,                      /* I: Number of unknown parameters */
    double *pb,                    /* I/O: Array of "best" parameters */
    double *yb,                    /* I/O: Objective function value at pb */
    double ftol,                   /* I: Convergence tolerance */
    double (*eval_func)(int n, double a[], void *func_params),
                                   /* I: Pointer to the objective function */
    void *func_params,             /* I: Pointer to a structure for any 
                                         parameters that need to be passed to
                                         the eval function (can be NULL) */
    int *iter,                     /* I/O: Iteration limit/counter */
    double temptr                  /* I: Annealing temperature */
);

int ias_math_merge_mean_and_stdev_samples
(
    const int *counts_buffer, /* I: Counts per entry */
    const float *mean_buffer, /* I: Mean per entry */
    const float *stdev_buffer,/* I: Standard deviation per entry */
    int buffer_entry_count,   /* I: Total number of entries */
    float *combined_stdev,    /* O: Combined Standard deviation */
    float *combined_mean,     /* O: Combined Mean */
    float *combined_counts    /* O: Combined Count */
);

int ias_math_point_in_closed_polygon
(
    unsigned int num_sides,     /* I: Number of sides in polygon */
    const double *vert_x,       /* I: Vertices of polygon */
    const double *vert_y,       /* I: Vertices of polygon */
    double point_x,             /* I: X coordinate of point */
    double point_y,             /* I: Y coordinate of point */
    unsigned int num_segs,      /* I: Number of polygon segments */
    const IAS_POLYGON_SEGMENT *poly_seg/* I: Array of polygon segments */
);

int ias_math_point_in_closed_polygon_distance
(
    unsigned int num_sides,              /* I: Number of sides in polygon */
    const double *vert_x,                /* I: Vertices of polygon */
    const double *vert_y,                /* I: Vertices of polygon */
    double point_x,                      /* I: X coordinate of point */
    double point_y,                      /* I: Y coordinate of point */
    unsigned int num_segs,               /* I: Number of polygon segments */
    const IAS_POLYGON_SEGMENT *poly_seg,/* I: Array of polygon segments */
    unsigned int direction, /* I: Direction to measure distance: 0=x, 1=y */
    double *distance        /* O: Distance from point to polygon boundary in
                               specified direction */
);

/* math constants */
double ias_math_get_pi();
double ias_math_get_two_pi();
double ias_math_get_arcsec_to_radian_conversion();
double ias_math_get_radians_per_degree();
double ias_math_get_degrees_per_radian();
double ias_math_get_degrees_per_hour();

#endif
