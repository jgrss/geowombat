#ifndef _GXX_GEO_MATH_H_
#define _GXX_GEO_MATH_H_

#include <math.h>
#include <stdlib.h>
#include "xxx_Types.h"
#include "gxx_const.h"
#include "gxx_structures.h"

/* GPS GEO MATH */
void gxx_anglebetwn
(
    const VECTOR *vec1, /* I: Input vector one           */
    const VECTOR *vec2, /* I: Input vector two           */
    double *angle       /* O: Angle between vectors (radians)*/
);

void gxx_cross
(
    const VECTOR *vec1, /* I: Input vector number one */
    const VECTOR *vec2, /* I: Input vector number two */
    VECTOR *vec3        /* O: Output vector (cross product) */
);

double gxx_dot
(
    const VECTOR *vec1, /* I Vector one to be multiplied     */
    const VECTOR *vec2  /* I Vector two to be multiplied         */
);

int gxx_earth2orbit
(
    VECTOR satpos,              /* I: satellite position vector in Earth-fixed
                                      system, in meters */
    VECTOR satvel,              /* I: inertial satellite velocity vector in
                                      Earth-fixed system, in m/sec */
    double transf_matrix[3][3]  /* O: 3 X 3 transformation matrix, from the
                                      Earth-fixed system to the orbit-oriented
                                      cartesian system. */
);

int gxx_invmatrix
(
    const double inmatrix[3][3], /* I: Input matrix for inversion   */
    double outmatrix[3][3]       /* O: Inverted matrix              */
);

double gxx_norm
(
    const VECTOR *vec  /* I: Vector to find the length of         */
);

int gxx_point_in_polygon
(
    unsigned int nsides, /* I: Number of sides in polygon */
    double *vert_x,      /* I: Vertices of polygon */
    double *vert_y,      /* I: Vertices of polygon */
    double x,            /* I: X coordinate of point */
    double y,            /* I: Y coordinate of point */
    unsigned int nsegs,  /* I: Number of polygon segments */
    psegment *pseg       /* I: Array of polygon segments */
);

int gxx_point_in_polygon_dist
(
    unsigned int nsides, /* I: Number of sides in polygon */
    double *vert_x,      /* I: Vertices of polygon */
    double *vert_y,      /* I: Vertices of polygon */
    double x,            /* I: X coordinate of point */
    double y,            /* I: Y coordinate of point */
    unsigned int nsegs,  /* I: Number of polygon segments */
    psegment *pseg,      /* I: Array of polygon segments */
    unsigned int dir,    /* I: Direction to measure distance: 0=x, 1=y */
    double *dist         /* O: Distance from point to polygon boundary in
                               y direction */
);
 
void gxx_transvec
(
    VECTOR Xold,        /* I: vector in old system */
    double Trans[3][3], /* I:transformation matrix from old to new system*/
    VECTOR *Xnew        /* O: vector in the new system */
);

int gxx_unit
(
    const VECTOR *vec, /* I: Input vector */
    VECTOR *uni        /* O: Unit vector of the input vector */
);

int gxx_vector_lagrange
(
    VECTOR *p_YY,    //!<[in] Y-axis or values to be interpolated
    double *p_XX,    //!<[in] X-axis of value to be interpolated
    unsigned int n_pts, //!<[in] Number of points to use in interpolation
    double in_time,  //!<[in] X-value to for which Y value is to be calculated
    VECTOR *output,  //!<[out] Result
    char *msg        //!<[out] Status message
);

int gxx_corr_coeff
(
    const double *x_sample,     /* I: pointer to the x data */
    const double *y_sample,     /* I: pointer to the y data */
    int  nsamps,                /* I: number of samples in x and in y */
    double x_mean,              /* I: mean of the x data */
    double y_mean,              /* I: mean of the y data */
    double x_std_dev,           /* I: standard deviation of the x data */
    double y_std_dev,           /* I: standard deviation of the y data */
    double *corr_coeff          /* O: correlation coefficient between 
                                      the x and y data */
);

int gxx_matrix_multiply
( 
    const double *a,    /* I: ptr to matrix */
    const double *b,    /* I: ptr to matrix */
    double *m,          /* O: ptr to output matrix */
    const int arow,     /* I: number of rows in a */
    const int acol,     /* I: number of cols in a */
    const int brow,     /* I: number of rows in b */
    const int bcol      /* I: number of cols in b */
);

/* gxx_cvtime.c functions */
int  gxx_cvtime
(
    int  hour,  /* I: Hours   */
    int  min,   /* I: Minutes */
    int  sec    /* I: Seconds */
);

double gxx_cvt2sec
(
    int  year,  /* I: Year (4digits) */
    int  day,   /* I: Day            */
    int  hour,  /* I: Hours          */
    int  min,   /* I: Minutes        */
    double sec  /* I: Seconds        */
);

void gxx_cvtsec
(
    int  tsec,  /* I: Total seconds  */
    int  *hour, /* O: Hours          */
    int  *min,  /* O: Minutes        */
    int  *sec   /* O: Seconds        */
);

void gxx_cvtmil
(
    int  milsec,    /* I: Total milliseconds */
    int  *hour,     /* O: Hours              */
    int  *min,      /* O: Minutes            */
    int  *sec,      /* O: Seconds            */
    int  *fsec      /* O: Fractional seconds */
);

/* gxx_rotate.c functions */
void  gxx_rotatex
(
    VECTOR r_old, /* I: coordinates (x, y, z) in the old system */
    double angle, /* I: angle to be rotated, in radian, positive
                        anticlockwise */
    VECTOR *r_new /* O: coordinates in the new system */
);

void  gxx_rotatey
(
    VECTOR r_old, /* I: coordinates (x, y, z) in the old system */
    double angle, /* I: angle to be rotated, in radian, positive
                        anticlockwise */
    VECTOR *r_new /* O: coordinates in the new system */
);

void  gxx_rotatez
(
    VECTOR r_old, /* I: coordinates (x, y, z) in the old system */
    double angle, /* I: angle to be rotated, in radian, positive
                        anticlockwise */
    VECTOR *r_new /* O: coordinates in the new system */
);

/* ************************
   Kalman filter functions
   and definitions
   ************************* */

int gxx_predict_state
(
    double *S,   /* I: State transition matrix */
    double *Xk,  /* I: State at K */
    double *Xk1, /* O: State at K+1 */
    int m        /* I: size of matrix */
);

int gxx_kalman_gain
(
    double *Pn, /* I: Predicted error covariance matrix */
    double *H,  /* I: size of state matrix */
    double *R,  /* I: Covariance of measured noise matrix */
    double *K,  /* O: Kalman gain matrix */
    int m,      /* I: size in m direction */
    int n       /* I: size in n direction */
);

int gxx_predict_error_covar
(
    double *S,   /* I: State transition matrix */
    double *Pn,  /* I: Filtered error covariance matrix */
    double *Pn1, /* O: Predicted error covariance matrix at k+1 */
    double *Q,   /* I: Process noise matrix */
    int m        /* I: size in m direction */
);

int gxx_filter_state
(
    double *Xk, /* I: State matrix at time K */
    double *Xk1,/* O: State matrix at time K+1 */
    double *K,  /* I: Kalman gain matrix */
    double *z,  /* I: measure ment of gain matrix */
    double *H,  /* I: size of state matrix */
    int m,      /* I: size in m direction */
    int n       /* I: size in n direction */
);

int gxx_filter_error_covar
(
    double *K,  /* I: Kalman gain matrix */
    double *H,  /* I: size of state matrix */
    double *Pn, /* I: Filtered error covariance matrix */
    double *Pn1,/* O: predicted error covariance matrix */
    int m,      /* I: size in m direction */
    int n       /* I: size in n direction */
);

int gxx_smooth_gain
(
    double *P,  /* I: Filtered covariance matrix */
    double *Pn, /* I: predicted error covariance matrix at time K+1 */
    double *S,  /* I: state transition matrix */
    double *A,  /* O: smoothing gain matrix */
    int m       /* I: size in m direction */
);

int gxx_smooth_state
(
    double *X,  /* I: state matrix */
    double *Xk, /* I: state matrix at time K */
    double *XN, /* I: estimate of state [X] up to N */
    double *A,  /* I: smoothing gain matrix */
    double *XN1,/* O: predicted error covariance at time K+1*/
    int m       /* I: size in m direction */
);

#endif
