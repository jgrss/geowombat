#ifndef _GXX_CONST_H_
#define _GXX_CONST_H_

#include "xxx_Band.h"

/*
 * GPS subsystem-wide definitions
 */
/* GPS data field sizes */
#define PTIDLEN 16  /* length of a point id */
#define STRLEN 256  /* length of a string */

/* resolutions for the different bands */
typedef enum raw_pixel_size 
{
    unknown_size=-1,
    _30m=30,
    _15m=15,
    _60m=60,
    _120m=120
} RawPixelSize;      /* raw resolution pixel sizes */

/* fit method used in correlation */
typedef enum corr_fit_method
{
    FIT_ERROR=-1,
    FIT_ELLIP_PARA=1,
    FIT_ELLIP_GAUSS,
    FIT_RECIP_PARA,
    FIT_ROUND
} CorrFitMethod;

/* type of correction being used */
typedef enum correction_type
{
    correction_type_error=-1,
    SYSTEMATIC=1,
    PRECISION,
    TERRAIN
} CorrectionType;

/* additional error codes */
#define IAS_EXISTS -2

#define MAX_PATH_WRS1 251       /* maximum WRS-1 path     */
#define MAX_PATH_WRS2 233       /* maximum WRS-2 path     */
#define MAX_ROW_WRS 248         /* maximum WRS row      */
#define WRS_SCENE_SIZE_LINES 5965       /* Number of lines in WRS scene */
#define WRS_SCENE_SIZE_SAMPS 6967       /* Number of samps in WRS scene */
#define EPOCH_2000      2451545.0       /* Julian date of epoch 2000    */
#define JULIAN_CENTURY      36525.0     /* Julian century       */
#define HALFPIE 1.5707963267948966      /* One half PI                  */
#define PIE  3.141592653589793238       /* PI                           */
#define PIE2 6.283185307179586476       /* two PI                       */
#define LITESPEED 2.99793e8             /* Speed of light               */
#define ABER_CORRECTION -25.2267e-6     /* aberration correction for
                                           across_angle calculation     */
#define WE  7.272205217e-5              /* Solar earth rotation rate    */
#define WEI 7.292115428e-5              /* Inertial earth rotation rate */
#define RPD (PIE / 180.0)               /* Radians per degree           */
#define DPR (180.0 / PIE)               /* Degrees per radian           */
#define A2R 4.848136811e-6      /* arc seconds to radians   */
#define MU 3.986012e14                  /* MU                           */
#define REM 6.378165000e6             /* Mean radius of the earth at equator*/

#define SEC_PER_DAY     86400        /* Seconds Per Day                     */
#define MIL_PER_DAY     86400000     /* Milliseconds Per Day                */
#define MIL_PER_HOUR    3600000      /* Milliseconds Per Hour               */
#define MIL_PER_MIN     60000        /* Milliseconds Per Hour               */
#define MIL_PER_SEC     1000         /* Milliseconds Per Second             */
 
#define SECS2RADS 206264.806247096355  /* conversion from Seconds to Radians*/

#ifndef D2R
#define D2R     1.745329251994328e-2 /* conversion for Degrees to Radians   */
#endif

#ifndef R2D
#define R2D     57.295779513082321   /* conversion for Radians to Degrees   */
#endif

/* NOTE: the MIN and MAX definitions must appear the same as sys/param.h */
#ifndef MIN                          /* macro for min of two numbers        */
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef MAX                          /* macro for max of two numbers        */
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif
#endif
