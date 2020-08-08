#ifndef _XXX_CONST_H_
#define _XXX_CONST_H_

/*
 * IAS subsystem-wide definitions
 */
/* data field sizes */
#define STRLEN 256  /* length of a string */

/* file access modes */
typedef enum access_mode
{
    access_error=-1,
    READ,
    WRITE,
    UPDATE
} AccessMode;

#ifndef XXX_DATA_TYPE
#define XXX_DATA_TYPE
/* This is also re-implemented in the IAS Qt Library. */
/* valid data types */
typedef enum data_type
{
    type_error=-1,  /* invalid type       */
    BYTE,     /* 1-byte           */
    I2,       /* 2-byte signed integer    */
    UI2,      /* 2-byte unsigned integer     */
    I4,       /* 4-byte signed integer    */
    UI4,      /* 4-byte signed integer    */
    R4,       /* 4-byte floating point    */
    R8,       /* 8-byte floating point    */
    NDTYPES   /* number of different data types */
} DataType;
#endif /* XXX_DATA_TYPE */

/* additional error codes */
#define IAS_EXISTS -2

#define MAX_PATH_WRS1 251               /* maximum WRS1 path    */
#define MAX_PATH_WRS2 233               /* maximum WRS2 path     */
#define MAX_ROW_WRS 248                 /* maximum WRS row, WRS1 & WRS2 same */
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
