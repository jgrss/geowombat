#ifndef IAS_TYPES_H
#define IAS_TYPES_H

/* valid data types */
typedef enum ias_data_type
{
    type_error=-1,  /* invalid type */
    IAS_BYTE,           /* 1-byte */
    IAS_I2,             /* 2-byte signed integer */
    IAS_UI2,            /* 2-byte unsigned integer */
    IAS_I4,             /* 4-byte signed integer */
    IAS_UI4,            /* 4-byte unsigned integer */
    IAS_R4,             /* 4-byte floating point */
    IAS_R8,             /* 8-byte floating point */
    IAS_CHAR,           /* C-character type */
    IAS_UCHAR,          /* unsigned C-character type */
    IAS_NDTYPES         /* number of different data types */
} IAS_DATA_TYPE;

/* Define the C types for the above enumerations */
typedef  unsigned char IAS_BYTE_TYPE;  /* 1-byte */
typedef          short IAS_I2_TYPE;    /* 2-byte signed integer */
typedef unsigned short IAS_UI2_TYPE;   /* 2-byte unsigned integer */
typedef            int IAS_I4_TYPE;    /* 4-byte signed integer */
typedef   unsigned int IAS_UI4_TYPE;   /* 4-byte unsigned integer */
typedef          float IAS_R4_TYPE;    /* 4-byte floating point */
typedef         double IAS_R8_TYPE;    /* 8-byte floating point */
typedef           char IAS_CHAR_TYPE;  /* C-character type */
typedef  unsigned char IAS_UCHAR_TYPE; /* unsigned C-character type */

/* file access modes */
typedef enum ias_access_mode
{
    IAS_READ,
    IAS_WRITE,
    IAS_UPDATE
} IAS_ACCESS_MODE;

typedef enum ias_frame_type
{
    IAS_UNKNOWN_FRAME_TYPE = -1, 
    IAS_GEOBOX = 1,                 /* User specifies UL and LR lat/long */
    IAS_PROJBOX,                    /* User specifies LR output projection
                                       coordinates and another output space
                                       point, plus input space line/sample */
    IAS_UL_SIZE,                    /* User specifies UL output projection
                                       coordinate, number of output space 
                                       lines and samples */ 
    IAS_MINBOX,                     /* Minbox framing - optimal band */
    IAS_MAXBOX,                     /* Maxbox framing - output frame contains
                                       all input pixels from all bands */
    IAS_PATH_ORIENTED,              /* Path oriented framing - framing based 
                                       on preset number of lines/samples */
    IAS_PATH_MINBOX,                /* Combines path oriented and minbox */
    IAS_PATH_MAXBOX,                /* Combines path oriented and maxbox */
    IAS_LUNAR_MINBOX,               /* Lunar based Minbox framing */
    IAS_LUNAR_MAXBOX,               /* Lunar based Maxbox framing */
    IAS_STELLAR_FRAME               /* Stellar based framing */
} IAS_FRAME_TYPE;


/* fit method used in correlation */
typedef enum ias_corr_fit_method
{
    IAS_FIT_ERROR=-1,
    IAS_FIT_ELLIP_PARA=1,           /* Elliptical paraboloid */
    IAS_FIT_ELLIP_GAUSS,            /* Elliptical Gaussian */
    IAS_FIT_RECIP_PARA,             /* Reciprocal paraboloid */
    IAS_FIT_ROUND                   /* Round to nearest integer */
} IAS_CORRELATION_FIT_TYPE;


typedef enum ias_acquisition_type
{
    IAS_ACQUISITION_TYPE_ERROR = -1,/* invalid type */
    IAS_EARTH,                      /* Earth based acquisition */
    IAS_LUNAR,                      /* Lunar based acquisition */
    IAS_STELLAR,                    /* Stellar based acquisition */
    IAS_OTHER_ACQUISITION           /* Other acquisition (e.g. an RPS type) */
} IAS_ACQUISITION_TYPE;

typedef enum ias_correction_type
{
    IAS_CORRECTION_TYPE_ERROR = -1, /* invalid type */
    IAS_SYSTEMATIC,                 /* Systematic image correction */
    IAS_PRECISION,                  /* Precision image correction */
    IAS_TERRAIN                     /* Systematic terrain correction */
} IAS_CORRECTION_TYPE;


/* Date/time string format types */
typedef enum ias_datetime_format_type
{
    IAS_DATETIME_TYPE_ERROR = -1,   /* Invalid date/time string format */
    IAS_DATETIME_L0R_FORMAT,
    IAS_DATETIME_CPF_FORMAT
} IAS_DATETIME_FORMAT_TYPE;

/* Processing system types */
typedef enum ias_processing_system_type
{
    IAS_PROCESSING_SYSTEM_UNDEF = -1,  /* Undefined processing system */
    IAS_PROCESSING_SYSTEM_IAS,
    IAS_PROCESSING_SYSTEM_LPGS
} IAS_PROCESSING_SYSTEM_TYPE;

#endif
