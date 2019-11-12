#ifndef _IAS_ANGLE_GEN_DISTRO_H_
#define _IAS_ANGLE_GEN_DISTRO_H_

/* IAS Library Includes */
#include "ias_structures.h"     /* IAS_VECTOR and IAS_CORNERS */
#include "ias_satellite_attributes.h" /* IAS_MAX_NBANDS/NSCAS */
#include "ias_geo.h"            /* IAS_PROJECTION */

/* Local Defines */
#define IAS_ANGLE_GEN_NUM_RPC_COEF 5     /* Number of image RPC coefficients */
#define IAS_ANGLE_GEN_ANG_RPC_COEF 10    /* Number of angle RPC coefficients */
#define IAS_ANGLE_GEN_SPACECRAFT_SIZE 32 /* Spacecraft name size */
#define IAS_ANGLE_GEN_SCENE_ID_LENGTH 21 /* Scene ID length */
#define IAS_ANGLE_GEN_ZENITH_INDEX 0    /* Array index for the zenith angle */
#define IAS_ANGLE_GEN_AZIMUTH_INDEX 1   /* Array index for the azimuth angle */

typedef enum ias_angle_gen_type
{
    IAS_ANGLE_GEN_UNKNOWN = 0,
    IAS_ANGLE_GEN_SATELLITE,
    IAS_ANGLE_GEN_SOLAR
} IAS_ANGLE_GEN_TYPE;

/* Subset of rational polynomial coefficient terms used to Map the L1T
   line/sample to the L1R line/sample */
typedef struct IAS_ANGLE_GEN_IMAGE_RPC_TERMS
{
    double l1r_mean_offset;                             /* L1R mean offset */
    double l1t_mean_offset;                             /* L1T mean offset */
    double numerator[IAS_ANGLE_GEN_NUM_RPC_COEF];       /* Numerator coeffs */
    double denominator[IAS_ANGLE_GEN_NUM_RPC_COEF - 1]; /* Denominator coeffs */
} IAS_ANGLE_GEN_IMAGE_RPC_TERMS;

/* Rational polynomial coefficients that map the L1T line/sample to the
   L1R line/sample. */
typedef struct IAS_ANGLE_GEN_IMAGE_RPC
{
    int sca_number;                           /* SCA number*/
    double mean_height;                       /* Mean height offset */
    IAS_ANGLE_GEN_IMAGE_RPC_TERMS line_terms; /* Line terms */
    IAS_ANGLE_GEN_IMAGE_RPC_TERMS samp_terms; /* Sample terms */
} IAS_ANGLE_GEN_IMAGE_RPC;

/* Numerator and denominator rational polynomial coefficient terms used 
   in calculating the satellite viewing angle or solar illumination azimuth 
   and zenith angles for a specified L1T line/sample. */ 
typedef struct IAS_ANGLE_GEN_ANG_RPC_TERMS
{
    double numerator[IAS_ANGLE_GEN_ANG_RPC_COEF];      /* Numerator coeffs */
    double denominator[IAS_ANGLE_GEN_ANG_RPC_COEF - 1];/* Denominator coeffs */
} IAS_ANGLE_GEN_ANG_RPC_TERMS;

/* Offset rational polynomial coefficient terms used in calculating the
   satellite viewing angle or solar illumination azimuth and zenith angles for
   a specified L1T line/sample. */ 
typedef struct IAS_ANGLE_GEN_ANG_OFFSET_TERMS
{
    double l1r_mean_offset;  /* L1R mean offset */
    double l1t_mean_offset;  /* L1T mean offset */
} IAS_ANGLE_GEN_ANG_OFFSET_TERMS;

/* Rational polynomial coefficients used in calculating the satellite viewing
   angle or solar illumination azimuth and zenith angles for a specified L1T
   line/sample. */ 
typedef struct IAS_ANGLE_GEN_ANG_RPC
{
    double mean_height;                  /* Mean height offset */
    IAS_VECTOR mean_offset;              /* Mean vector offsets */
    IAS_ANGLE_GEN_ANG_OFFSET_TERMS line_terms; /* Line offset coefficients */
    IAS_ANGLE_GEN_ANG_OFFSET_TERMS samp_terms; /* Sample offset coefficients */
    IAS_ANGLE_GEN_ANG_RPC_TERMS x_terms; /* X axis coefficients */
    IAS_ANGLE_GEN_ANG_RPC_TERMS y_terms; /* Y axis coefficients */
    IAS_ANGLE_GEN_ANG_RPC_TERMS z_terms; /* Z axis coefficients */
} IAS_ANGLE_GEN_ANG_RPC;

/* All of the rational polynomial coefficients and metadata for a band needed
   to calculate the azimuth and zenith angles and map L1T line/sample to L1R
   line/sample */
typedef struct IAS_ANGLE_GEN_BAND
{
    int band_number;        /* User band number */
    int num_scas;           /* Number of SCAs in the band */
    int l1t_lines;          /* Number of lines in the L1T image. */
    int l1t_samps;          /* Number of samples in the L1T image. */
    int l1r_lines;          /* Number of lines in the L1R image. */
    int l1r_samps;          /* Number of samples in one L1R SCA. */
    double pixel_size;      /* Projection distance per pixel in meters. */
    double image_start_time;/* Image start time, offset in seconds from
                               ephemeris epoch */
    double seconds_per_line;/* L1R line time increment in seconds */
    double active_l1t_corner_lines[4]; /* UL, UR, LR, LL corner lines */
    double active_l1t_corner_samps[4]; /* UL, UR, LR, LL corner samps */ 
    IAS_ANGLE_GEN_ANG_RPC satellite;   /* Satellite viewing angles */
    IAS_ANGLE_GEN_ANG_RPC solar;       /* Solar angles */
    IAS_ANGLE_GEN_IMAGE_RPC sca_metadata[IAS_MAX_NSCAS]; /* SCA RPCs */
} IAS_ANGLE_GEN_BAND;

/* Sample time and position for each ephemeris or solar vector point */
typedef struct IAS_ANGLE_GEN_EPHEMERIS
{
    double sample_time;   /* Sample time from ephemeris epoch */
    IAS_VECTOR position;  /* Position vector */
} IAS_ANGLE_GEN_EPHEMERIS;

/* The main angle metadata structure used to contain all the information 
   need to generate the azimuth and zenith angles and map the L1T line/sample
   to the L1R line/sample for any band/sca combination */
typedef struct IAS_ANGLE_GEN_METADATA
{
    int num_bands;                /* Number of bands in the metadata */
    IAS_ANGLE_GEN_BAND band_metadata[IAS_MAX_NBANDS]; /* Band metadata */
    int band_present[IAS_MAX_NBANDS]; /* Flag to determine if band present */
    IAS_EPOCH_TIME ephem_epoch_time;/* Ephemeric epoch time */
    int ephem_count;                /* Number of ephemeris samples */
    IAS_ANGLE_GEN_EPHEMERIS *ephemeris;    /* Ephemeris data */
    IAS_ANGLE_GEN_EPHEMERIS *solar_vector; /* Solar ECEF vectors */
    IAS_PROJECTION projection;  /* Ground reference information */
    double earth_sun_distance;  /* Earth to sun distance */
    double wgs84_major_axis;    /* WGS 84 ellipsoid semi-major axis (meters) */
    double wgs84_minor_axis;    /* WGS 84 ellipsoid semi-minor axis (meters) */
    char datum[IAS_DATUM_SIZE]; /* Projection datum string */
    char units[IAS_UNITS_SIZE]; /* Projection units string */
    IAS_CORNERS corners;        /* Projection corners */
    char spacecraft_id[IAS_ANGLE_GEN_SPACECRAFT_SIZE]; /* Spacecraft ID */
    char landsat_scene_id[IAS_ANGLE_GEN_SCENE_ID_LENGTH + 1]; /* Scene ID */
    IAS_GEO_PROJ_TRANSFORMATION *transformation; /* Projection transformation */
} IAS_ANGLE_GEN_METADATA;

/**********************  FUNCTION PROTOTYPES START  ***************************/

int ias_angle_gen_calculate_angles_rpc
(
    const IAS_ANGLE_GEN_METADATA *metadata, /* I: Metadata structure */
    double l1t_line,        /* I: Output space line coordinate */
    double l1t_samp,        /* I: Output space sample coordinate */
    const double *elev,     /* I: Pointer to input elevation or NULL if mean
                              scene height should be used*/
    int band_index,         /* I: Current band index */
    IAS_ANGLE_GEN_TYPE sat_or_sun_type,     /* I: Angle calculation type */
    int *outside_image_flag,/* O: Flag indicating return was outside image */
    double *angle           /* O: Array containing zenith and azimuth angles */
);

void ias_angle_gen_free
(
    IAS_ANGLE_GEN_METADATA *metadata /* I: Metadata structure */
);

int ias_angle_gen_read_ang
(
    const char *ang_filename,        /* I: Angle file name to read */
    IAS_ANGLE_GEN_METADATA *metadata /* O: Metadata structure to load */
);

int ias_angle_gen_write_image
(
    const char *image_filename, /* I: Image file name */
    const short *azimuth,       /* I: Array of azimuth angles */
    const short *zenith,        /* I: Array of zenith angles */
    IAS_ANGLE_GEN_TYPE sat_or_sun_type, /* I: Image type to write */
    int band_index,             /* I: Output band index */
    int num_lines,              /* I: Number of image lines */
    int num_samps,              /* I: Number of image samples */
    IAS_DBL_XY ul_corner,       /* I: Image upper left corner */
    double pixel_size,          /* I: Image pixel size */
    const IAS_PROJECTION *projection /* I: Image framing information */
); 

#endif
