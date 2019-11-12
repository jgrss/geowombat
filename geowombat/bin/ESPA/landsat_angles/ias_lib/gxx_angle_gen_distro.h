#ifndef _GXX_ANGLE_GEN_DISTRO_H_
#define _GXX_ANGLE_GEN_DISTRO_H_

/* IAS Library Includes */
#include "gxx_structures.h"     /* VECTOR define */
#include "gxx_geo_math.h"
#include "gxx_proj.h"
#include "xxx_Band.h"           /* MAXBANDS */
#include "gxx_coord_conv.h"

/* Local Defines */
#define PROGRAM "Angle Gen"
#define MODE_SIZE 20
#define NUMBER_SCAN_DIRECTIONS 2
#define WGS84_SPHEROID 12
#define MAX_RPC NUMBER_SCAN_DIRECTIONS
#define IAS_ANGLE_GEN_NUM_RPC_COEF 5     /* Number of image RPC coefficients */
#define IAS_ANGLE_GEN_ANG_RPC_COEF 10    /* Number of angle RPC coefficients */
#define IAS_ANGLE_GEN_SPACECRAFT_SIZE 32 /* Spacecraft name size */
#define IAS_ANGLE_GEN_SCENE_ID_LENGTH 21 /* Scene ID length */
#define IAS_ANGLE_GEN_ZENITH_INDEX 0    /* Array index for the zenith angle */
#define IAS_ANGLE_GEN_AZIMUTH_INDEX 1   /* Array index for the azimuth angle */
#define SCAN_TIME_POLY_NCOEFF 4

typedef enum gxx_angle_gen_TYPE
{
    GXX_ANGLE_GEN_UNKNOWN = 0,
    GXX_ANGLE_GEN_SATELLITE,
    GXX_ANGLE_GEN_SOLAR
} gxx_angle_gen_TYPE;

typedef enum gxx_scan_direction_TYPE
{
    no_scan_direction=-1,
    first_scan_direction,
    second_scan_direction,
    both_scan_direction
} gxx_scan_direction_TYPE;

typedef struct gxx_epoch_time_TYPE
{
    double year;   /* Year of epoch time */
    double day;    /* Day of the year */
    double seconds;/* Seconds of the day */
} gxx_epoch_time_TYPE;

/* Subset of rational polynomial coefficient terms used to Map the L1T
   line/sample to the L1R line/sample */
typedef struct gxx_angle_gen_image_rpc_terms_TYPE
{
    double l1r_mean_offset;                             /* L1R mean offset */
    double l1t_mean_offset;                             /* L1T mean offset */
    double numerator[IAS_ANGLE_GEN_NUM_RPC_COEF];       /* Numerator coeffs */
    double denominator[IAS_ANGLE_GEN_NUM_RPC_COEF - 1]; /* Denominator coeffs */
} gxx_angle_gen_image_rpc_terms_TYPE;

/* Rational polynomial coefficients that map the L1T line/sample to the
   L1R line/sample. */
typedef struct gxx_angle_gen_image_rpc_TYPE
{
    int direction;                            /* Number of scan directions */
    double mean_height;                       /* Mean height offset */
    gxx_angle_gen_image_rpc_terms_TYPE line_terms; /* Line terms */
    gxx_angle_gen_image_rpc_terms_TYPE samp_terms; /* Sample terms */
} gxx_angle_gen_image_rpc_TYPE;

/* Numerator and denominator rational polynomial coefficient terms used 
   in calculating the satellite viewing angle or solar illumination azimuth 
   and zenith angles for a specified L1T line/sample. */ 
typedef struct gxx_angle_gen_ang_rpc_terms_TYPE
{
    double numerator[IAS_ANGLE_GEN_ANG_RPC_COEF];      /* Numerator coeffs */
    double denominator[IAS_ANGLE_GEN_ANG_RPC_COEF - 1];/* Denominator coeffs */
} gxx_angle_gen_ang_rpc_terms_TYPE;

/* Offset rational polynomial coefficient terms used in calculating the
   satellite viewing angle or solar illumination azimuth and zenith angles for
   a specified L1T line/sample. */ 
typedef struct gxx_angle_gen_ang_offset_terms_TYPE
{
    double l1r_mean_offset;  /* L1R mean offset */
    double l1t_mean_offset;  /* L1T mean offset */
} gxx_angle_gen_ang_offset_terms_TYPE;

/* Rational polynomial coefficients used in calculating the satellite viewing
   angle or solar illumination azimuth and zenith angles for a specified L1T
   line/sample. */ 
typedef struct gxx_angle_gen_ang_rpc_TYPE
{
    double mean_height;                  /* Mean height offset */
    VECTOR mean_offset;              /* Mean vector offsets */
    gxx_angle_gen_ang_offset_terms_TYPE line_terms; /* Line offset coeffs */
    gxx_angle_gen_ang_offset_terms_TYPE samp_terms; /* Sample offset coeffs */
    gxx_angle_gen_ang_rpc_terms_TYPE x_terms; /* X axis coefficients */
    gxx_angle_gen_ang_rpc_terms_TYPE y_terms; /* Y axis coefficients */
    gxx_angle_gen_ang_rpc_terms_TYPE z_terms; /* Z axis coefficients */
} gxx_angle_gen_ang_rpc_TYPE;

/* All of the rational polynomial coefficients and metadata for a band needed
   to calculate the azimuth and zenith angles and map L1T line/sample to L1R
   line/sample */
typedef struct gxx_angle_gen_band_TYPE
{
    BandNumber band_number; /* User band number */
    int lines_per_scan;     /* Number of lines per scan */
    int number_scan_dirs;   /* Scan direction */
    int l1t_lines;          /* Number of lines in the L1T image. */
    int l1t_samps;          /* Number of samples in the L1T image. */
    int l1r_lines;          /* Number of lines in the L1R image. */
    int l1r_samps;          /* Number of samples in one L1R SCA. */
    double pixel_size;      /* Projection distance per pixel in meters. */
    double image_start_time;/* Image start time, offset in seconds from
                               ephemeris epoch */
    double seconds_per_line;/* L1R line time increment in seconds */
    double active_l1t_corner_lines[4]; /* UL, UR, LL, LR, UL corner lines */
    double active_l1t_corner_samps[4]; /* UL, UR, LL, LR, UL corner samps */
    gxx_angle_gen_ang_rpc_TYPE satellite;   /* Satellite viewing angles */
    gxx_angle_gen_ang_rpc_TYPE solar;       /* Solar angles */
    gxx_angle_gen_image_rpc_TYPE scan_metadata[MAX_RPC]; /* Per direction RPC 
                                                           coefficient 
                                                           structures */
} gxx_angle_gen_band_TYPE;

/* Sample time and position for each ephemeris or solar vector point */
typedef struct gxx_angle_gen_ephemeris_TYPE
{
    double sample_time;   /* Sample time from ephemeris epoch */
    VECTOR position;      /* Position vector */
} gxx_angle_gen_ephemeris_TYPE;

/* Type defines for projection related structures */
typedef struct gxx_proj_transformation gxx_PROJ_TRANSFORMATION;

typedef struct gxx_angle_gen_scan_time_TYPE
{
    unsigned int ncoeff;
    unsigned int number_scan_dirs;          /* Number of scan directions */
    int direction[NUMBER_SCAN_DIRECTIONS];  /* Scan direction */
    double mean_activescan[NUMBER_SCAN_DIRECTIONS]; /* Mean of active scan 
                                                       times for a given 
                                                       direction */
    double mean_eol[NUMBER_SCAN_DIRECTIONS];/* Mean of end of line length */
    /* Polynomial fit to scan times */
    double scan_time_poly[NUMBER_SCAN_DIRECTIONS][SCAN_TIME_POLY_NCOEFF];
    double mean_scan_times[NUMBER_SCAN_DIRECTIONS];
} gxx_angle_gen_scan_time_TYPE;

/* The main angle metadata structure used to contain all the information 
   need to generate the azimuth and zenith angles and map the L1T line/sample
   to the L1R line/sample for any band/sca combination */
typedef struct gxx_angle_gen_metadata_TYPE
{
    int wrs_path;               /* WRS path number (target) */
    int wrs_row;                /* WRS row number (target) */
    char mode[MODE_SIZE];       /* Bumper and SLC mode */
    char first_scan_direction;  /* Direction of first scan */
    unsigned int num_bands;     /* Number of bands in the metadata */
    gxx_angle_gen_band_TYPE band_metadata[MAXBANDS]; /* Band metadata */
    int band_present[MAXBANDS]; /* Flag to determine if band present */
    gxx_epoch_time_TYPE ephem_epoch_time;/* Ephemeric epoch time */
    unsigned int ephem_count;   /* Number of ephemeris samples */
    gxx_angle_gen_ephemeris_TYPE *ephemeris;    /* Ephemeris data */
    gxx_angle_gen_ephemeris_TYPE *solar_vector; /* Solar ECEF vectors */
    gxx_angle_gen_scan_time_TYPE scan_time; /* Fit of scan times */
    gxx_projection_TYPE projection;  /* Ground reference information */
    double earth_sun_distance;  /* Earth to sun distance */
    double wgs84_major_axis;    /* WGS 84 ellipsoid semi-major axis (meters) */
    double wgs84_minor_axis;    /* WGS 84 ellipsoid semi-minor axis (meters) */
    char datum[DATUM_SIZE];     /* Projection datum string */
    char units[UNITS_SIZE];     /* Projection units string */
    CORNERS corners;            /* Projection corners */
    char spacecraft_id[IAS_ANGLE_GEN_SPACECRAFT_SIZE]; /* Spacecraft ID */
    char landsat_scene_id[IAS_ANGLE_GEN_SCENE_ID_LENGTH + 1]; /* Scene ID */
    gxx_PROJ_TRANSFORMATION *transformation; /* Projection transformation */
} gxx_angle_gen_metadata_TYPE;

/**********************  FUNCTION PROTOTYPES START  ***************************/

int gxx_angle_gen_calculate_angles_rigor
(
    gxx_angle_gen_metadata_TYPE *metadata,/* I: Metadata structure */
    double l1t_line,    /* I: Output space line coordinate */
    double l1t_samp,    /* I: Output space sample coordinate */
    const double *elev, /* I: Pointer to input elevation or NULL if
                              mean scene height should be used */
    int band_index,     /* I: Current band index */
    double scan_buffer, /* I: scan buffer */
    int subsamp,        /* I: sub sample factor */
    gxx_angle_gen_TYPE sat_or_sun_type, /* I: Angle calculation type */
    int *outside_image_flag,/* O: Flag indicating return was outside image */
    double *angle       /* O: Array containing zenith and azimuth angles */
);

int gxx_angle_gen_calculate_angles_rpc
(
    const gxx_angle_gen_metadata_TYPE *metadata, /* I: Metadata structure */
    double l1t_line,        /* I: Output space line coordinate */
    double l1t_samp,        /* I: Output space sample coordinate */
    const double *elev,     /* I: Pointer to input elevation or NULL if mean
                              scene height should be used*/
    int band_index,         /* I: Current band index */
    double scan_buffer,     /* I: Scan buffer */
    int subsamp,            /* I: Sub sample factor */
    gxx_angle_gen_TYPE sat_or_sun_type,     /* I: Angle calculation type */
    int *outside_image_flag,/* O: Flag indicating return was outside image */
    double *angle           /* O: Array containing zenith and azimuth angles */
);

void gxx_angle_gen_free
(
    gxx_angle_gen_metadata_TYPE *metadata /* I: Metadata structure */
);

int gxx_angle_gen_read_ang
(
    char *ang_filename,                   /* I: Angle file name to read */
    gxx_angle_gen_metadata_TYPE *metadata /* O: Metadata structure to load */
);

int gxx_angle_gen_write_image
(
    const char *image_filename, /* I: Image file name */
    const short *azimuth,       /* I: Array of azimuth angles */
    const short *zenith,        /* I: Array of zenith angles */
    gxx_angle_gen_TYPE sat_or_sun_type, /* I: Image type to write */
    int band_index,             /* I: Output band index */
    int band_number,            /* I: Name of the band */
    int num_lines,              /* I: Number of image lines */
    int num_samps,              /* I: Number of image samples */
    DBL_XY ul_corner,           /* I: Image upper left corner */
    double pixel_size,          /* I: Image pixel size */
    const gxx_projection_TYPE *projection /* I: Image framing information */
);

int gxx_angle_gen_find_dir
(
    double l1t_line,                /* I: L1T line */
    double l1t_samp,                /* I: L1T sample */
    double height,                  /* I: height */
    double scan_buffer,             /* I: scan buffer */
    int subsamp,                    /* I: sub sample factor */
    const gxx_angle_gen_band_TYPE *eband, /* I/O: metadata current band */
    double              *l1r_line,  /* O: Array of output L1R line numbers */
    double              *l1r_samp,  /* O: Array of output L1R sample numbers */
    int                 *num_dir_found, /* O: Number of directions found */
    gxx_scan_direction_TYPE *scan_dir /* O: Scan direction found */
);

#endif
