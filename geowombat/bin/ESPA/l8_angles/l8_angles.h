#ifndef _L8_ANGLES_H_
#define _L8_ANGLES_H_

/* Standard Library Includes */
#include <limits.h>

/* IAS Library Includes */    
#include "ias_angle_gen_distro.h"
#include "ias_miscellaneous.h"

/* Angle type to calculate */
typedef enum angle_type
{
    AT_UNKNOWN = 0, /* Unknown angle type */
    AT_BOTH,        /* Both Solar and Satellite angle types */
    AT_SATELLITE,   /* Satellite angle type */
    AT_SOLAR        /* Solar angle type */
} ANGLE_TYPE;

/* This struct is used as a transport median from the API and the band meta data
   to the L8 Angles application and the write image routine. */
typedef struct angles_frame
{
    int band_number;        /* Band number */
    int num_lines;          /* Number of lines in frame */
    int num_samps;          /* Number of samples in frame */
    IAS_DBL_XY ul_corner;   /* Upper left corner coordinates */
    double pixel_size;      /* Pixel size in meters */
    IAS_PROJECTION projection; /* Projection information */ 
} ANGLES_FRAME;

/* Holds all the information needed to run L8 Angles */
typedef struct l8_angles_parameters
{
    int process_band[IAS_MAX_NBANDS]; /* Process bands array */
    char metadata_filename[PATH_MAX]; /* Metadata filename */
    int use_dem_flag;            /* Flag used to check if DEM should be used */
    ANGLE_TYPE angle_type;       /* Type of angles to be generated */
    int sub_sample_factor;       /* Sub-sampling factor to be used */
    short background;            /* Background value used for fill pixels */
} L8_ANGLES_PARAMETERS;

/***************************** PROTOTYPES START *******************************/
int calculate_angles
(
    const IAS_ANGLE_GEN_METADATA *metadata, /* I: Angle metadata structure */ 
    int line,                               /* I: L1T line coordinate */
    int samp,                               /* I: L1T sample coordinate */
    int band_index,                         /* I: Spectral band number */
    ANGLE_TYPE angle_type,                  /* I: Type of angles to generate */
    double *sat_angles,                     /* O: Satellite angles */
    double *sun_angles                      /* O: Solar angles */
);

const double *get_active_lines
(
    const IAS_ANGLE_GEN_METADATA *metadata, /* I: Angle metadata structure */ 
    int band_index                          /* I: Band index */
);

const double *get_active_samples
(
    const IAS_ANGLE_GEN_METADATA *metadata, /* I: Angle metadata structure */ 
    int band_index                          /* I: Band index */
);

int get_frame
(
    const IAS_ANGLE_GEN_METADATA *metadata, /* I: Angle metadata structure */ 
    int band_index,                         /* I: Band index */
    ANGLES_FRAME *frame                     /* O: Image frame info */
);               

int read_parameters
(
    const char *parameter_filename,       /* I: Parameter file name */
    L8_ANGLES_PARAMETERS *parameters      /* O: Generate Chips parameters */
);

#endif
