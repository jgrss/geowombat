#ifndef _CREATE_ANGLE_BANDS_H_
#define _CREATE_ANGLE_BANDS_H_

#include "gxx_structures.h"

/* Used as median between API and main routine for the band metadata needed to
    write the image to file */
typedef struct angle_frame
{
    int band_number;                /* Band number */
    int num_lines;                  /* Number of lines in frame */
    int num_samps;                  /* Number of samples in frame */
    DBL_XY ul_corner;               /* Upper left corner coordinates */
    double parms[PROJPRMS_SIZE];    /* Projection coefficients */
    double pixel_size;              /* Pixel size in meters */
    gxx_projection_TYPE projection; /* Projection information */
} angle_frame_TYPE;

#endif
