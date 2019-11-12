#ifndef IAS_MISCELLANEOUS_H
#define IAS_MISCELLANEOUS_H
#include "ias_structures.h"
#include "ias_types.h"
#include "ias_const.h"
#include "ias_geo.h"
#include "ias_math.h"
typedef struct ias_misc_line_extent
{
    int start_sample;
    int end_sample;
} IAS_MISC_LINE_EXTENT;
IAS_MISC_LINE_EXTENT *ias_misc_create_output_image_trim_lut
(
    const double *line,         /* I: Line trim box(ul, ur, lr, ll) */
    const double *samp,         /* I: Sample trim box(ul, ur, lr, ll) */
    int output_lines,           /* I: Number of lines */
    int output_samples          /* I: Number of samples */
);
int ias_misc_write_envi_header
(
    const char *image_filename, /* I: Full path name of the image file */
    const IAS_PROJECTION *proj_info, /* I: Optional projection info, set to 
                                           NULL if not known or needed */
    const char *description,    /* I: Optional description, set to NULL if not 
                                      known or needed */
    int lines,                  /* I: Number of lines in the data */
    int samples,                /* I: Number of samples in the data */
    int bands,                  /* I: Number of bands in the data */
    double upper_left_x,        /* I: Optional upper-left X coordinate, set to 
                                      0.0 if not known or needed (requires
                                      proj_info) */
    double upper_left_y,        /* I: Optional upper-left Y coordinate, set to 
                                      0.0 if not known or needed (requires
                                      proj_info) */
    double projection_distance_x, /* Optional pixel size in X projection, set
                                     to 0.0 if not known or needed (requires
                                     proj_info) */
    double projection_distance_y, /* Optional pixel size in Y projection, set 
                                     to 0.0 if not known or needed (requires
                                     proj_info) */
    const char *band_names,     /* I: Optional single string for all band names,
                                      set to NULL if not known or needed */
    IAS_DATA_TYPE data_type     /* I: The IAS type of the data */
);
char *ias_misc_convert_to_uppercase 
(
    char *string_ptr  /* I/O: pointer to string to convert */
);
#endif
