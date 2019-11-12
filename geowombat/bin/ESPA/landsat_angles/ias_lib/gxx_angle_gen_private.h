#ifndef GXX_ANGLE_GEN_PRIVATE_H
#define GXX_ANGLE_GEN_PRIVATE_H

/* Local Library Includes */
#include "gxx_angle_gen_distro.h" /* Angle gen structs */

int gxx_angle_gen_calculate_vector
(
    gxx_angle_gen_metadata_TYPE *metadata,/* I: Metadata structure */
    double l1t_line,         /* I: Current L1T line number */
    double l1t_samp,         /* I: Current L1T sample number */
    double l1r_line,         /* I: Current L1R line number */
    double l1r_samp,         /* I: Current L1R sample number */
    double height,           /* I: Current L1T height */
    unsigned int band_index, /* I: Current band index */
    unsigned int dir,        /* I: scan direction */
    gxx_angle_gen_TYPE sat_or_sun_type, /* I: Angle type */
    VECTOR *view         /* O: View vector */
);

int gxx_angle_gen_find_scas
(
    const gxx_angle_gen_metadata_TYPE *metadata, /* I: Metadata for current
                                                       band */
    double l1t_line,      /* I: Input L1T line */
    double l1t_samp,      /* I: Input L1T sample */
    const double *height, /* I: Input height, NULL for zero height */
    double *l1r_line,     /* O: Array of output L1R line numbers */
    double *l1r_samp      /* O: Array of output L1R sample numbers */
);

int gxx_angle_gen_initialize_transformation
(
    gxx_angle_gen_metadata_TYPE *metadata,  /* I/O: Angle metadata struct */
    double in_x,                            /* I: X coordinate */
    double in_y,                            /* I: y coordinate */
    double *out_x,                          /* O: x coordinate */
    double *out_y                           /* O: y coordinate */
);

int gxx_angle_gen_interpolate_ephemeris
(
    const gxx_angle_gen_ephemeris_TYPE *ephemeris, /* I: Metadata ephemeris
                                                         points */
    unsigned int ephem_count, /* I: Number of entries in ephem and time */
    double in_time,    /* I: Time from epoch to interpolate */
    VECTOR *vector /* O: Output interpolated vector */
);

int gxx_angle_gen_initialize
(
    gxx_angle_gen_metadata_TYPE *metadata /* O: Angle metadata struct */
);

double gxx_angle_gen_valid_band_index
(
    int band_index                          /* I: Band index to check */
);

#endif
