#ifndef IAS_ANGLE_GEN_PRIVATE_H
#define IAS_ANGLE_GEN_PRIVATE_H

/* Local Library Includes */
#include "ias_angle_gen_distro.h" /* Angle gen structs */

int ias_angle_gen_calculate_vector
(
    const IAS_ANGLE_GEN_METADATA *metadata,/* I: Metadata structure */
    double l1t_line,         /* I: Current L1T line number */
    double l1t_samp,         /* I: Current L1T sample number */
    double l1r_line,         /* I: Current L1R line number */
    double height,           /* I: Current L1T height */
    int band_index,          /* I: Current band index */
    IAS_ANGLE_GEN_TYPE sat_or_sun_type, /* I: Angle type */
    IAS_VECTOR *view         /* O: View vector */
);

int ias_angle_gen_find_scas
(
    const IAS_ANGLE_GEN_BAND *metadata,/* I: Metadata for current band */
    double l1t_line,      /* I: Input L1T line */
    double l1t_samp,      /* I: Input L1T sample */
    const double *height, /* I: Input height, NULL for zero height */
    double *l1r_line,     /* O: Array of output L1R line numbers */
    double *l1r_samp      /* O: Array of output L1R sample numbers */
);

int ias_angle_gen_interpolate_ephemeris
(
    const IAS_ANGLE_GEN_EPHEMERIS *ephemeris,/* I: Metadata ephemeris points */
    int ephem_count,   /* I: Number of entries in ephem and time */
    double in_time,    /* I: Time from epoch to interpolate */
    IAS_VECTOR *vector /* O: Output interpolated vector */
);

int ias_angle_gen_initialize
(
    IAS_ANGLE_GEN_METADATA *metadata /* O: Angle metadata struct */
);

double ias_angle_gen_valid_band_index
(
    const IAS_ANGLE_GEN_METADATA *metadata, /* I: Angle metadata */
    int band_index                          /* I: Band index to check */
);

#endif
