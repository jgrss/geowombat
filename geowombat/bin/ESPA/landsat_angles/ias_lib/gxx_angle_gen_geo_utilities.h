#ifndef GXX_ANGLE_GEN_GEO_UTILITIES_H
#define GXX_ANGLE_GEN_GEO_UTILITIES_H

#include <math.h>
#include "xxx_Types.h"
#include "xxx_LogStatus.h"
#include "gxx_const.h"
#include "gxx_geo_math.h"
#include "gxx_angle_gen_distro.h"

void gxx_angle_gen_geo_transform_polar_motion_true_pole_to_mean
(
    const VECTOR *r_old, /* I: coordinates (x, y, z) in the old system */
    double xp,    /* I: true pole position in the mean pole coords system, 
                     x-axis pointing along Greenwich meridian; in arc seconds */
    double yp,    /* I: true pole position in the mean pole coords system, 
                     y-axis pointing along west 90 degree meridian; in arc 
                     seconds */
    double jd_tdb,/* I: Julian date (Barycentric) */
    VECTOR *r_new /* O: coordinates in the new system */
);

void gxx_angle_gen_geo_transform_nutation_mod2tod
(
    const VECTOR *r_old,/* I: coordinates (x, y, z) in the mean-of-date system */
    double jd_tdb,      /* I: Julian date (Barycentric) for conversion */
    VECTOR *r_new       /* O: coordinates in the true-of-date equator and  
                              equinox sys. */
);

int gxx_angle_gen_geo_transform_precession_j2k2mod
(
    const VECTOR *r_old,/* I: coordinates (x, y, z) in the J2000.0 system */
    double jd_tdb,      /* I: Julian date (Barycentric) for conversion */
    VECTOR *r_new       /* O: coordinates in the mean-of-date equator and 
                              equinox sys. */
);

int gxx_angle_gen_geo_transform_precession_mod2j2k
(
    const VECTOR *r_old,    /* I: coordinates (x, y, z) in the mean-of-date
                                  system */
    double jd_tdb,          /* I: Julian date (Barycentric) for conversion */
    VECTOR *r_new           /* O: coordinates in the J2000.0 system */
);

int gxx_angle_gen_geo_eci2ecef
(
    double xp, /* I: Earth's true pole offset from mean pole, in arc second */
    double yp, /* I: Earth's true pole offset from mean pole, in arc second */
    double ut1_utc, /* I: UT1-UTC, in seconds, due to variation of Earth's spin 
                          rate */
    const VECTOR *craft_pos, /* I: Satellite position in ECI */
    const VECTOR *craft_vel, /* I: Satellite velocity in ECI */
    const double ephem_time[3],  /* I: UTC Ephemeris time (year, doy and sod) */
    VECTOR *fe_satpos, /* O: Satellite position in ECEF */
    VECTOR *fe_satvel  /* O: Satellite velocity in ECEF */
);

void gxx_angle_gen_geo_to_ecef
(
    const gxx_angle_gen_metadata_TYPE *metadata, /* I: Projection information */
    double latitude,                        /* I: Latitude in radians */
    double longitude,                       /* I: Longitude in radians */
    double height,                          /* I: Height in meters */
    VECTOR *ecef_vector                 /* O: ECEF vector */
);

int gxx_geo_transform_coordinate
(
    gxx_projection_TYPE in_proj, /* I: Projection Information */
    double proj_x,               /* I: projection X coordinate */
    double proj_y,               /* I: projection Y coordinate */
    double *lat,                 /* O: latitude (radians) */
    double *lon                  /* O: longitude (radians)*/
);

int geo_transform_tod2j2k
(
    double ut1_utc,              /* I: UT1-UTC, in seconds, due to variation
                                       of Earth's spin rate */
    const VECTOR *ecitod_pos,    /* I: Satellite position in ECITOD */
    const double ephem_time[3],  /* I: UTC Ephemeris time (year, doy and sod) */
    VECTOR *ecij2k_pos           /* O: Satellite position in ECIJ2K */
);

#endif
