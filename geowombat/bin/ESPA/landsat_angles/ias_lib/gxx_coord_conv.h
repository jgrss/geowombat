#ifndef GXX_COORD_CONV_H
#define GXX_COORD_CONV_H

#include "gxx_structures.h"

int gxx_cart2sph
(
    const VECTOR *vec, /* I: Vector containing Cartesian coords   */
    double *lat,       /* O: Latitude in spherical coordinates    */
    double *longs,     /* O: Longitude in spherical coordinates   */
    double *radius     /* O: Distance from origin to the point    */
);

int gxx_centh2det
(
    double latc,            /* I: Geocentric latitude (radians)  */
    double radius,          /* I: Radius of the point (meters)   */
    double semi_major_axis, /* I: Semi-major axis */
    double eccentricity,    /* I: Eccentricity */
    double *latd,           /* O: Geodetic latitude (radians)    */
    double *height          /* O: Height of the point (meters)   */
);

int gxx_decdeg
(
    double *angle,    /* I/O: Angle in seconds, minutes, or DMS    */
    char *coform,     /* I: Angle units (SEC, MIN, DMS)            */
    char *type        /* I: Angle usage type (LAT, LON, or DEGREES)*/
);

int gxx_degdms
(
    double *deg,    /* I: Angle in seconds, minutes, or degrees      */
    double *dms,    /* O: Angle converted to DMS                     */
    char *code,     /* I: Angle units (SEC, MIN, DEG, DMS)           */
    char *check     /* I: Angle usage type (LAT, LON, or DEGREES)    */ 
);

int gxx_dmsdeg
(
    double dms,         /* I: Angle in DMS (DDDMMMSSS) format */
    double *deg,        /* O: Angle in decimal degrees */
    char *check         /* I: Angle usage type (LAT, LON, or DEGREES) */
);

int  gxx_det2centh
(
    double latd,            /* I: Geodetic latitude (radians) */
    double height,          /* I: Height in meters (elevation) */
    double semi_major_axis, /* I: Semi-major axis */
    double eccentricity,    /* I: Eccentricity */
    double *latc,           /* O: Geocentric latitude (radians) */
    double *radius          /* O: Satellite radius at point (m) */
);

int gxx_find_deg
(
    double angle,  /* I: Angle in total degrees      */
    int  *degree   /* O: Degree portion of the angle */
);

int gxx_find_min
(
    double angle,  /* I: Angle in total degrees      */
    int  *minute   /* O: Minute portion of the angle */
);

int gxx_find_sec
(
    double angle,   /* I: Angle in total degrees      */
    double *second  /* O: Second portion of the angle */
);

void gxx_geod2cart
(
    double latitude,    /* I: Lat of geodetic coordinates in radians */
    double longitude,   /* I: Long of geodetic coordinates in radians*/
    double height,      /* I: Height (elevation) of geodetic coord in meters*/
    double semimajor,   /* I: Reference ellipsoid semi-major axis in meters */
    double flattening,  /* I: Flattening of the ellipsoid 
                              (semimajor-semiminor)/semimajor    */
    VECTOR *cart        /* O: Cartesian vector for the coord    */
);

int  gxx_sph2cart
(
    double *latp,    /* I: */
    double *longp,   /* I: */
    double *radius,  /* I: */
    VECTOR *vec      /* O: */
);

#endif
