#ifndef IAS_GEO_H
#define IAS_GEO_H
typedef struct ias_geo_proj_transformation IAS_GEO_PROJ_TRANSFORMATION;
typedef struct ias_projection
{
    int proj_code;      /* Projection code */
    int zone;           /* Projection zone number - only has meaning for
                           projections like UTM and stateplane */
    int units;          /* Units of coordinates */
    int spheroid;       /* Spheroid code for the projection */
    double parameters[IAS_PROJ_PARAM_SIZE];
                        /* Array of projection parameters */
} IAS_PROJECTION;
int ias_geo_convert_dms2deg
(
    double angle_dms,     /* I: Angle in DMS (DDDMMMSSS) format */
    double *angle_degrees,/* O: Angle in decimal degrees */
    const char *type      /* I: Angle usage type (LAT, LON, NOLIMIT, 
                                or DEGREES) */
);
#endif
