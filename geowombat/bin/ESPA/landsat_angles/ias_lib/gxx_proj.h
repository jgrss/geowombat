#ifndef GXX_PROJ_H
#define GXX_PROJ_H

/* Define a zone number for all projections except State Plane and UTM
   NOTE: this define is specific to gps and therefore was not added to proj.h
   or cproj.h which are both files from the gctpc software package */

#define NULLZONE 62

/* Define values normally read from GCTP's proj.h. */
#ifndef GEO
#define GEO 0
#endif

#ifndef UTM
#define UTM 1
#endif

#ifndef PS
#define PS 6
#endif

#ifndef AEA
#define AEA 3
#endif

#define UNITS_SIZE 12
#define DATUM_SIZE 16
#define PROJPRMS_SIZE 15

typedef struct gxx_projection_TYPE
{
    /* ==== Projection parameters and info. ==== */
    char units[UNITS_SIZE];   /* Projection units string */
    int  code;                /* Projection code for the output space image.
                                 Values for this field are defined in the
                                 "gctp.h" include file. */
    char datum[DATUM_SIZE];   /* Projection datum string */
    int spheroid;             /* Projection spheroid code */
    int zone;                 /* Projection zone code for UTM or
                                 State Plane projections. */
    double projprms[PROJPRMS_SIZE];
    /* Array of 15 projection coefficients as required by the projection 
       transformation package.  Refer to the projection package documentation 
       for a description of each field for a given projection. */
} gxx_projection_TYPE;

int gxx_earthradius
(
    double latc,            /* I: Lat to find the radius at (rad) */
    double semi_major_axis, /* I: Semi-major axis */
    double eccentricity,    /* I: Eccentricity */
    double *radius          /* O: Radius of earth at given lat (m)*/
);

int gxx_get_units
(
    const char *unit_name,    /* I: Units name */
    int *unit_num             /* O: Units number */
);

int gxx_projtran
(
    int *inproj,      /* I: Input projection code                     */
    int *inunit,      /* I: Input projection units code               */
    int *inzone,      /* I: Input projection zone code                */
    double *inparm,   /* I: Array of 15 projection parameters--input  */
    int *inspheroid,  /* I: Input spheroid code                       */
    int *outproj,     /* I: Output projection code                    */
    int *outunit,     /* I: Output projection units code              */
    int *outzone,     /* I: Output projection zone code               */
    double *outparm,  /* I: Array of 15 projection parameters--output */
    int *outspheroid, /* I: Output spheroid code                      */
    double *inx,      /* I: Input X projection coordinate             */
    double *iny,      /* I: Input Y projection coordinate             */
    double *outx,     /* O: Output X projection coordinate            */
    double *outy      /* O: Output Y projection coordinate            */
);

#endif
