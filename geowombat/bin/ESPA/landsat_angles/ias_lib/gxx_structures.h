#ifndef _GXX_STRUCTURES_H_
#define _GXX_STRUCTURES_H_

#include "gxx_const.h"

typedef struct DBL_XY
{
    double x;           /* X value                           */
    double y;           /* Y value                           */
} DBL_XY;

typedef struct VECTOR
{
    double x;           /* Vector X component           */
    double y;           /* Vector Y component           */
    double z;           /* Vector Z component           */
} VECTOR;

#define NO_SPACE -1

typedef struct LNG_XY
{
    int x;                  /* X value                           */
    int y;                  /* Y value                           */
} LNG_XY;

typedef struct LNG_HW
{
    int height;             /* Height value                      */
    int width;              /* Width value                       */
} LNG_HW;

typedef struct LNG_LS
{
    int line;               /* Line value                        */
    int samp;               /* Sample value                      */
} LNG_LS;

typedef struct DBL_LS
{
    double line;             /* Line value                        */
    double samp;             /* Sample value                      */
} DBL_LS;

typedef struct DBL_LAT_LONG
{
    double lat;              /* Latitude value                    */
    double lng;              /* Longitude value                   */
} DBL_LAT_LONG;

#define COEFS_SIZE 4

typedef struct COEFFICIENTS
{
    double a[COEFS_SIZE];    /* Array of a coefficients           */
    double b[COEFS_SIZE];    /* Array of b coefficients           */
} COEFFICIENTS;

typedef struct CORNERS
{
    DBL_XY upleft;    /* X/Y value of the upper left corner  */
    DBL_XY upright;   /* X/Y value of the upper right corner */
    DBL_XY loleft;    /* X/Y value of the lower left corner  */
    DBL_XY loright;   /* X/Y value of the lower right corner */
} CORNERS;

typedef struct QUATERNION
{
    VECTOR v;   /* Vector component */
    double w;   /* Scalar component */
} QUATERNION;

/* polygon segment:  a group of line segments, typically representing a
   portion of a polygon;  mostly used to break down polygons with a large
   number of vertices */
typedef struct psegment
{
    unsigned int first_pt; /* indices identifying the first and last points */
    unsigned int last_pt;  /* in the batch of line segments */
    double minx;           /* minimum and maximum bounds for x and y */
    double maxx;
    double miny;
    double maxy;    
} psegment;

typedef struct poly
{
    unsigned int pid;        /* polygon id */
    unsigned int npts;       /* number of polygon vertices */
    double *x;               /* polygon vertex x values */
    double *y;               /* polygon vertex y values */
    double minx;             /* minimum and maximum bounds for x and y */
    double maxx;
    double miny;
    double maxy;
    unsigned int nsegs;      /* number of polygon segment groups */
    psegment *pseg;          /* array of polygon segment groups */
    struct poly *prev;       /* pointer to previous polygon in linked list */
    struct poly *next;       /* pointer to next polygon in linked list */
    struct poly *child;      /* pointer to linked list of children (polygons
                                within this polygon) */
} poly;

/* GPS subsystem-wide definitions */
#define MAX_BANDS 9

#define NPARMS_FIT 6  /* number of precision fit variables */

/* The Ground Control Point structure */
typedef struct
{
    char pt_id[PTIDLEN];/* Ground control point identifier */
    BandNumber bandno;  /* Ground control point band numb of systematic image*/
    int  outflag;       /* Ground control point outlier flag 1 good, 0 bad */
    VECTOR satpos;      /* satellite position of a point */
    VECTOR satvel;      /* satellite velocity of a point */
    VECTOR gcppos;      /* True ground point position of a point in
                           Cartesion space */
    VECTOR pixpos;      /* position of the observed feature calculated by the
                           satellite model in Cartesion space */
    double pred_line;   /* predicted line from model */
    double pred_samp;   /* predicted sample from model */
    double time;        /* Time when GCP was imaged in seconds */
    double lat;         /* true latitude of the GCP */
    double lon;         /* true longitude of the GCP */
    double height;      /* Height of the ground control point */
    double delta;       /* Along-scan look angle (line of sight) in rads */
    double psi;         /* Across-scan look angle (line of sight) in rads */
    double alpha;       /* Along-scan Residual in micro-radians */
    double residual_x;  /* Residual value in ground space for x, meters */
    double residual_y;  /* Residual value in ground space for y, meters */
    double beta;        /* Across-scan Residual in micro-radians */
    double x1Gline;     /* level 1R image line number */
    double x1Gsamp;     /* level 1R image sample number */
    VECTOR satpos0;     /* original satellite position */
    VECTOR satvel0;     /* original satellite velocity */
    double apd[NPARMS_FIT]; /* GCP alpha partial derivatives */
    double bpd[NPARMS_FIT]; /* GCP beta partial derivatives */
    double delta0;      /* original along-scan look angle */
    double psi0;        /* original cross-scan look angle */
    double corr_coeff;  /* correlation coefficient */
    char source[STRLEN];/* GCP source */
} GCP_STRUCT;

#endif
