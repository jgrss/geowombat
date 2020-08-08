#ifndef _IAS_STRUCTURES_H_
#define _IAS_STRUCTURES_H_


/* A data structure for decomposed date/time information */
typedef struct IAS_DATETIME
{
    int    year;
    int    month;
    int    day_of_month;
    int    day_of_year;
    int    hour;
    int    minute;
    double second;            /* Allows for fractional seconds */
}   IAS_DATETIME;

typedef struct IAS_DBL_XY
{
    double x;           /* X value                           */
    double y;           /* Y value                           */
} IAS_DBL_XY;

typedef struct IAS_VECTOR
{
    double x;           /* Vector X component */
    double y;           /* Vector Y component */
    double z;           /* Vector Z component */
} IAS_VECTOR;

typedef struct IAS_FLOAT_VECTOR
{
    float x;
    float y;
    float z;
}IAS_FLOAT_VECTOR;

typedef struct IAS_QUATERNION
{
    IAS_VECTOR vector;
    double scalar;
}IAS_QUATERNION;

typedef struct IAS_COMPLEX
   {
   double re;
   double im;
   } IAS_COMPLEX;

typedef struct IAS_LNG_XY
{
  int x;                  /* X value                           */
  int y;                  /* Y value                           */
} IAS_LNG_XY;

typedef struct IAS_LNG_HW
{
  int height;             /* Height value                      */
  int width;              /* Width value                       */
} IAS_LNG_HW;

typedef struct IAS_LNG_LS
{
  int line;               /* Line value                        */
  int samp;               /* Sample value                      */
} IAS_LNG_LS;

typedef struct IAS_DBL_LS
{
  double line;             /* Line value                        */
  double samp;             /* Sample value                      */
} IAS_DBL_LS;

typedef struct IAS_DBL_LAT_LONG
{
  double lat;              /* Latitude value                    */
  double lng;              /* Longitude value                   */
} IAS_DBL_LAT_LONG;

#define COEFS_SIZE 4
typedef struct IAS_COEFFICIENTS
{
  double a[COEFS_SIZE];    /* Array of a coefficients           */
  double b[COEFS_SIZE];    /* Array of b coefficients           */
} IAS_COEFFICIENTS;

typedef struct IAS_CORNERS
{
  IAS_DBL_XY upleft;    /* X/Y value of the upper left corner  */
  IAS_DBL_XY upright;   /* X/Y value of the upper right corner */
  IAS_DBL_XY loleft;    /* X/Y value of the lower left corner  */
  IAS_DBL_XY loright;   /* X/Y value of the lower right corner */
} IAS_CORNERS;

typedef struct IAS_IMAGE
{
  int data_type;
  int nl;
  int ns;
  int band_number;
  void *data; 
  double pixel_size_x;
  double pixel_size_y;
  IAS_CORNERS corners;
} IAS_IMAGE;

/* Polygon segment:  a group of line segments, typically representing a
   portion of a polygon;  mostly used to break down polygons with a large
   number of vertices */
typedef struct ias_polygon_segment
{
    unsigned int first_point;   /* Indice identifying segments first point */
    unsigned int last_point;    /* Indice identifying segments last point */
    double min_x;               /* Minimum x bounds */
    double max_x;               /* Maximum x bounds */
    double min_y;               /* Minimum y bounds */
    double max_y;               /* Maximum y bounds */
} IAS_POLYGON_SEGMENT;

typedef struct ias_polygon_linked_list
{
    unsigned int id;                     /* Polygon id */
    unsigned int num_points;             /* Number of polygon vertices */
    double *point_x;                     /* Polygon vertex x values */
    double *point_y;                     /* Polygon vertex y values */
    double min_x;                        /* Minimum x bounds */
    double max_x;                        /* Maximum x bounds */
    double min_y;                        /* Minimum y bounds */
    double max_y;                        /* Maximum y bounds */
    unsigned int num_segs;               /* Number of polygon segment groups */
    IAS_POLYGON_SEGMENT *poly_seg;       /* Array of polygon segment groups */
    struct ias_polygon_linked_list *prev;/* Pointer to previous polygon */
    struct ias_polygon_linked_list *next;/* Pointer to next polygon */
    struct ias_polygon_linked_list *child;/* Pointer to linked list of children 
                                             (polygons within this polygon) */
} IAS_POLYGON_LINKED_LIST;

typedef struct ias_epoch_time
{
    double year;   /* Year of epoch time */
    double day;    /* Day of the year */
    double seconds;/* Seconds of the day */
} IAS_EPOCH_TIME;

#endif
