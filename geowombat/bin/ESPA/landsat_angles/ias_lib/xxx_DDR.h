#ifndef XXX_DDR_H
#define XXX_DDR_H

#include <stdlib.h>

#define XXX_DDR_EXTENSION       "DDR"   /* DDR  extension where HDF is in HDF
                                           name */

#define XXX_DDR_FIELD_SIZE      500 /* latest strlen on fields was 497 */
#define XXX_DATUM_CODE_SIZE     16
#define XXX_PROJECTION_UNITS_SIZE   12
#define XXX_PROJECTION_COEF_SIZE    15  
#define XXX_POINT_SIZE          2
#define XXX_SOURCE_SIZE         32
#define XXX_INSTRUMENT_SIZE     32
#define XXX_DIRECTION_SIZE      64
#define XXX_CAPTURE_DATE            11
#define XXX_CAPTURE_TIME        8
#define XXX_VALID_FLAGS         8

typedef enum
{
    xxx_Systematic = 1,
    xxx_Precision  = 2,
    xxx_Terrain    = 3
} 
xxx_CorrectionType_TYPE;

/* Valid CaptureDirection field values */
#define XXX_DIRECTION_ASCEND        "ascending"
#define XXX_DIRECTION_DESCEND       "descending"

typedef struct xxx_DDR_TYPE
{
    char BandNumber;              /* Band number */
    short NumberScansInScene;     /* Number of scans in the scene */
    char NumberdetectorsPerScan;  /* Number of detectors per scan */
    int NumberLinesInScene;       /* Number of lines in the scene */
    int NumberSamplesInScene;     /* Number of samples in the scene */
    short DataType;               /* unit8, int16, etc.*/
    double MaxPixelValue;         /* Maximum pixel value */
    double MinPixelValue;         /* Minimum pixel value */
    double MaxRadianceValue;      /* Maximum radiance value */
    double MinRadianceValue;      /* Minimum radiance value */
    int MasterLine;               /* Line relative to original image */
    int MasterSample;             /* Sample relative to original image */
    short ProjectCode;            /* GCTP project code */
    short ZoneCode;               /* UTM zone */
    char DatumCode[XXX_DATUM_CODE_SIZE+1]; /* GCTP datum code */
    int SpheroidCode;                      /* Spheroid code, GPS only */
    char ProjectionUnits[XXX_PROJECTION_UNITS_SIZE+1];  /* Projection units */
    double ProjectionCoef[XXX_PROJECTION_COEF_SIZE];    /* GCTP projection
                                                           parameters */
    double UpperLeft[XXX_POINT_SIZE];  /* Upper left Y (lat), x (long) */
    double LowerLeft[XXX_POINT_SIZE];  /* Lower left Y (lat), x (long) */
    double UpperRight[XXX_POINT_SIZE]; /* Upper right Y (lat), x (long) */
    double LowerRight[XXX_POINT_SIZE]; /* Lower right Y (lat), x (long) */
    double ProjectionDistY;      /* Projection distance per pixel (y) */
    double ProjectionDistX;      /* Projection distance per pixel (x) */
    double LineSubsampleInc;     /* Line subsampleing factor */
    double SampleSubsampleInc;   /* Sample subsampleing factor */
    char SpaceCraftSource[XXX_SOURCE_SIZE+1];     /* Spacecraft source */
    char InstrumentSource[XXX_INSTRUMENT_SIZE+1]; /* Instrument source */
    unsigned char WRSPath;       /* Staring WRS path for the scene, GPS only */
    double WRSRows;  /* Starting fractional row of the WRS scene, GPS only */
    char CaptureDirection[XXX_DIRECTION_SIZE+1]; /* Capture direction */
    char CaptureDate[XXX_CAPTURE_DATE+1];  /* Capture date */
    char CaptureTime[XXX_CAPTURE_TIME+1];  /* Capture time */
    char FieldValidityFlags[XXX_VALID_FLAGS];     /* Field validity flags */
    char CorrectionType;          /* Indicates image type, systematic,
                                     precision, or terrain, GPS only */
} xxx_DDR_TYPE;

/* HDF parameters -- should be kept in sync with the values in hlimits.h
   from the HDF library */
#ifndef VSNAMELENMAX
#define VSNAMELENMAX 64
#endif
#ifndef MAX_PATH_LEN
#define MAX_PATH_LEN 1024
#endif

typedef struct xxx_DDRControl_TYPE
{
    int Count;            /* Number of rows in DDR format */
    xxx_DDR_TYPE *p_DDR;  /* Format DDR */

    /* vdata information about the file */
    char vdataName[VSNAMELENMAX+1];
    char vdataClass[VSNAMELENMAX+1];
    int Interlace;
    int vdataSize;
} xxx_DDRControl_TYPE;

typedef struct xxx_DDRInfo_TYPE
{
    char FieldName[24];
    int Datatype;
    int Order;
} xxx_DDRInfo_TYPE;

#define XXX_FREE_DDR(DDRControl) \
{ \
    if (DDRControl) \
    { \
        free(DDRControl->p_DDR); \
        free(DDRControl); \
    } \
}

xxx_DDRControl_TYPE *xxx_ReadDDR
(
    char *p_PathName,     /* I: Pathname to DDR file */
    char *p_ErrorMessage  /* O: Error message */
);

int xxx_WriteDDR
(
    char *p_PathName,                  /* I: Pathname to DDR file */
    xxx_DDRControl_TYPE *p_DDRControl, /* I: DDR information */
    char *p_ErrorMessage               /* O: Error message */
);

#endif
