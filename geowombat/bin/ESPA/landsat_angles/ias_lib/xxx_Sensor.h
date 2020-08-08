#ifndef XXX_SENSOR_H
#define XXX_SENSOR_H

#ifndef IAS_NO_SENSOR_META_SUPPORT
#include "xxx_MSCD.h"
#include "xxx_PCD.h"
#include "xxx_L0R_Report.h"
#include "xxx_EDCMeta.h"
#include "xxx_LPGSMeta.h"
#include "xxx_LPSMeta.h"
#endif
#include "xxx_Band.h"
#include "xxx_TimeConvert.h"

#define MAX_SSS 10      /* max sensor/satellite string length */

typedef enum IAS_SENSOR_TYPE
{
    IAS_SENSOR_UNKNOWN,
    IAS_SENSOR_MSS,
    IAS_SENSOR_TM,
    IAS_SENSOR_ETM
} IAS_SENSOR_TYPE;

typedef enum IAS_SATELLITE_TYPE
{
    IAS_SAT_UNKNOWN,
    IAS_SAT_L1,
    IAS_SAT_L2,
    IAS_SAT_L3,
    IAS_SAT_L4,
    IAS_SAT_L5,
    IAS_SAT_L7
} IAS_SATELLITE_TYPE;

typedef enum IAS_FORMAT_TYPE  /* TM or MSS data format */
{
    IAS_INVALID,
    IAS_TM_R,
    IAS_TM_A,
    IAS_MSS_X,
    IAS_MSS_A,
    IAS_MSS_P
} IAS_FORMAT_TYPE;

typedef enum IAS_MSS_X_TYPE  /* MSS-X data format */
{
    IAS_MSS_X_INVALID,
    IAS_MSS_X_WBV,
    IAS_MSS_X_CCT,
    IAS_MSS_X_ORF
} IAS_MSS_X_TYPE;

typedef struct xxx_FuncPtrs
{
    /* Band functions */
    void (*InitBands)(void);
    void (*CreateBandNode)
    (
        BAND_HDR_TYPE *band_list, /* I: band header information */
        BAND_OPTION_TYPE option,  /* I: band option (standard or custom) */
        BandNumber band_number    /* I: band number */
    );

#ifndef IAS_NO_SENSOR_META_SUPPORT
    /* Read MSCD function */
    xxx_MSCDControl_TYPE *( * ReadMSCD )
    ( 
        char * p_PathName,    /* I: Pathname to one for the PCD files */
        char * p_ErrorMessage /* O: Error message, if any             */
    );   

    /* Read PCD function */
    xxx_PCDControl_TYPE *( * ReadPCD )
    ( 
        char * p_PathName,    /* I: Pathname to one for the PCD files */
        char * p_ErrorMessage /* O: Error message, if any             */
    );   

    /* Read L0R Report function */
    int ( * Read0R_Report )
    ( 
        xxx_ReportData_TYPE *p_0R_Report, /* O: 0R report structure */
        char * p_PathName,                /* I: Path to 0R report   */
        char * p_ErrorMessage             /* O: Error message       */
    );   

    /* Read the EDC metadata file */
    xxx_EDCMetaDataFile_TYPE *( * ReadEDCMetadata )
    (
        char *p_PathName,     /* I: Pathname to L0R HDF directory file */
        char *p_ErrorMessage  /* O: Error message, if any */
    );

    /* Read the LPGS metadata file */
    xxx_LPGSMetaDataFile_TYPE *( * ReadLPGSMetadata )
    (
        char *p_PathName,     /* I: Pathname to L0R HDF directory file */
        char *p_ErrorMessage  /* O: Error message, if any */
    );

    /* Read the MTA metadata file */
    xxx_AllMetaDataFile_TYPE *( * ReadMetadata )
    (   
        char *p_PathName,    /* I: Pathname to L0R HDF directory file */
        char *p_ErrorMessage /* O: Error message, if any */
    );
#endif
} xxx_FuncPtrs;


/* Function prototypes */

IAS_SENSOR_TYPE xxx_initialize_sensor_type
(
    const char * SensorTypeStr   /* I: sensor type string */
);

IAS_FORMAT_TYPE xxx_set_format
(
    const char *format           /* I: TM or MSS format string */
);

IAS_MSS_X_TYPE xxx_set_mss_x_type
(
    const char *type             /* I: MSS-X data type string */
);

const char *xxx_get_sensor_from_fname
(
    const char *PathName         /* I: full path filename */
);

IAS_SENSOR_TYPE xxx_get_sensor_type(void);

char *xxx_get_sensor_type_str(void);

IAS_SATELLITE_TYPE xxx_get_satellite_type(void);

char *xxx_get_satellite_str(void);

IAS_FORMAT_TYPE xxx_get_format(void);

IAS_MSS_X_TYPE xxx_get_mss_x_type(void);

int xxx_initialize_func_ptr(void);

xxx_FuncPtrs *xxx_get_funcs( void );

char *xxx_get_dbsensor(void);

IAS_FORMAT_TYPE xxx_get_format_from_ddr
(
    char *hdfname       /* I: HDF filename */
);

int xxx_map_band
(
    const int band,       /* I: Internal band number */
    const char *bandstr,  /* I: Internal band string; may be NULL */
    char *newbandstr      /* O: Public band string */
);

/*======================================================================
========================= Metadata Section =============================
The following group of functions are for setting and getting metadata.
These data are retrieved from the MTA and MTP structures.
======================================================================*/
#ifndef IAS_NO_SENSOR_META_SUPPORT
xxx_EDCMetaDataFile_TYPE *xxx_set_EDCMetaDataFile_TYPE( void );
xxx_LPGSMetaDataFile_TYPE *xxx_set_LPGSMetaDataFile_TYPE( void );
xxx_EDCMetaDataFile_TYPE *xxx_getEDCMetaData( void );
xxx_AllMetaDataFile_TYPE *xxx_getMetadata( void );
#endif
float xxx_getTotalWRSScenes_MTP( void );
long xxx_getTotalWRSScenes_MTA( void );
long xxx_getWRSPath( void );
long xxx_getStartingRow( void );
long xxx_getEndingRow( void );
long xxx_getStartingSubintervalScan( void );
long xxx_getEndingSubintervalScan( void );
long xxx_getTotalScenes( void );
char *xxx_getSpacecraftId( void );
char *xxx_getSensorId( void );
char *xxx_getAcquisitionDate( void );
char *xxx_getTapeDate( void );
long xxx_getNumberOfScans( void );
char *xxx_getCPF_FileName( void );
long *xxx_get_SceneCenterScanNoArray( void );
long *xxx_get_WRSPathArray( void );
long *xxx_get_WRSRowArray( void );
long *xxx_getHorizontalDisplayShiftArray( void );
float xxx_getProductLLCornerLat( void );
float xxx_getProductLLCornerLon( void );
float xxx_getProductLRCornerLat( void );
float xxx_getProductLRCornerLon( void );
float xxx_getProductULCornerLat( void );
float xxx_getProductULCornerLon( void );
float xxx_getProductURCornerLat( void );
float xxx_getProductURCornerLon( void );
long xxx_getTotalScans( void );
float xxx_getSceneCenterLat( int );
float xxx_getSceneCenterLon( int );
float xxx_getSunElevationAngle( int );
xxxISOYDHMSFTime *xxx_getSceneCenterScanTime( void );
char *xxx_getSceneCenterScanTimeStr( int );
long *xxx_getWRSSceneNoArray( void );
long xxx_get_Total_PCD_MajorFrames( void );
xxxISOYDHMSFTime *xxx_get_PCD_StartTime( void );
xxxISOYDHMSFTime *xxx_get_PCD_StopTime( void );
float *xxx_get_Scene_UL_CornerLatArray( void );
float *xxx_get_Scene_UL_CornerLonArray( void );
float *xxx_get_Scene_UR_CornerLatArray( void );
float *xxx_get_Scene_UR_CornerLonArray( void );
float *xxx_get_Scene_LL_CornerLatArray( void );
float *xxx_get_Scene_LL_CornerLonArray( void );
float *xxx_get_Scene_LR_CornerLatArray( void );
float *xxx_get_Scene_LR_CornerLonArray( void );
char *xxx_getDayNightFlag( int index );
long *xxx_getSceneQualityArray( void );

#endif /* XXX_SENSOR_H */
