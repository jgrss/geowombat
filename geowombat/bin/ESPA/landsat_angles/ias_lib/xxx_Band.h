
#ifndef _XXX_BAND_H_
#define _XXX_BAND_H_

#define BAND_PROG_NAME "xxx_Band"

/* These are now a superset of all of the bands in all sensors. */

/* This is typically used in the GPS Library as the band indexing  */
/* method. OPTICAL_AXIS is the center of the optical/focal plane.  */

/* Maximum number of resolutions for all sensors.
 * This define should always be set to the define
 * of the sensor with the largest number of resolutions.
 */
#define  MAX_NUM_RES  NUM_RESOLUTIONS_ETM 

/* This is the maximum for all sensors.  This could 
 * be used, eg, in array size definitions that are  
 * defined at compile time.  Use xxx_get_max_bands  
 * if possible - it's sensor-specific. 

   NOTE that this does not encompass all of the band numbers 
   in the enumerated type "BandNumber" below.  In places 
   where that is needed use NBANDS instead.             
 */ 
#define MAXBANDS  MAX_LOGICAL_BANDS_ETM

#define BAND_STR_SZ 5 /* size of typical band string like B30 with room */
                      /* for null.                                      */
#define BAND_SLO_STR_SZ 6 /* size of numerical band string with period  */
                      /* with room for null.  eg: ".030"                */

/* The following 2 enumerated types uniquely identify bands for a specific */
/* satellite and sensor.  There is overlap across multiple satellites and  */
/* sensors.                                                                */

#ifndef XXX_BAND_NUMBER
#define XXX_BAND_NUMBER
/* This is also re-implemented in the IAS Qt Library. */
/* The band number is typically used in the GPS. */
typedef enum band_number
{
    berror=-1,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6l,
    b6h,
    b7,
    b8,
    b6,
    NBANDS,
    OPTICAL_AXIS
} BandNumber;
#endif /* XXX_BAND_NUMBER */

/* This is typically used in the RPS.  It could be merged with      */
/* BandNumber above, with some benefit in terms of maintainability. */

typedef enum band_enums
{
    B1,
    B2,
    B3,
    B4,
    B5,
    B6L,
    B6H,
    B7,
    B8,
    B6,
    BINVALID
} BAND_TYPE;

/* This is used by band nodes to group bands with common characteristics   */
/* together.  For example, with ETM, REFLECTIVE can be used as a short-    */
/* hand for B1, B2, B3, B4, B5, and B7.  This is most useful when multiple */
/* multi-sensor lists would otherwise be needed.                           */

typedef enum band_mode_enum 
{
    REFLECTIVE,
    THERMAL,
    PANCHROMATIC
} band_mode_type;

/* These are different temperature correction options for a band. */

typedef enum temp_corr_enum
{                           
    TEMP_SILICON_FP_ASSEMBLY, 
    TEMP_CFPA_MON,
    TEMP_NO_CORRECTION
} temp_corr_type;

/* These are options used when building a band list. */

typedef enum band_option  
{
    STANDARD,   /* Build a regular full band list.                         */
    CUSTOM      /* Build a band list with algorithm-specific set of bands. */
} BAND_OPTION_TYPE;


/* For the linked list implementation, the LMask is used as a model.    */

typedef struct band_node /* band-specific information */
{
    int band_number;        /* This is the physical band number.            */
    int band_long_number;   /* This is a long version of the band number.   */
    BandNumber band_id;     /* This is an enumerated type (eg: b1, etc)     */
                            /* that is often used in GPS. Can it be merged? */
    BAND_TYPE band_id_rps;  /* This is an enumerated type (eg: B1, etc)     */
                            /* akin to band_id that is often used in RPS.   */
    char *band_short_name;  /* This is a short version of the band's name.  */
    char *band_name;        /* This is the band's name for display.         */
    char *sds_band_name;    /* This is the SDS band filename section.       */
    char *band_name_ext;    /* This is the band name with a "_".            */
    char *band_name_ext_81; /* This is the band name with a "_" (and "81"). */
    float band_resolution;  /* This is in meters.                           */
    band_mode_type band_mode; /* This is an enumerated type (THERMAL...)   */
    temp_corr_type temp_corr; /* enumerated type (NO_TEMP_CORRECTION...)   */
    int num_detectors;     /* This is the number of detectors for the band. */
    long int maxImgPixels; /* These are from rxx_Constants.h.  They will be */
    long int maxCalPixels; /* sent to structures that look like the         */
    long int CalDataLen;   /* originals to minimize algorithm changes in    */
    long int CalDkRegLen;  /* the initial multi-sensor release.             */
    int CalDkRegLength[2];     /* Forward/Reverse Bias Length in CPF        */ 
    int CalDkRegOffset[2];     /* Forward/Reverse Bias Location in CPF      */
    int IC_Usable_Length[2];   /* Forward/Reverse IC Region in CPF          */
    char band_id_msg[BAND_STR_SZ];       /* were from rxx_BAND_TYPE_MSG,    */
    char IMGnames[BAND_STR_SZ];          /* were from RXX_BAND1, etc...     */
    char CALnames[BAND_STR_SZ];          /* were from RXX_CAL_BAND1, etc... */
    char SLOvdata[BAND_SLO_STR_SZ];      /* were from XXX_SLO_BAND1, etc... */
    int joffset;           /* offset - used in IC Trend write               */
    /*    ETM joffset: (10n)+1 where n=1 for bands 1-5,7 and n=2 for band 8 */
    int active;    /* This indicates whether the band is in use.  It must
                      be initialized by the algorithm that uses it.         */
    int num_lines;   /* Number of lines for the band.                       */
    int num_samples; /* Number of samples for the band.                     */

    /* Band structure components go here.                                   */

    struct band_node * Prev;  /* points to previous band in linked list     */
    struct band_node * Next;  /* points to next band in linked list         */
} BAND_NODE_TYPE;

typedef struct band_hdr   /* Global band information */
{
    int num_bands;                /* Maximum number of bands for the sensor */
    int last_band;                /* The last band for the sensor           */
    BAND_NODE_TYPE *ref_band;     /* This points to the reference band.     */
    BAND_NODE_TYPE *Head;        /* points to head of linked list           */
    BAND_NODE_TYPE *End;         /* points to end of linked list            */
    BAND_NODE_TYPE *Current;     /* used during processing to point to last */
                                 /* accessed band_node                      */
} BAND_HDR_TYPE;

#define BAND_NODE_TYPE_SIZE sizeof (BAND_NODE_TYPE)
#define BAND_HDR_TYPE_SIZE  sizeof (BAND_HDR_TYPE)

/* The following are used to indentify which field of the Band */
/* structure is of interest.  For example, a particular Band   */
/* parameter can be searched for (eg: search the band list for */
/* a match with band_name "B10").                              */

typedef enum band_fields
{
    BAND_NUMBER_TYPE,
    BAND_LONG_NUMBER_TYPE,
    BAND_ID_TYPE,
    BAND_ID_RPS_TYPE,
    BAND_SHORT_NAME_TYPE,
    BAND_NAME_TYPE,
    BAND_SDS_NAME_TYPE,
    BAND_RESOLUTION_TYPE
} BAND_FIELD;


int xxx_BuildBandNode
(
    BAND_HDR_TYPE *band_hdr, /* I/O: Band list to put the band node in */
    BAND_OPTION_TYPE option, /* I: Standard band list, or customized one */
    int band_number,         /* I: Physical band number */
    char *band_name,         /* I: Band's name for display */
    int num_lines,           /* I: Number of lines */
    int num_samples,         /* I: Number of samples */
    float band_resolution    /* I: Pixel size (in meters) */
);

int xxx_BuildBandNode_full
(
    BAND_HDR_TYPE *band_hdr, /* I/O: Band list to put the band node in */
    BAND_OPTION_TYPE option, /* I: Standard band list, or customized one */
    int band_number,         /* I: Physical band number */
    int band_long_number,    /* I: Long version of the band number */
    BandNumber band_id,      /* I: Enumerated type (eg: b1, etc.) */
    BAND_TYPE band_id_rps,   /* I: Enumerated type (eg: B1, etc.) */
                             /* I: Akin to band_id that is often used in RPS */
    char *band_short_name,   /* I: Short version of the band's name */
    char *band_name,         /* I: Band's name for display */
    char *sds_band_name,     /* I: SDS band filename section */
    char *band_name_ext,     /* I: Band name with a "_" */
    char *band_name_ext_81,  /* I: Band name with a "_" (and "81") */
    int num_lines,           /* I: Number of lines */
    int num_samples,         /* I: Number of samples */
    float band_resolution,   /* I: Pixel size (in meters) */
    band_mode_type band_mode, /* I: Enumerated type (THERMAL...)  */
    temp_corr_type temp_corr, /* I: Enumerated type (NO_TEMP_CORRECTION...) */
    int num_detectors,     /* I: Number of detectors */
    long int maxImgPixels, /* I: From rxx_Constants.h. They will be */
    long int maxCalPixels, /* I: sent to structures that look like the */
    char band_id_msg[BAND_STR_SZ],   /* I: were from rxx_BAND_TYPE_MSG */
    char IMGnames[BAND_STR_SZ],      /* I: were from RXX_BAND1, etc... */
    char CALnames[BAND_STR_SZ],      /* I: were from RXX_CAL_BAND1, etc... */
    char SLOvdata[BAND_SLO_STR_SZ],  /* I: were from XXX_SLO_BAND1, etc... */
    int joffset,           /* I: Offset - used in IC Trend write */
    int active             /* I: Indicates whether the band is in use */
);

void xxx_DisplayBandHdr  
(
    BAND_HDR_TYPE *band_hdr /* I: the header of the band list being displayed */
);

void xxx_DisplayBandNode 
(
    BAND_NODE_TYPE *band_node /* I: a band node being displayed */
);

void xxx_DisplayBandList 
(
    BAND_HDR_TYPE *band_hdr /* I: the band list that is being displayed */
);

BAND_NODE_TYPE * xxx_GetBandNode 
(
    BAND_HDR_TYPE *band_hdr, /* I: header of the band list being searched */
    int search_value,        /* I: the value we are searching for         */
    BAND_FIELD search_field  /* I: where we are looking                   */
);

BAND_NODE_TYPE * xxx_GetBandNodeStr 
(
    BAND_HDR_TYPE *band_hdr, /* I: header of the band list being searched */
    char *search_value,      /* I: the value we are searching for         */
    BAND_FIELD search_field  /* I: where we are looking                   */
);

int xxx_FreeBands 
(
    BAND_HDR_TYPE *band_hdr /* I: Pointer to band header structure */
);

BAND_HDR_TYPE *xxx_GetBands (void);

void xxx_SetBands(BAND_HDR_TYPE *bh);

BAND_HDR_TYPE *xxx_CreateBandHdr 
(
    BAND_OPTION_TYPE option /* I: create standard or customized band list */
);

int xxx_initialize_max_bands(void);        /* includes physical and logical */
int xxx_initialize_num_resolutions(void);
int xxx_get_max_bands(void);
int xxx_get_max_logical_bands(void);
int xxx_get_num_resolutions(void);
int xxx_get_num_reflective_bands(void);

int xxx_get_res_list
(
    BAND_HDR_TYPE *band_hdr,    /* I: band list header */
    int array_size,             /* I: size of res and numbands_in_res arrays*/
    int *numres,                /* I/O: number of different resolutions in
                                        band list */
    float res[],                /* I/O: different pixel resolutions in
                                        band list */
    int numbands_in_res[],      /* I/O: number of bands in each resolution */
    int bandflag[]              /* I/O: array of flags indicating bands that
                                        have been accounted for; must be
                                        large enough for all bands in list */
);

int xxx_get_user_band
(
    BandNumber band_id   /* I: Enumerated band number */
);

BandNumber xxx_parse_user_band
(
    int user_band_number   /* I: Integer band number */
);

/* These need to be defined after BAND_HDR_TYPE, since they use it. */

#include "etm/xxx_Band_etm.h"
#include "mss/xxx_Band_mss.h"
#include "tm/xxx_Band_tm.h"

#endif

