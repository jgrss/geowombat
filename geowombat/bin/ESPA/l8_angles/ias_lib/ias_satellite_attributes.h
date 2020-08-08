#ifndef IAS_SATELLITE_ATTRIBUTES_H
#define IAS_SATELLITE_ATTRIBUTES_H

#include "ias_const.h"

/* Defines the maximum number of normal bands present on any of the supported
   satellites.  Note that this should only be used if absolutely necessary.
   It is preferred that arrays be sized dynamically. */
#define IAS_MAX_NBANDS 11

/* Defines the maximum number of SCAs present on any of the supported 
   sensors.   Note that this should only be used if absolutely necessary.
   It is preferred that arrays be sized dynamically. */
#define IAS_MAX_NSCAS 14

/* Defines the maximum number of total bands present on any of the supported
   satellites.  Note that this should only be used if absolutely necessary.
   It is preferred that arrays be sized dynamically. */
#define IAS_MAX_TOTAL_BANDS 30

/* Accepted names for the Landsat 8 satellite and sensor. */
#define IAS_SATELLITE_NAME_L8 "L8"
#define IAS_SENSOR_NAME_L8    "OLITIRS"

/* Satellite and sensor "names" when the satellite and/or sensor cannot
   be determined from the satellite attributes information. */
#define IAS_SATELLITE_NAME_UNKNOWN   "Unknown"
#define IAS_SENSOR_NAME_UNKNOWN      "Unknown"


/* An enumerated type for the supported sensors */
typedef enum
{
    IAS_INVALID_SENSOR_ID = -1,
    IAS_OLI,        /* OLI sensor  */
    IAS_TIRS,       /* TIRS sensor */
    IAS_MAX_SENSORS, /* Total number of supported sensors */
} IAS_SENSOR_ID;

/* An enumerated type for the supported satellites. */
typedef enum
{
    IAS_L8          /* Landsat 8 */

} IAS_SATELLITE_ID;

/* Band types to allow finding bands by the data they contain instead of an
   arbitrary number.  The list here is preliminary. */
typedef enum
{
    IAS_UNKNOWN_BAND_TYPE = -1,
    IAS_COASTAL_AEROSOL_BAND,
    IAS_BLUE_BAND,
    IAS_GREEN_BAND,
    IAS_RED_BAND,
    IAS_NIR_BAND,       /* near infrared */
    IAS_SWIR1_BAND,     /* mid-infrared band 1 (for hydrology uses) */
    IAS_SWIR2_BAND,     /* mid-infrared band 1 (for geological uses) */
    IAS_PAN_BAND,       /* panchromatic band */
    IAS_CIRRUS_BAND,    /* thermal infrared */
    IAS_THERMAL1_BAND,  /* 10.8um thermal TIRS band */
    IAS_THERMAL2_BAND,  /* 12um thermal TIRS band */
    IAS_NUM_BAND_TYPES  /* Must be last to provide the number of band types */
} IAS_BAND_TYPE;

/* The band classification provides a mechanism to determine whether a band is
   a "normal" band, or a pseudo-band like the VRP or blind bands.  The 
   classification is a bit mask.  More than one bit can be set for special
   cases (like the TIRS blind secondary band) */
typedef enum
{
    IAS_NORMAL_BAND = 1,    /* a normal visible, thermal, or pan band */
    IAS_VRP_BAND = 2,       /* a video reference pixel band */
    IAS_BLIND_BAND = 4,     /* a blind band */
    IAS_SECONDARY_BAND = 8  /* a secondary band */
} IAS_BAND_CLASSIFICATION;

/* The band spectral type provides a mechanism to determine whether a band is
   a pan, multispectral, or thermal band. */
typedef enum
{
    IAS_UNKNOWN_SPECTRAL_TYPE = -1,
    IAS_SPECTRAL_PAN,
    IAS_SPECTRAL_VNIR,
    IAS_SPECTRAL_SWIR,
    IAS_SPECTRAL_THERMAL
} IAS_SPECTRAL_TYPE;

/* The sensor types supported.  Currently, only pushbroom sensors are
   supported.  However, this will allow expanding the sensor types supported
   in the future. */
typedef enum
{
    IAS_PUSHBROOM_SENSOR

} IAS_SENSOR_TYPE;

/* This defines the band number to use to indicate the band number is not
   valid.  Legal band numbers start at one. */
#define IAS_INVALID_BAND_NUMBER 0


/* This structure stores attributes for a single band.  The members of the
   structure are currently tailored to a pushbroom sensor.  If/when a scanning
   sensor, like Landsat 1-7, is added, new members can be added that are
   tailored to that type of sensor. */
typedef struct ias_band_attributes
{
    int band_number;            /* band number (1-based) */
    int band_index;             /* band index = (band number - 1) */
    char band_name[IAS_BAND_NAME_SIZE];
    IAS_SENSOR_ID sensor_id;    /* sensor id for this band */
    IAS_SENSOR_TYPE sensor_type;/* sensor type (pushbroom is only option now) */
    IAS_BAND_TYPE band_type;
    IAS_BAND_CLASSIFICATION band_classification;
                        /* the band classification is a bit mask for the
                           classifications that apply to the band */
    IAS_SPECTRAL_TYPE spectral_type;

    /* the following fields are included to allow defining the relationship 
       between bands that are associated with each other.  For example, when a
       normal band has an associated blind and VRP band, each of them has a
       unique band number.  The VRP band number of a normal band can be found
       by using the vrp_band_number from the normal band.  Likewise, the 
       normal band for a blind band can be found by using the
       normal_band_number field.  If there isn't an associated band for a
       type, the associated band number will be set to zero.  Likewise, a
       normal band would have zero in the normal_band_number since it isn't
       useful to fill in that field with a duplicate of the band_number
       field.  */
    int normal_band_number;
    int vrp_band_number;
    int blind_band_number;
    int secondary_band_number;  /* The secondary band number if there is one 
                                   associated with this band */

    int scas;                   /* number of SCAs in the band */
    int detectors_per_sca;      /* detectors in each of the SCAs */
    int lines_per_frame;        /* image lines per collected frame */
    int qcal_min;               /* The QUANTIZE_CAL_MIN_BAND_X value */
    int qcal_max;               /* The QUANTIZE_CAL_MAX_BAND_X value */
    double pixel_resolution;    /* pixel resolution in meters */
    double wavelength_nm_range[2]; /* range of wavelengths in nm in this band 
                                      - this really isn't needed for anything, 
                                      but could prove useful for metadata. */
    int can_saturate;           /* Flag indicating whether or not a band can
                                   contain saturation or not */
} IAS_BAND_ATTRIBUTES;

/* Defines the structure to store the top-level satellite attributes */
typedef struct ias_satellite_attributes
{
    char satellite_name[IAS_SPACECRAFT_NAME_SIZE];
    char sensor_name[IAS_SENSOR_NAME_SIZE];
    IAS_SATELLITE_ID satellite_id; /* id for this satellite */
    int sensors;        /* number of sensors on the satellite */
    int bands;          /* number of normal bands for all sensors combined */
    int total_bands;    /* total number of bands */
    IAS_SENSOR_ID *sensor_ids; /* ids of sensors on the satellite */
    IAS_BAND_ATTRIBUTES *band_attributes; /* pointer to an array of band
                           attributes for all the bands */
} IAS_SATELLITE_ATTRIBUTES;

int ias_sat_attr_initialize
(
    int satellite_id        /* I: Satellite id */
);

const IAS_SATELLITE_ATTRIBUTES *ias_sat_attr_get_attributes();

IAS_SATELLITE_ID ias_sat_attr_get_satellite_id();

IAS_SATELLITE_ID ias_sat_attr_get_satellite_id_from_satellite_number
(
    int satellite_number /* I: The satellite number */
);

int ias_sat_attr_get_sensor_count();

int ias_sat_attr_get_satellite_number();

int ias_sat_attr_get_normal_band_count();

const IAS_BAND_ATTRIBUTES *ias_sat_attr_get_band_attributes
(
    int band_number             /* I: Band number */
);

int ias_sat_attr_get_sensor_band_numbers
(
    int sensor_id,              /* I: Sensor id or IAS_MAX_SENSORS if any
                                      sensor */
    int band_class,             /* I: band classification or OR'ed
                                      classifications that all must be
                                      included */
    int band_class_exclusion,   /* I: Band classification exclusion(s) an OR
                                      of classifications to exclude or the
                                      value 0 for no exclusions */
    int *band_number,           /* O: Pointer to an integer band number array 
                                      to populate */
    int size,                   /* I: Size of the band number array */
    int *number_of_bands        /* O: Pointer to the number of bands returned 
                                      in the band number array */
);

int ias_sat_attr_get_any_sensor_band_numbers
(
    int sensor_id,              /* I: Sensor id or IAS_MAX_SENSORS if any
                                      sensor */
    int band_class,             /* I: Match any of the band classification
                                      OR'ed together in this parameter */
    int band_class_exclusion,   /* I: Band classification exclusion(s) an OR
                                      of classifications to exclude or the
                                      value 0 for no exclusions */
    int *band_number,           /* O: Pointer to an integer band number array 
                                      to populate */
    int size,                   /* I: Size of the band number array */
    int *number_of_bands        /* O: Pointer to the number of bands returned 
                                      in the band number array */
);

int ias_sat_attr_band_classification_matches
(
    int band_number,        /* I: band number to check */
    IAS_BAND_CLASSIFICATION band_classification /* I: exact classification 
                                                      to match */
);

const char *ias_sat_attr_convert_band_number_to_name
(
    int band_number     /* I: band number to convert to a name */
);

int ias_sat_attr_convert_band_index_to_number
(
    int band_index      /* I: band index to convert to a number */
);

int ias_sat_attr_convert_band_number_to_index
(
    int band_number     /* I: band number to convert to an index */
);

int ias_sat_attr_get_band_number_from_type
(
    IAS_BAND_TYPE band_type     /* I: band number */
);


int ias_sat_attr_get_scas_per_band
(
    int band_number     /* I: band to get number of SCAs from */
);

int ias_sat_attr_get_detectors_per_sca
(
    int band_number     /* I: band to get number of detectors from */
);

int ias_sat_attr_get_lines_per_frame
(
    int band_number     /* I: band to get lines per frame for */
);

const char *ias_sat_attr_get_sensor_name
(
    int sensor_id       /* I: sensor id to get the name of */
);

const char *ias_sat_attr_get_satellite_name();

int ias_sat_attr_get_sensor_sca_count
(
    int sensor_id
);

IAS_SPECTRAL_TYPE ias_sat_attr_get_spectral_type_from_band_number
(
    int band_number                    /* I: Current 1-based band number */
);

int ias_sat_attr_get_sensor_max_normal_detectors
(
    int sensor_id                      /* I: Sensor ID number for OLI OR
                                          TIRS */
);

int ias_sat_attr_convert_band_name_to_number
(
    const char *band_name              /* I: Band name to search on */
);

int ias_sat_attr_get_band_name_from_type_and_class
(
    IAS_BAND_TYPE band_type,                       /* I: Band type */
    IAS_BAND_CLASSIFICATION band_classification,   /* I: Classification */
    char *band_name                                /* O: Band name */
);

IAS_BAND_TYPE ias_sat_attr_get_band_type_from_band_number
(
    int band_number                   /* I: Current band number */
);

IAS_SENSOR_ID ias_sat_attr_get_sensor_id_from_band_number
(
    int band_number                   /* I: Current band number */
);

int ias_sat_attr_get_quantization_cal_min
(
    int band_number,            /* I: Current band number */
    int *qcal_min               /* O: qcal min value */
);

int ias_sat_attr_get_quantization_cal_max
(
    int band_number,            /* I: Current band number */
    int *qcal_max               /* O: qcal max value */
);

int ias_sat_attr_report_saturation_for_band
(
    int band_number,                   /* I: Band number */
    int *should_report_saturation_flag /* O: Flag to say if saturation should
                                             be reported */
);

/* Routine to retrieve the sensor name associated with the satellite name. */
const char *ias_sat_attr_get_satellite_sensor_name(void);

/*---------------------------------------------------------------------------
WARNING: This routine should not be used in the IAS.  IAS routines should
         use:
             ias_sat_attr_convert_band_number_to_index
         This routine was developed for use by Ingest.  It returns an index
         that is relative to the sensor not all the bands. 
---------------------------------------------------------------------------*/
int ias_sat_attr_convert_band_number_to_sensor_band_index
(
    int sensor_id,
    int band_number
);

#endif

