#ifndef IAS_CONST_H
#define IAS_CONST_H

#ifndef TRUE
#define TRUE  1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define SUCCESS  0
#define ERROR   -1
#define WARNING -2

/* Put IAS-specific constants here. */
#define IAS_DAYS_IN_YEAR        365
#define IAS_DAYS_IN_LEAP_YEAR   366              /* Number of days per year
                                                     (+1 for leap years) */
#define IAS_SOFTWARE_VERSION_SIZE 11
#define IAS_INSTRUMENT_SOURCE_SIZE 32
#define IAS_BAND_NAME_SIZE 30
#define IAS_COLLECT_TYPE_SIZE 50

#define IAS_EPOCH_2000      2451545.0       /* Julian date of epoch 2000    */
#define IAS_JULIAN_CENTURY  36525.0         /* Julian century */

#define IAS_SEC_PER_DAY     86400                /* Seconds Per Day */
#define IAS_DAILY_SECONDS_IN_LEAP_YEAR 86401     /* Number of seconds per day
                                                     (+1 to account for leap
                                                     seconds) */

/*
 * GPS epoch year.  1970 was chosen to support DOQ GCPs that have slightly bogus
 * dates of 7/1/1970
 */
#define IAS_MIN_YEAR 1970

#define IAS_MAX_YEAR 2099
#define IAS_MAX_MONTH 12
#define IAS_MIN_MONTH 1

/* Length of the composite IAS sensor name "L8_OLITIRS" */
#define IAS_SENSOR_NAME_LENGTH 11

/* Maximum size of the buffer used to hold IAS_SENSOR_NAME_LENGTH
   characters, including the NULL terminator. */
#define IAS_SENSOR_NAME_SIZE   (IAS_SENSOR_NAME_LENGTH + 1)

/* Size of the composite IAS sensor ID "OLI_TIRS",
   including null terminator */
#define IAS_SENSOR_ID_SIZE 9

/* Length of the satellite name "L8" or "Landsat_8", sized to match the
   'SATELLITE' database field name.  */
#define IAS_SPACECRAFT_NAME_LENGTH 10

/* Maximum size of the buffer used to hold IAS_SPACECRAFT_NAME_LENGTH
   characters, including the NULL terminator. */
#define IAS_SPACECRAFT_NAME_SIZE   (IAS_SPACECRAFT_NAME_LENGTH + 1)

#define IAS_MODIFIED_JULIAN_DATE 2400000.5
/* The Modified Julian Date Constant Modified Julian Date = FullJuilanDate -
 * MDJ (where FullJulianDate is a running day count since Jan 1 4713 B.C. */ 

/* define some constants related to projection definitions */
#define IAS_UNITS_SIZE 12
/* maximum size of the units string (i.e. METERS, FEET, etc) */

#define IAS_DATUM_SIZE 16
/* maximum size of the datum name string */

#define IAS_PROJ_PARAM_SIZE 15
/* size of the array for holding projection parameters */

#define IAS_WORKORDER_ID_LENGTH 12
/* length of a Work Order ID */
#define IAS_WORKORDER_ID_SIZE (IAS_WORKORDER_ID_LENGTH + 1)
/* size of a Work Order ID string, including null terminator */

#define IAS_PRODUCT_REQUEST_ID_LENGTH 20
/* length of a Product Request ID */
#define IAS_PRODUCT_REQUEST_ID_SIZE (IAS_PRODUCT_REQUEST_ID_LENGTH + 1)
/* size of a Product Request ID string, including null terminator */

#define IAS_CHAR_ID_LENGTH 20
/* length of a characterization ID */
#define IAS_CHAR_ID_SIZE (IAS_CHAR_ID_LENGTH + 1)
/* size of a characterization ID string, including null terminator */

/* RPS report header string sizes (include NULL terminator) */
#define IAS_DIFFUSER_TYPE_SIZE           9
#define IAS_BIAS_MODEL_TYPE_SIZE        11

/* Number of points to use in Lagrange interpolation */
#define IAS_LAGRANGE_PTS 4

/* Number of values in the LOS LEGRENDRE Coefficeint arrays */
#define IAS_LOS_LEGENDRE_TERMS 4

 /* If roll angle is this far from 0 degrees, call it off-nadir */
#define IAS_OFF_NADIR_BOUNDARY 0.5

/* Macro to determine odd/even image data lines.  The convention assumed
   here is that line index 0 is an "odd" line, line index 1 is an "even"
   line, etc.  This convention applies to both radiometric and geometric
   processing.

   This is needed for radiometric processing of the OLI PAN band (band 8),
   because there is a coherent noise-like effect creating a response
   difference between "odd" and "even" image lines.  Several RPS processing
   algorithms need to account for this difference.

   The parentheses around the parameter name in the macro definition ARE
   IMPORTANT, so don't delete them */
#define IAS_IS_EVEN_LINE(line_index)                                 \
    (((line_index) & 0x01) != 0)


/* Common value used by create_grid and terrain_occlusion to represent
   an invalid elevation value.  Other GPS (and perhaps RPS) needing
   a value to represent an invalid elevation data value should use
   this constant */
#define INVALID_ELEVATION -999999

/*           ******* QUALITY BAND BIT IDENTIFICATION *******

   Each sample is assigned a 16-bit mask indicating the status of that sample.
   The bit positions of the mask are described below.  The representation of a
   16-bit mask is shown on the right with a flag indicating where each class's
   confidence level is stored.
  
       Flag Description      Bits      Flag       RRRi issS SccC rrtf
       ------------------    -----     ----       1111 1111 1111 1111
       Designated Fill           0    f   
       Terrain Occlusion         1    t
       Radiometric Saturation  2-3    r
       Cloud                     4    C
       Cloud confidence        5-6    c
       Cloud Shadow            7-8    S
       Snow / Ice confidence  9-10    s
       Cirrus confidence     11-12    i
       <Reserved>            13-15    R

    The following defines can be used to extract the bit information from a
    quality band sample.
*/

#define IAS_QUALITY_BIT_NOTHING           0x0000 /* No bits are selected */

#define IAS_QUALITY_BIT_FILL              0x0001 /* Bit 0 */

#define IAS_QUALITY_BIT_TERRAIN_OCCLUSION 0x0002 /* Bit 1 */

#define IAS_QUALITY_BIT_RAD_SAT_01        0x0004 /* Bit 2 */
#define IAS_QUALITY_BIT_RAD_SAT_10        0x0008 /* Bit 3 */
#define IAS_QUALITY_BIT_RAD_SAT_11        0x000C /* Bit 2 and 3 */

#define IAS_QUALITY_BIT_CLOUD             0x0010 /* Bit 4 */

#define IAS_QUALITY_BIT_CLOUD_01          0x0020 /* Bit 5 */
#define IAS_QUALITY_BIT_CLOUD_10          0x0040 /* Bit 6 */
#define IAS_QUALITY_BIT_CLOUD_11          0x0060 /* Bit 5 and 6 */

#define IAS_QUALITY_BIT_CLOUD_SHADOW_01   0x0080 /* Bit 7 */
#define IAS_QUALITY_BIT_CLOUD_SHADOW_10   0x0100 /* Bit 8 */
#define IAS_QUALITY_BIT_CLOUD_SHADOW_11   0x0180 /* Bit 7 and 8 */

#define IAS_QUALITY_BIT_SNOW_ICE_01       0x0200 /* Bit 9 */
#define IAS_QUALITY_BIT_SNOW_ICE_10       0x0400 /* Bit 10 */
#define IAS_QUALITY_BIT_SNOW_ICE_11       0x0600 /* Bits 9 and 10 */

#define IAS_QUALITY_BIT_CIRRUS_01         0x0800 /* Bit 11 */
#define IAS_QUALITY_BIT_CIRRUS_10         0x1000 /* Bit 12 */
#define IAS_QUALITY_BIT_CIRRUS_11         0x1800 /* Bits 11 and 12 */

#define IAS_QUALITY_BIT_RESERVED_1        0x2000 /* Bit 13 */
#define IAS_QUALITY_BIT_RESERVED_2        0x4000 /* Bit 14 */
#define IAS_QUALITY_BIT_RESERVED_3        0x8000 /* Bit 15 */

#endif
