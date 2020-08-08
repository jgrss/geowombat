#ifndef XXX_TIMECONVERT_H
#define XXX_TIMECONVERT_H

#include "xxx_Types.h"

/******************************************************************************
These define sizes and data types for time formats used on IAS. Note that for
ASCII formats, the defined size does NOT include a terminating '\0' char.
******************************************************************************/

#define OUT_OF_RANGE(num,min,max)   ((num) < (min) || (num) > (max))
#define IS_LEAP_YEAR(year) (!((year)%4) && (((year)%100) || !((year)%400)))
#define DAYS_IN_MONTH(year,month)   ((IS_LEAP_YEAR(year) && (month) == 2) ? 29 : days_in_month[(month) - 1])
#define DAYS_IN_YEAR(year)      (IS_LEAP_YEAR(year) ? 366 : 365)

#define XXX_DMYHMS_TIME_SIZE        20
#define XXX_DYHMSM_TIME_SIZE        21
#define XXX_ISO8601_TIME_SIZE       20
#define XXX_ISOYDHMS_TIME_SIZE      18
#define XXX_ISOYDHMSF_TIME_SIZE     26
#define XXX_JDHMS_TIME_SIZE         12
#define XXX_MJDHMS_TIME_SIZE        14
#define XXX_ORACLE_TIME_SIZE        14
#define XXX_TIME_T_TIME_SIZE        4
#define XXX_TF_TIME_SIZE            8           
#define XXX_YDHMS_TIME_SIZE         17
#define XXX_YDHMSF_TIME_SIZE        25
#define XXX_YMD_TIME_SIZE           10

#define MAX_YEAR   2099       /* Max year allowed in xxx_TimeConvert routine*/
#define MIN_YEAR   1970       /* Prior to Landsat 1 launch (arbitrary)      */
#define MAX_YDAY   366
#define MIN_YDAY   1
#define MAX_DAY    31
#define MIN_DAY    1
#define MAX_MONTH  12
#define MIN_MONTH  1
#define MAX_HOUR   24
#define MIN_HOUR   0
#define MAX_MIN    59
#define MIN_MIN    0
#define MAX_SEC    59
#define MIN_SEC    0
#define MAX_MICRO  99999999
#define MIN_MICRO  0

/* xxxDMYHMSTime - ASCII format " DD-MMM-YYYY HH:MM:SS.MMMM",
   example "27-May-1997 23:59:59" */
typedef char xxxDMYHMSTime[XXX_DMYHMS_TIME_SIZE + 1];

/* xxxDYHMSMTime - ASCII format " DDD/YYYY HH:MM:SS.MMM",
   example "365/1993 23:59:59.999" */
typedef char xxxDYHMSMTime[XXX_DYHMSM_TIME_SIZE + 1];

/* xxxISO8601Time - ASCII format "YYYY-MM-DDTHH:MM:SSZ",
   example "1993-12-31T12:59:59Z" */
typedef char xxxISO8601Time[XXX_ISO8601_TIME_SIZE + 1];

/* xxxISOYDHMSTime - ASCII format "YYYY-DDDTHH:MM:SSZ",
   example "1993-365T23:59:59Z" */
typedef char xxxISOYDHMSTime[XXX_ISOYDHMS_TIME_SIZE + 1];

/* xxxISOYDHMSFTime - ASCII format "YYYY-DDDTHH:MM:SS.FFFFFFFZ",
   example "1997-365T23:59:59.9999375Z" */
typedef char xxxISOYDHMSFTime[XXX_ISOYDHMSF_TIME_SIZE + 1];

/* xxxJDHMSTime - ASCII format "DDD HH:MM:SS", example "365 23:59:59" */
typedef char xxxJDHMSTime[XXX_JDHMS_TIME_SIZE + 1];

/* xxxMJDHMSTime - ASCII format "DDDDD HH:MM:SS", example "50234 23:59:59" */
typedef char xxxMJDHMSTime[XXX_MJDHMS_TIME_SIZE + 1];

/* xxxOracleTime - ASCII format "YYYYMMDDHHMMSS", example "19930101123456" */
typedef char xxxOracleTime[XXX_ORACLE_TIME_SIZE + 1];

/* time_t - binary UNIX time, seconds since 00:00:00 on 1 Jan 1970,
   example 855619199 */
/* time_t's typedef is in <time.h> */

/* double - binary UNIX time,  seconds since 00:00:00 on 1 Jan 1993 and
   fractional seconds (0-9999375), example 98236799.99999375 */

/* xxxYDHMSTime - ASCII format "YYYY:DDD:HH:MM:SS",
   example "1993:365:23:59:59" */
typedef char xxxYDHMSTime[XXX_YDHMS_TIME_SIZE + 1];

/* xxxYDHMSFTime - ASCII format "YYYY:DDD:HH:MM:SS.FFFFFFF",
   example "1997:365:23:59:59.9999375" */
typedef char xxxYDHMSFTime[XXX_YDHMSF_TIME_SIZE + 1];

/* xxxYMDTime - ASCII format "YYYY-MM-DD", example "1993-12-31" */
typedef char xxxYMDTime[XXX_YMD_TIME_SIZE + 1];

/******************************************************************************
This defines the time formats supported by xxx_TimeConvert(). If
this typedef is changed, be sure to update xxx_TimeConvert.c to match.
******************************************************************************/
typedef enum xxxTimeFormat
{
    XXX_TIME_DMYHMS,        /* xxxDMYHMSTime */
    XXX_TIME_DYHMSM,        /* xxxDYHMSMTime */
    XXX_TIME_ISO8601,       /* xxxISO8601Time */
    XXX_TIME_ISOYDHMS,  /* xxxISOYDHMSTime */
    XXX_TIME_ISOYDHMSF,     /* xxxISOYDHMSFTime */
    XXX_TIME_JDHMS,     /* xxxJDHMSTime */
    XXX_TIME_MJDHMS,    /* xxxMJDHMSTime */
    XXX_TIME_ORACLE,    /* xxxOracleTime */
    XXX_TIME_T,     /* time_t */
    XXX_TIME_TF,        /* double */
    XXX_TIME_YDHMS,     /* xxxYDHMSTime */
    XXX_TIME_YDHMSF,    /* xxxYDHMSFTime */
    XXX_TIME_YMD        /* xxxYMDTime */
}
xxxTimeFormat;


boolean xxx_TimeConvert
(
    void *p_old_time,         /* I: Time to be converted */
    xxxTimeFormat old_format, /* I: Indicates format of p_old_time */
    void *p_new_time,         /* O: Memory where converted time is written
                                    (see NOTES) */
    xxxTimeFormat new_format  /* I: Indicates format of p_new_time */
);

/* utility to check the year, month and day of a given date */
int xxx_check_year_month_day
(
    int year,                   /* I: year (YYYY) */
    int month,                  /* I: month (MM)  */
    int day                     /* I: day (DD)    */
);

#endif
