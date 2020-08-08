#ifndef GXX_ANGLE_GEN_GEO_GET_LEAP_SECONDS_H
#define GXX_ANGLE_GEN_GEO_GET_LEAP_SECONDS_H

#include "gxx_angle_gen_distro.h"

typedef struct gxx_leap_seconds_data_TYPE
{
    int leap_seconds_count;         /* number leap seconds since 1972 */
    int *leap_years;                /* list of years of occurance */
    int *leap_months;               /* list of months of occurance (1-12) */
    int *leap_days;                 /* list of days of occurance (1-31) */
    int *num_leap_seconds;          /* number of leap seconds added at each
                                       time */
} gxx_leap_seconds_data_TYPE;

int math_convert_year_doy_sod_to_j2000_seconds
(
    const double *time, /* I: Year, day of year, seconds of day to convert to
                              J2000 epoch time */
    double *j2secs      /* O: TAI seconds from J2000 */
);

int get_leap_seconds_at_year_doy_sod
(
    int year,               /* I: Year to calculate leap seconds */
    int doy,                /* I: Day of year in calculation */
    double sod,             /* I: Seconds of day */
    int time_is_tai,        /* I: Flag that indicates the reference time is in
                                  TAI and therefore needs to have leap seconds
                                  applied to use the leap second table that is
                                  in UTC */
    const gxx_leap_seconds_data_TYPE *leap_seconds,
                            /* I: Leap seconds info from CPF */
    int *num_leap_seconds   /* O: Num relevant leap seconds from CPF info */
);

int gxx_angle_gen_get_leap_seconds
(
    gxx_leap_seconds_data_TYPE *leap_seconds /* O: leap seconds Params */
);

int gxx_angle_gen_init_leap_seconds_from_UTC_time
(
    gxx_epoch_time_TYPE *ref_time,                 /* I: Year, day of year, 
                                                      seconds of day to use to
                                                      intialize leap seconds */
    const gxx_leap_seconds_data_TYPE *leap_seconds /* I: Leap seconds info */
);

#endif
