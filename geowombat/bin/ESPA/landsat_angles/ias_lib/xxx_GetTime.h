#ifndef XXX_GETTIME_H
#define XXX_GETTIME_H


#define TIMELEN 24

int xxx_GetTime
(
    char *p_stamp,        /* I/O: Pointer to string to place time/date */
    int stampsize,        /* I: Size of p_stamp */
    const char *p_format  /* I: Pointer to format string */
);

void xxx_GetTime_system
(
    char *s  /* O: formatted time string */
);

void xxx_psfTime
(
    int sat,                /* I: Satellite */
    double sec,             /* I: Seconds since mission epoch */
    char s[TIMELEN]         /* O: formatted time string: MM/DD/YYYY HH:MM:SS*/
);

#endif
