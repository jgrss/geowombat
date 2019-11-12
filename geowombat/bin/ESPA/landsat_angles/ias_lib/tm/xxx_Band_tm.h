#ifndef xxx_BandGetInfo_tm_H
#define xxx_BandGetInfo_tm_H

#define BAND_PROG_NAME_TM "xxx_BandGetInfo_tm"

#define MAX_BANDS_TM               7
#define MAX_LOGICAL_BANDS_TM       7
#define NUM_REFLECTIVE_BANDS_TM    6 /* 1 2 3 4 5 7 */
#define NUM_RESOLUTIONS_TM         2 /* 30, 120 */
#define NUM_LAMP_STATES_TM         8 /* allowing for a "0" state */

void xxx_BandGetInfo_tm ();

void xxx_CreateBandNode_tm 
(
    BAND_HDR_TYPE *band_list, /* I/O: the band list to put the band node in */
    BAND_OPTION_TYPE option,  /* I: standard band list, or customized one   */
    BandNumber band_number    /* I: the number of the band to add           */
);

#endif
