#ifndef xxx_BandGetInfo_etm_H
#define xxx_BandGetInfo_etm_H

#define BAND_PROG_NAME_ETM "xxx_BandGetInfo_etm"

#define MAX_BANDS_ETM            8 /* does not include 62     */
#define MAX_LOGICAL_BANDS_ETM    9 /* includes both 61 and 62 */
#define NUM_REFLECTIVE_BANDS_ETM 7 /* 1 2 3 4 5 7 8           */
#define NUM_RESOLUTIONS_ETM      3 /* 15, 30, 60              */
#define NUM_LAMP_STATES_ETM      3

void xxx_BandGetInfo_etm ();

void xxx_CreateBandNode_etm 
(
    BAND_HDR_TYPE *band_list, /* I/O: the band list to put the band node in */
    BAND_OPTION_TYPE option,  /* I: standard band list, or customized one   */
    BandNumber band_number    /* I: the number of the band to add           */
);

#endif
