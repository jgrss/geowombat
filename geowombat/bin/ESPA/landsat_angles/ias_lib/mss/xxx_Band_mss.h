#ifndef XXX_BANDGETINFO_MSS_H
#define XXX_BANDGETINFO_MSS_H

#define BAND_PROG_NAME_MSS "xxx_BandGetInfo_mss"

#define MAX_BANDS_MSS              4
#define MAX_LOGICAL_BANDS_MSS      4
#define NUM_REFLECTIVE_BANDS_MSS   4 /* 4 5 6 7 */
#define NUM_RESOLUTIONS_MSS        1 /* 60 */
#define NUM_LAMP_STATES_MSS        0 /* allowing for a "0" state */

void xxx_BandGetInfo_mss ();

void xxx_CreateBandNode_mss
(
    BAND_HDR_TYPE *band_list, /* I/O: the band list to put the band node in */
    BAND_OPTION_TYPE option,  /* I: standard band list, or customized one   */
    BandNumber band_number    /* I: the number of the band to add           */
);

#endif
