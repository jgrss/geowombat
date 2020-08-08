#ifndef XXX_TYPES_H
#define XXX_TYPES_H

#include <sys/types.h>

#ifndef NULL
#include <stddef.h>
#endif

typedef signed      char    byte;       /* byte-sized integer */
typedef unsigned    char    ubyte;      /* byte-sized integer */
typedef unsigned    short   boolean;    /* boolean variable */

typedef int xxx_MessageId_TYPE;             /* Message Id */
typedef int xxx_ProcedureId_TYPE;           /* Product id variable */
typedef int xxx_ProgramId_TYPE;             /* Program id variable */
typedef int xxx_ScriptId_TYPE;              /* Script id variable */
typedef int xxx_WorkOrderId_TYPE;           /* Work Order id variable */


#ifndef TRUE
#define TRUE            1
#endif

#ifndef FALSE
#define FALSE           0
#endif

#ifndef ERROR
#define ERROR       -1
#endif

#ifndef WARNING
#define WARNING         -2
#endif

#ifndef SUCCESS
#define SUCCESS     0
#endif

/* EDC DAAC SDS name and vdata name keywords */
#define XXX_INDEX_LISTING  "IndexListing"  /* Returns a list of all vgroups
                                              and their attached objects */

#define XXX_BAND1       ".B10"
#define XXX_BAND2       ".B20"
#define XXX_BAND3       ".B30"
#define XXX_BAND4       ".B40"
#define XXX_BAND5       ".B50"
#define XXX_BAND6       ".B60"
#define XXX_BAND6L      ".B61"
#define XXX_BAND6H      ".B62"
#define XXX_WO_BAND6L   ".B61"
#define XXX_WO_BAND6H   ".B62"
#define XXX_BAND7       ".B70"
#define XXX_BAND8       ".B80"       /* used only by work order band files */
#define XXX_BAND81      ".B81"
#define XXX_BAND82      ".B82"
#define XXX_BAND83      ".B83"

#define XXX_CAL_BAND1      ".C10"
#define XXX_CAL_BAND2      ".C20"
#define XXX_CAL_BAND3      ".C30"
#define XXX_CAL_BAND4      ".C40"
#define XXX_CAL_BAND5      ".C50"
#define XXX_CAL_BAND6      ".C60"
#define XXX_CAL_BAND6L     ".C61"
#define XXX_CAL_BAND6H     ".C62"
#define XXX_WO_CAL_BAND6L  ".C6L"
#define XXX_WO_CAL_BAND6H  ".C6H"
#define XXX_CAL_BAND8      ".C80"
#define XXX_CAL_BAND7      ".C70"
#define XXX_CAL_BAND81     ".C81"
#define XXX_CAL_BAND82     ".C82"
#define XXX_CAL_BAND83     ".C83"

/* MSS calibration gains and biases */
#define XXX_CGB_BAND1   ".GB1"
#define XXX_CGB_BAND2   ".GB2"
#define XXX_CGB_BAND3   ".GB3"
#define XXX_CGB_BAND4   ".GB4"
#define XXX_CGB_BAND5   ".GB5"
#define XXX_CGB_BAND6   ".GB6"
#define XXX_CGB_BAND7   ".GB7"

#define XXX_GEO         ".GEO"

#define XXX_SLO_BAND1   ".O10"
#define XXX_SLO_BAND2   ".O20"
#define XXX_SLO_BAND3   ".O30"
#define XXX_SLO_BAND4   ".O40"
#define XXX_SLO_BAND5   ".O50"
#define XXX_SLO_BAND6   ".O60"
#define XXX_SLO_BAND6L  ".O61"
#define XXX_SLO_BAND6H  ".O62"
#define XXX_SLO_BAND7   ".O70"
#define XXX_SLO_BAND8   ".O80"
#define XXX_SLO_BAND81  ".O81"
#define XXX_SLO_BAND82  ".O82"
#define XXX_SLO_BAND83  ".O83"

#define XXX_PCD         ".PCD"
#define XXX_PCD1        ".PC1"
#define XXX_PCD2        ".PC2"
#define XXX_MSCD        ".MSD"
#define XXX_MSCD1       ".MD1"
#define XXX_MSCD2       ".MD2"

#define XXX_MTL         ".MTL"
#define XXX_MTA         ".MTA"
#define XXX_MTP         ".MTP"
#define XXX_MT1         ".MT1"
#define XXX_MT2         ".MT2"
#define XXX_CPF_ETM     "L7CPF"
#define XXX_NEW_CPF_ETM     "LE07CPF"
#define XXX_CPF_TM_L5   "L5CPF"
#define XXX_NEW_CPF_TM_L5   "LT05CPF"
#define XXX_CPF_TM_L4   "L4CPF"
#define XXX_NEW_CPF_TM_L4   "LT04CPF"
#define XXX_CPF_MSS_L1  "LM1CPF"
#define XXX_NEW_CPF_MSS_L1  "LM01CPF"
#define XXX_CPF_MSS_L2  "LM2CPF"
#define XXX_NEW_CPF_MSS_L2  "LM02CPF"
#define XXX_CPF_MSS_L3  "LM3CPF"
#define XXX_NEW_CPF_MSS_L3  "LM03CPF"
#define XXX_CPF_MSS_L4  "LM4CPF"
#define XXX_NEW_CPF_MSS_L4  "LM04CPF"
#define XXX_CPF_MSS_L5  "LM5CPF"
#define XXX_NEW_CPF_MSS_L5  "LM05CPF"

#define XXX_HDF_DIR_EXTENSION "_HDF" /* EDC DAAC HDF directory extension */
#define XXX_WO_ENV_VAR  "PCS_WOID"   /* name of work order id environment
                                        variable*/

#define XXX_PR_ENV_VAR  "LPGS_PR_ID" /* name of LPGS product request id
                                        environment variable */

#define XXX_IAS_SYSTEM_ID "IAS_SYSTEM_ID" /* name of system user */

#define XXX_PR_ENV_VAR_LEN  10       /* length of name of PR ID environment
                                        variable */

#define XXX_L0R_PRODUCT 0
#define XXX_L1_PRODUCT  1

/* system macros */
/* Macro XXX_ARRAY_SIZE will return the size of an array */
#define XXX_ARRAY_SIZE(array)   (sizeof(array) / sizeof(*(array)))
/* Macro XXX_INTERLACE_STRING returns ascii string for interlace value */
#define XXX_INTERLACE_STRING(Interlace) ((Interlace)  == 0 ? "FULL_INTERLACE" : "NO_INTERLACE")
/* right-shift macro for system() calls */
#define XXX_RIGHT_SHIFT(status) (((status)>>8) & 0xff)
#endif
