#ifndef XDB_DEFINES_H
#define XDB_DEFINES_H

/***** Database specific parameters                     */

#define XDB_ORAERRMSGLEN    512 /* Max len allowed for DBMS error string    */
#define XDB_NOTCONNECTED  -1012 /* not connected to database                */
#define XDB_NOTFOUND       1403 /* DBMS-defined value for data not found    */
#define XDB_SUCCESS           0 /* DBMS-defined value for successful
                                   dbms transaction              */
#define XXX_PRODUCT_REQUEST_ID_LEN 20 /* PRODUCT_REQUEST_ID for LPGS        */
#define XDB_NON_ERROR       99  /* Indicates db unit error not caused by
                                   dbms exception (e.g. out of memory)
                                   DB unit will set xxx_errno to appropriate
                                   error classification - see xxx_Errno.h   */
#define XDB_WARNING     99999   /* DB unit return error val if db warning
                                   found */
#define XDB_FETCH_SIZE      100 /* Number of records per array fetch        */
#define XDB_INSERT_SIZE     100 /* Number of records per array insert       */
#define XDB_PARSE_STMT_SIZE 2048    /* Size dynamic sql parse string        */
#define XDB_NULL        -1  /* Value returned if a retrieved db field = NULL*/
#define XDB_NULLSTR     ' ' /* Value returned if a retrieved db field = NULL*/
#define XDB_NULL_ZERO       0   /* a retrieved db field = NULL          */
#define XDB_NULL_ONE        1   /* Value returned if a retrieved db
                                   field = NULL*/
#define XDB_ALL_ROWS        99  /* Indicates retrieve all rows          */



/***** Miscellaneous defines                            */

#define XDB_WARN_MSG     "Fetched value was truncated."/* db warning message*/
#define XDB_NOTFOUND_MSG "ORA-01403: no data found"  /*no data found message*/

/***** Auto-Work Order defines                          */

#define AUTO_WO_SMGET_FMT  \
   "Error in getting sub_module record for product request ID %s::%s"
#define AUTO_WO_ROLLBACK_FMT  \
   "Cannot roll back changes to DB for product request ID %s::%s"
#define AUTO_WO_SMGET_ERR   2340
#define AUTO_WO_DBVAL_FMT2  \
   "The returned value of %s has been truncated from %d characters"
#define AUTO_WO_DBVAL_ERR    2331
#define AUTO_WO_INSERTWS_FMT  \
   "Could not insert a new WO_SCRIPTS record for product request ID %s::%s"
#define PWG_GETTIME_FMT  \
   "Error in getting current time"
#define PWG_GETTIME_ERR   2337
#define PWG_INSERTWO_FMT  \
   "Could not insert a new WO record for product request ID %s::%s"
#define PWG_INSERTWO_ERR   2338
#define PWG_BADPRODTYPE_ERR   2342
#define PWG_BADPRODTYPE_FMT  \
   "Supplied product type %s for product request ID %s is invalid"


#ifndef FAILURE
#define FAILURE                 -1
#endif

/* define for null string */
#ifndef XXX_NULL_STRING
#define XXX_NULL_STRING         ""
#endif

#ifndef LPGS
#define XDB_WOS_NOPAUSE     0

#define XXX_REQ_ID_LEN          20 /* Length of the Product Request ID */

/***** Auto-Work Order Database defines                */

#define DUMMY_REQUESTOR        "DUMMY_REQ"
/* format to use when using xxx_GetTime to get current time for DB */
#define XXX_DBTIME_FMT          "%d-%b-%Y %H:%M:%S"

#define AUTO_WO_INSERTWS_ERR   2341 
#define XDB_PR_REQ_TYPE_TT      "TT"
#define XDB_WO_TROUBLE_TICKET   "T"
#define XDB_WO_AAS_CREATED      "A"
#define XDB_PR_REQ_TYPE_BENCH   "BEN"
#define XDB_WO_BENCHMARK        "B"
#define XDB_WO_NOMINAL          "S"

#endif // #ifndef LPGS

/***** Macros                                   */

#define XDB_IF_WARNING  ( (sqlca.sqlwarn[1] == ' ') ? SUCCESS : ERROR)

#define XXX_EPHEMERIS_FILE_SIZE             19
#define XXX_EPHEMERIS_TYPE_SIZE             1
#define XXX_FILE_TYPE_SIZE                  4
#define XXX_HOST_ADDRESS_SIZE               30
#define XXX_HOST_NAME_SIZE                  15
#define XXX_INVOICE_SIZE                    50   /* ORDER_ID */
#define XXX_L0R_HDFNAME_SIZE                50
#define XXX_L0R_PRODUCT_ID_SIZE             15
#define XXX_MAXPATH_SIZE                    256
#define XXX_MESSAGE_SIZE                    256
#define XXX_PASSWORD_SIZE                   30
#define XXX_PRODUCT_ID_SIZE                 15
#define XXX_PROGRAM_ID_SIZE                 25
#define XXX_SCRIPT_ID_SIZE                  25
#define XXX_SCRIPT_EXIT_STATUS_SIZE         15
#define XXX_SOURCE_FILE_SIZE                80
#define XXX_TARGET_PATH_SIZE                80
#define XXX_WORKORDER_ID_SIZE               12
#define XXX_WORKORDER_STATE_SIZE            2
#define XXX_TRANS_TYPE_SIZE                 3
#define XXX_USERID_SIZE                     15
#define XXX_WOPARMNAME_SIZE             25          
#define XXX_WOPARMVALUE_SIZE                256
#define XXX_PR_ID_SIZE                      21
#define XXX_SYSTEM_ID_SIZE                   3
#define XXX_SCENE_ID_SIZE                   21

/* IAS Configuration file types */
#define XDB_DAN           "DAN"    /* DDM path */
#define XDB_DAAC          "DAAC"   /* DAAC path */
#define XDB_CPF           "CPF"    /* CPF path */
#define XDB_EMOC          "EMOC"   /* error MOCC path */
#define XDB_L0R           "L0R"    /* L0R product path */
#define XDB_L1RG          "L1RG"   /* L1R/L1G product path */
#define XDB_MISC          "MISC"   /* misc. path */
#define XDB_RPT           "RPT"    /* reports path */
#define XDB_SCRP          "SCRP"   /* wo script path */
#define XDB_SCRTP         "SCRT"   /* wo script test path */
#define XDB_TAR           "TAR"    /* temp tar path */
#define XDB_TMOC          "TMOC"   /* temp MOCC path */
#define XDB_WO            "WO"     /* work order path */

/* L0R Product deletion states */
#define XDB_DONT_DELETE    0
#define XDB_OK_TO_DELETE   1
#define XDB_USER_DELETED   2
#define XDB_DELETED        3

#endif
