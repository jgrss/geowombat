#ifndef XXX_ODL_H
#define XXX_ODL_H

#define NOT_ENOUGH_BUFFER_SIZE    -1

#include <lablib3.h>
#include "xxx_Types.h"
#include "xdb_Defines.h"

/* Warning status codes returned from xxx_GetODLField() */
#define XXX_E_NOT_ENOUGH_MEMORY_SUPPLIED        2 
#define XXX_E_ODL_NOT_FOUND                     3
#define XXX_E_ODL_INVALID_DATA_TYPE             4 

#define XXX_ADDED_KEYWORD                      'A'
#define XXX_DELETED_KEYWORD                    'K'
#define XXX_MODIFIED_KEYWORD                   'M'

#define XXX_NEW_GROUP                          'N'
#define XXX_DELETED_GROUP                      'G'

/* Constants for global work order param names */
#define XXX_PARM_WO_DIR         "WO_DIRECTORY"
#define XXX_PARM_CPF            "CAL_PARM_FILE"
#define XXX_PARM_EPHEM_FILE     "FDF_NAME"
#define XXX_PARM_EPHEM_TYPE "EPHEM_TYPE"
#define XXX_PARM_L0R_HDFNAME    "L0R_HDFNAME"
#define XXX_PARM_L1R_IMAGE      "L1R_IMAGE"
#define XXX_PARM_SCRIPT_NAME    "SCRIPT_NAME"
#define XXX_PARM_L0R_ID         "L0R_ID"
#define XXX_PARM_PR_ID      "PRODUCT_REQUEST_ID"
#define XXX_PARM_WO_ID      "WORK_ORDER_ID"

/* Flag passed to xxx_OpenODL() indicating the type of ODL file */
typedef enum
{
    xxx_NoEDCMeta,          /* regular odl file */
    xxx_EDCMeta,            /* L0R's EDC Metadata file */
    xxx_LPGSMeta,           /* L1R's LPGS Metadata file */
    xxx_LPSMeta0,           /* L0R's LPS Metadata file (TM-A) */
    xxx_LPSMeta1,           /* L0R's LPS Metadata file */
    xxx_LPSMeta2,           /* L0R's LPS Metadata file */
    xxx_CPFMeta             /* L0R's Cal Parm File */
} xxx_EDCOdl_TYPE;

/* Flag passed to xxx_GetODLField() indicating the ODL keyword's expected
   data type */
typedef enum
{
    XXX_Double, 
    XXX_ArrayOfString, 
    XXX_String,
    XXX_Int, 
    XXX_Long, 
    XXX_Float,
    XXX_Sci_Not
} xxx_ValType_TYPE;

typedef struct
{
    char      Object_Name[XXX_MAXPATH_SIZE+1];
    char      *p_GroupFileName; /* Group file name (DCM task only) */
    char      *p_DerivedFrom;   /* Calibration parameter file the group was
                                   extracted from (DCM task only) */

    char      Kwd_Name[XXX_MAXPATH_SIZE+1];
    char      Pre_Kwd_Name[XXX_MAXPATH_SIZE+1];
    char      *Old_Value;
    char      *New_Value;
    char      Pre_Obj[XXX_MAXPATH_SIZE+1];
    int       Pre_Level;
    char      Flag;
    int       Level;
    char      Message[XDB_ORAERRMSGLEN+1]; /* Problem found when adding,
                                              deleting, or modifying group or
                                              keyword */
} xxx_Diff_TYPE;


/* TYPE DEFINITIONS */

typedef enum
{
    xxx_ODL_PARAM_OPTIONAL,
    xxx_ODL_PARAM_REQUIRED
} xxx_ODL_PARAM_REQUIRED_TYPE;
/* enumerated type for defining whether an item in a xxx_READ_ODL_LIST_TYPE is
   required or optional. */

typedef struct
{
    char *group_name;  /* group name of parameter */
    char *param_name;  /* name of parameter */
    void *param_ptr;   /* location to return data read from ODL file for
                          this parameter */
    int param_size;    /* number of bytes at param_ptr */
    xxx_ValType_TYPE value_type; /* type of parameter */
    int min_count;     /* minimum count of items to read. 0 = no minimum. */
    xxx_ODL_PARAM_REQUIRED_TYPE required; /* flag to indicate the parameter is
                                             either optional or required */
} xxx_READ_ODL_LIST_TYPE;
/* type definition for the list of parameters to read from an ODL file for
   a call to xxx_read_odl_file. */

int xxx_read_odl_file
(
    const char *odl_file_name, /* I: ODL file name to read */
    int find_cpf,              /* I: set this flag to non-zero to indicate
                                     the file name is a 0R filename and the CPF
                                     should be read from it */
    xxx_READ_ODL_LIST_TYPE *odl_list_ptr, /* I/O : ptr to list of items to
                                                   read from the ODL file */
    int list_length            /* I: number of items in the list */
);


int xxx_BuildWOParmODL
(
    char *p_Pathname,     /* I: Pathname to store the ODL file */
    char *p_Work_ID,      /* I: Work Order Id number */
    char *p_Script_ID,    /* I: Script Name (ODL parameter:SCRIPT_NAME) */
    char *p_WO_Dir,       /* I: WO directory name
                                (ODL parameter: WO_DIRECTORY) */
    char *p_L0R_ID,       /* I: L0R id (ODL parameter: L0R_ID) */ 
    char *p_CPF,          /* I: Calibration parameter filename (optional) (ODL
                                        parameter:CAL_PARM_FILE) */
    char *p_L0R_HDFName,  /* I: L0R HDF filename
                                (ODL parameter: L0R_HDFNAME) */
    char *p_ephem_type,   /* I: Ephemeris type (P = PCD, D = definitive) */
    char *p_PR_ID,        /* I: Product Request ID */
    char *p_ErrorMessage  /* I: Error Messages */
);

char *xxx_FindFileName
(
    char *p_PathName      /* I: Pathname */
);

OBJDESC *xxx_OpenODL
(
    char *p_ODLFile,         /* I: If a normal odl file is to be opened, the
                                   path is to the ODL file otherwise the path
                                   is to the L0R directory file */
    xxx_EDCOdl_TYPE ODLType, /* I: Indicates if a ODL file to read or 
                                   a normal odl file (ie.xxx_NoEDCMeta) */
    char *p_ErrorMessage     /* O: Error message */
);

int xxx_CloseODL
(
    OBJDESC *p_lp,       /* I: ODL tree */
    char *p_ErrorMessage /* O: Error message */
);

int xxx_GetODLField
(
    void *p_MemoryAddr,         /* I: p_Memory to copy fields into */
    int MemorySize,             /* I: Size in bytes of p_Memory */
    xxx_ValType_TYPE ValueType, /* I: What type the P_FieldName is, enum
                                      field */
    OBJDESC *p_ODLTree,         /* I: ODL tree */
    char *p_ClassName,          /* I: Group/Object name, optional */
    char *p_LabelName,          /* I: Field to retrieve */
    char *p_ErrorMessage,       /* O: Error Message */
    int *p_Count                /* O: Count the number of values in a array */
);

int xxx_ConvertString
(
    void *p_destination,        /* O: Location where to copy converted
                                      values */
    xxx_ValType_TYPE parm_type, /* I: What type the p_destination is, enum
                                      field */
    char *kvalue,               /* I: String to be converted */
    char *p_ErrorMessage        /* O: Error Message */
);

int xxx_DiffCPF
(
    OBJDESC *p_BaseCPF,       /* I: ODL tree for the base CPF file */
    OBJDESC *p_NewCPF,        /* I: ODL tree for the new CPF file */
    xxx_Diff_TYPE **p_Values, /* O: Array of diff. between base CPF and new
                                    CPF */
    int *p_count,             /* O: Count the number of rows */
    char *p_errmsg            /* O: Error message */
);

int xxx_PrintDiff
(
    char *p_baseline,           /* I: Names of the baseline file{s} */
    char *p_newfile,            /* I: Name of the new file */
    int count,                  /* I: Numbers of rows in p_UpdValues */
    xxx_Diff_TYPE *p_UpdValues, /* I: Array of diff to print */
    boolean Trunc,              /* I: Truncate file names */
    char *p_errmsg              /* O: Error message */
);

void xxx_FreeDiffCPF
(
    int *p_Count,             /* I: Count the number of rows */
    xxx_Diff_TYPE **p_Values, /* I: Array of diff. between base CPF and
                                    new CPF */
    boolean FreeValues        /* I: Flag to determine whether to free the
                                    p_Values */
);

#endif

