#ifndef XXX_IMAGERY_H
#define XXX_IMAGERY_H

#include <limits.h>
#include <sys/param.h>
#include <hdf.h>
#include "xxx_Const.h"
#include "xxx_Types.h"

/* Hack to allow the MSS formatter to build with HDF 4.1. */
#ifdef MSS_FORMATTER_HDF41
#define H4_MAX_NC_NAME MAX_NC_NAME
#define H4_MAX_VAR_DIMS MAX_VAR_DIMS
#endif

#define XXX_MAX_RANK  2

typedef struct xxx_SubDivideInfo_TYPE
{
    int32       StartOffset;
    int32       RowsToRead;
    int32       ByteOffset;
    int32       BytesToRead;
    int32       CurrectRow;
} xxx_SubDivideInfo_TYPE;

typedef struct xxx_ImageryInfo_TYPE
{
    boolean     EDC_DAAC_HDF;               /* opened a HDF directory file */
    int32       FileId;                     /* for vgroup checks */
    int32       SDS_Id;
    int32       SDId;
    char        SDSName[H4_MAX_NC_NAME];
    int32       DataType;
    int32       Rank;
    int32       DimensionSizes[H4_MAX_VAR_DIMS];
    char        DirectoryPath[MAXPATHLEN];
    xxx_SubDivideInfo_TYPE SD;
} xxx_ImageryInfo_TYPE;


int xxx_ChkForSDS
(
    xxx_ImageryInfo_TYPE *p_ImageryInfo, /* I: HDF imagery information */
    int ProductType,                 /* I: Indicates L0R or L1 product type */
    char *p_ErrorMessage             /* O: Error message */
);

int xxx_ChkForSDSInHDF
(
    xxx_ImageryInfo_TYPE *p_ImageryInfo, /* I: HDF imagery info */
    int ProductType,                 /* I: Indicates L0R or L1 product type */
    char *p_ErrorMessage             /* O: Error message */
);

long xxx_GetDataTypeSize
(
    int32 DataType /* I: HDF data type */
);

int xxx_ConvertDataType
(
    void *inbuf,         /* I: Input buffer */
    void *outbuf,        /* O: Output buffer; in-place conversion if NULL */
    unsigned int n,      /* I: Number of data samples */
    DataType idtype,     /* I: Input data type */
    DataType odtype,     /* I: Output data type */
    unsigned int *nlow,  /* O: Number of samples clipped on the low end */
    unsigned int *nhigh, /* O: Number of samples clipped on the high end */
    char *msg            /* O: Status message */
);

int xxx_ConvertDataTypeToString
(
    DataType data_type,              /* I: IAS data type */
    const char **data_type_string    /* O: Data type converted to a string */
);

int xxx_ConvertStringToDataType
(
    const char *data_type_string,    /* I: String data type */
    DataType *data_type              /* O: String to its IAS data type */
);

unsigned int xxx_GetIASDataTypeSize
(
    DataType dtype  /* I: IAS data type */
);

unsigned int xxx_GetHDFDataType
(
    DataType dtype  /* I: IAS data type */
);

DataType xxx_GetIASDataType
(
    unsigned int dtype  /* I: HDF data type */
);

int xxx_OpenImagery
(
    char *p_PathName,                    /* I: Path name to SDS HDF file */
    int32 AccessMode,                    /* I: HDF access mode */
    char *p_SDSName,                     /* I: SDS name to attach to or
                                               create */
    int32 offset,                        /* I: Where to start writing in
                                               external file */
    char *p_ExternalFile,                /* I: External file path */
    xxx_ImageryInfo_TYPE *p_ImageryInfo, /* I/O: HDF imagery information */
    char *p_ErrorMessage                 /* O: Error message  */
);

int xxx_CloseImagery
(
    xxx_ImageryInfo_TYPE *p_ImageryInfo, /* I: HDF imagery information */
    char *p_ErrorMessage                 /* O: Error message */
);

void xxx_ConvertToString
(
    char *output_string,  /* O: Output string (with null termination) */
    char *input_array,    /* I: Input array of chars to convert */
    int max_chars         /* I: Maximum number of chars to be converted */
);

int xxx_read_image
(
    const char *filename,             /* I: Filename */
    const char *imgname,              /* I: Band identifier */
    xxx_ImageryInfo_TYPE *sceneInfo,  /* O: Scene file info */
    void **image_buf,                 /* O: Image data */
    char *msg                         /* O: Error message */
);

int xxx_ReadImagery
(
    xxx_ImageryInfo_TYPE *p_ImageryInfo, /* I/O: HDF imagery information */
    void *p_Imagery,       /* I/O: Array of data read in from the file */
    int32 RowsToRead,      /* I: Number of rows to read */
    int32 *p_StartOffset,  /* I: Position where the read will start, the
                                 p_StartOffset is increment by RowsToRead */
    int32 FirstCol,        /* I: Position were the read will start for column,
                                 if not subsetting set to -1 */
    int32 ColsToRead,      /* I: Number of columns to read
                                 if not subsetting set to -1 */
    char *p_ErrorMessage   /* O: Error message */
);

void xxx_TrimString
(
    char *string  /* I/O: String to trim */
);

int xxx_WriteImagery
(
    xxx_ImageryInfo_TYPE *p_ImageryInfo, /* I/O: HDF imagery information */
    void *p_Imagery,           /* I: Array of data to write to file */
    int32 RowsToWrite,         /* I: Number of rows in imagery */
    int32 StartOffset,         /* I: Position where the write will start */
    char *p_ErrorMessage       /* O: Error message */
);

#endif
