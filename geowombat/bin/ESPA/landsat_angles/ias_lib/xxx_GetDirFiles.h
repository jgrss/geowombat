#ifndef XXX_GETDIRFILES_H
#define XXX_GETDIRFILES_H

#include <sys/param.h>

typedef struct xxx_DirList_TYPE
{
    char FilePathName[MAXPATHLEN];
} xxx_DirList_TYPE;


xxx_DirList_TYPE *xxx_GetDirFiles
(
    int *p_Count,        /* O: Number of rows in p_PartionList */
    char *p_PathName,    /* I: Directory path name */
    char *p_Key,         /* I: List options like "*" */
    char *p_ErrorMessage /* O: Error message */
);

#endif
