#ifndef XXX_ERRNO_H
#define XXX_ERRNO_H

extern char *xxx_errlist[];
extern long xxx_nerr;
extern long xxx_errno;

#define XXX_E_NO_MEMORY             0
#define XXX_E_NULL_INPUT            1
#define XXX_E_NO_MORE_OBJECTS       2
#define XXX_E_OBJECT_NOT_FOUND      3
#define XXX_E_OUT_OF_RANGE          4
#define XXX_E_INVALID_DATA          5
#define XXX_E_TIMEOUT               6
#define XXX_E_FAILURE               7
#define XXX_E_INSUFFICIENT_INPUT    8
#define XXX_E_FILE_NOT_FOUND        9
#define XXX_E_ODL_SYNTAX           10

#endif
