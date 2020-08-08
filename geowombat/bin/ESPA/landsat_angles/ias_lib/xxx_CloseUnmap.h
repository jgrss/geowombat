#ifndef XXX_CLOSEUNMAP_H
#define XXX_CLOSEUNMAP_H

#ifndef _SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifndef XXX_TYPES_H
#include <xxx_Types.h>
#endif

/* error conditions */
#define XXX_CLOSE_ERR   5   /* Indicates file close error */
#define XXX_UNMAP_ERR   6   /* Indicates unmap error */

int xxx_CloseUnmap
(
    int fd,              /* I: File descriptor */
    void *p_pa,          /* I: Memory map address */
    long file_len,       /* I: File length */
    boolean mapflag,     /* I: Mapping indicator */
    boolean lockflag,    /* I: locking indicator */
    char *p_ErrorMessage /* O: Error Message */
);

#endif
