#ifndef XXX_OPENMAP_H
#define XXX_OPENMAP_H

#ifndef _SYS_MMAN_H
#include <sys/mman.h>
#endif

#include <xxx_Types.h>

/* error conditions */
#define XXX_OPEN_ERR        1   /* File open error */
#define XXX_STAT_ERR        2   /* File status error */
#define XXX_MMAP_ERR        3   /* Memory map error */
#define XXX_MMAP_ZERO_ERR   4   /* Indicates zero file size on memory map
                                   request */
int xxx_OpenMap
(
    char *p_filename,   /* I: Name of file to open */
    int *p_fd,          /* O: File descriptor */
    int file_options,   /* I: File options for opening */
    void **p_pa,        /* O: Memory map address */
    off_t file_start,   /* O: Start offset of file or -1 for map enter file */
    long *p_file_len,   /* O: File length */
    long *p_remainder,  /* O: Byte offset into the page, used only if not
                              defaulting */
    boolean mapflag,    /* I: Mapping indicator */
    boolean lockflag,   /* I: locking indicator */
    char *p_ErrorMessage/* O: Error Message */
);

#endif
