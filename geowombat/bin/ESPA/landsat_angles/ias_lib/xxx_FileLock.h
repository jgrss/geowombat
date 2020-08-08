#ifndef XXX_FILELOCK_H
#define XXX_FILELOCK_H

#ifndef _FCNTL_H
#include <fcntl.h>
#endif

/* Set read lock, this will prevent another process from write-locking 
   the file */
#define XXX_READ_LOCK(fd, offset, whence, length) \
        xxx_FileLock(fd, F_SETLK, F_RDLCK, offset, whence, length)

/* Set read lock with wait option this will prevent another process from 
   write-locking the file */
#define XXX_READ_WAIT_LOCK(fd, offset, whence, length) \
        xxx_FileLock(fd, F_SETLKW, F_RDLCK, offset, whence, length)

/* Set write lock, this will prevent another process from writing to the 
   file */
#define XXX_WRITE_LOCK(fd, offset, whence, length) \
        xxx_FileLock(fd, F_SETLK, F_WRLCK, offset, whence, length)

/* Set write lock with wait option, this will prevent another process from 
   writing to the file */
#define XXX_WRITE_WAIT_LOCK(fd, offset, whence, length) \
        xxx_FileLock(fd, F_SETLKW, F_WRLCK, offset, whence, length)

/* Unlock file */
#define XXX_UNLOCK(fd, offset, whence, length) \
        xxx_FileLock(fd, F_SETLK, F_UNLCK, offset, whence, length)


int xxx_FileLock
(
    int fd,       /* I: File descriptor */
    int command,  /* I: fcntl commands: F_SETLK or F_SETLK */
    short type,   /* I: Type of lock desired: F_RDLCK, FWRLCK, or F_UNLOCK */
    off_t offset, /* I: Byte offset, relative to l_whence */
    short whence, /* I: SEEK_SET, SEEK_CUR, SEEK_END */
    off_t length  /* I: Number of bytes (0 means to EOF) */
);

#endif
