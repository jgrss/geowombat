#ifndef XXX_GETTMPNAME_H        
#define XXX_GETTMPNAME_H

char *xxx_GetTempName
(
    const char *p_dir,  /* I: Name of the directory in which the file is to
                              be created */
    const char *p_pfx   /* I: Favorite initial letter sequences (up to 5
                              characters) */
);

#endif

