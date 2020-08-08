#ifndef XXX_LOGSTATUS_H
#define XXX_LOGSTATUS_H

void xxx_LogStatus
(
    const char *p_program,  /* I: Program name */
    const char *p_function, /* I: Filename of calling unit 
                                  (see C macro __FILE__) */
    int LineNumber,         /* I: Line number of calling unit 
                                  (see C macro __LINE__) */
    const char *p_msg       /* I: Error message to write to standard out */
);

#endif
