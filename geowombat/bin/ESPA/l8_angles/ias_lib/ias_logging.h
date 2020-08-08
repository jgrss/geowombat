#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>

/****************************************************************************
Notes on use:
    - All applications should call ias_log_initialize
    - The IAS_LOG_* macros should be used to issue log messages.
      ias_log_message should not be called directly, except in unusual
      circumstances.
    - The level of messages can be controlled by setting the IAS_LOG_LEVEL 
      environment variable to DEBUG, INFO, WARN, or ERROR.  If the environment
      variable is not set, the default level of output is INFO.
    - If you want to change the output level for specific routines during 
      debugging, you can call ias_log_set_output_level with an IAS_LOG_LEVEL_*
      setting.  But that should not be done on a permanent basis.
    - The child processor needs to call ias_log_initialize to get the pid
      set properly.
    - If you want a debug log message written to a specific channel, in the
      code calling the logging functions define a IAS_LOG_CHANNEL constant
      before including this header file.  The value of IAS_LOG_CHANNEL 
      should be the name of the channel.  If IAS_LOG_CHANNEL is defined all 
      debug messages will be written to that channel. 
    - The IAS_LOG_CHANNELS environment variable is used to enable channels
      for an application.  If IAS_LOG_CHANNELS is set, the logging library 
      expects the value to be a comma separated list of channels to enable
      and any channels not listed are disabled. If the initial character is
      a '-'  character, the list is treated as a blacklist and all channels
      will be enabled except the listed channels. If IAS_LOG_CHANNELS is not 
      set, all channels are enabled. 

****************************************************************************/
/* Allow GCC to error check the parameters to the ias_log_message routine
   like it is a printf statement */
#ifdef SWIG
#define PRINT_FORMAT_ATTRIBUTE
#define PRINT_FORMAT_ATTRIBUTE_WC
#else
#define PRINT_FORMAT_ATTRIBUTE  __attribute__ ((format(printf,4,5)))
#define PRINT_FORMAT_ATTRIBUTE_WC  __attribute__ ((format(printf,5,6)))
#endif

/* Definition of supported log message levels */
typedef enum IAS_LOG_MESSAGE_LEVEL {
   IAS_LOG_LEVEL_DEBUG = 0, 
   IAS_LOG_LEVEL_INFO, 
   IAS_LOG_LEVEL_WARN, 
   IAS_LOG_LEVEL_ERROR,
   IAS_LOG_LEVEL_DISABLE  /* disable logging entirely (only used for tests) */
} IAS_LOG_MESSAGE_LEVEL;

/* Declare the ias_log_message_level variable as an external variable for
   every file except the .c file where is is declared */
#ifndef LOGGING_C
extern enum IAS_LOG_MESSAGE_LEVEL ias_log_message_level;
#endif

int ias_log_initialize
(
    const char *log_program_name /* I: name to output with each log message */
);

IAS_LOG_MESSAGE_LEVEL ias_log_set_output_level
(
    int new_level   /* I: minimum logging level to output */
);

int ias_log_set_output_target
(
    FILE *new_fp    /* I: File pointer for output message */ );

void ias_log_message 
(
    int log_level,            /* I: message level for input */
    const char *filename,     /* I: source code file name for input */
    int line_number,          /* I: source code line number for input */
    const char *format, ...   /* I: format string for message */
) PRINT_FORMAT_ATTRIBUTE; 

void ias_log_message_with_channel
(
    int log_level,            /* I: message level for input */
    const char *channel,      /* I: message channel name */
    const char *filename,     /* I: source code file name for input */
    int line_number,          /* I: source code line number for input */
    const char *format, ...   /* I: format string for message */
) PRINT_FORMAT_ATTRIBUTE_WC; 


/************************************************************************/
#define IAS_LOG_DEBUG_ENABLED() \
    (((IAS_LOG_LEVEL_DEBUG) >= (ias_log_message_level)) ? (1) : (0))

/************************************************************************/
#define IAS_LOG_ERROR(format,...) \
ias_log_message(IAS_LOG_LEVEL_ERROR,__FILE__,__LINE__,format,##__VA_ARGS__)

/************************************************************************/
#define IAS_LOG_WARNING(format,...) \
    if (IAS_LOG_LEVEL_WARN >= ias_log_message_level)          \
        ias_log_message(IAS_LOG_LEVEL_WARN,__FILE__,__LINE__, \
                        format,##__VA_ARGS__)

/************************************************************************/
#define IAS_LOG_INFO(format,...) \
    if (IAS_LOG_LEVEL_INFO >= ias_log_message_level)          \
        ias_log_message(IAS_LOG_LEVEL_INFO,__FILE__,__LINE__, \
                        format,##__VA_ARGS__)

/************************************************************************/
#ifndef IAS_LOG_CHANNEL
#define IAS_LOG_DEBUG(format,...) \
    if (IAS_LOG_LEVEL_DEBUG >= ias_log_message_level)          \
        ias_log_message(IAS_LOG_LEVEL_DEBUG,__FILE__,__LINE__, \
                        format,##__VA_ARGS__) 
#else
#define IAS_LOG_DEBUG(format,...) \
    if (IAS_LOG_LEVEL_DEBUG >= ias_log_message_level)          \
        ias_log_message_with_channel(IAS_LOG_LEVEL_DEBUG, IAS_LOG_CHANNEL, \
            __FILE__,__LINE__, format,##__VA_ARGS__) 
#endif

/************************************************************************/
#define IAS_LOG_DEBUG_TO_CHANNEL(channel,format,...) \
    if (IAS_LOG_LEVEL_DEBUG >= ias_log_message_level) \
        ias_log_message_with_channel(IAS_LOG_LEVEL_DEBUG, channel, \
            __FILE__, __LINE__, format, ##__VA_ARGS__)

/************************************************************************/

#undef PRINT_FORMAT_ATTRIBUTE
#undef PRINT_FORMAT_ATTRIBUTE_WC

#endif

