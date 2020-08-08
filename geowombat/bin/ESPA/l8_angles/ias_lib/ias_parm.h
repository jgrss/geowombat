/* How to build a parameter table using the Parameter File I/O Library

start parameter table definition specifying the number of parameters

   IAS_PARM_DECLARE_TABLE(parms, 2);

add a string parameter to the table, in this example, the work order directory

    IAS_PARM_ADD_STRING(parms, WO_DIRECTORY, "Work order directory",
        IAS_PARM_REQUIRED, 0, NULL, 1, NULL,
        work_order_dir, sizeof(work_order_dir), 0);

add a CC resampling alpha parameter, with a range of -1.0 to 0.0
and a default value of 0.0

    double default_pccalpha[] = {0.0};
    IAS_PARM_ADD_DOUBLE(parms, PCCALPHA, "Cubic convolution alpha parameter",
        IAS_PARM_OPTIONAL, IAS_PARM_NOT_ARRAY,
        1, -1.0, 0.0,
        1, default_pccalpha,
        &pccalpha, sizeof(pccalpha), 0);

calculate number of items in the ODL_param_list

    count = IAS_PARM_GET_TABLE_SIZE(parms);

call the routine to provide help for the parameters if requested 
       also verifies the file exists

    status = (ias_parm_provide_help(odl_file_name, parms, count, 
                                    IAS_INPUT_PARAMETERS));
    if (status)
    {
        if (status == 1)  help was successful
            exit(EXIT_SUCCESS);
        else
        {
            IAS_LOG_ERROR("Providing parameter file help failed");
            return ERROR;
        }
    }

read the ODL file to fill in the parameter table

    status = ias_parm_read(odl_file_name, parms, count);

update the ODL file using the parameters in the parameter table

    status = ias_parm_update(odl_file_name, parms, count);


An additional note about adding parameters, when adding one that is an array 
(IAS_PARM_ARRAY), you do not need the address operator in front of the 
variable. Notice the difference in code taken from the aliresample 
get_resample_parameters routine:

    double default_min_max_output[] = {0.0, 255.0};
    IAS_PARM_ADD_DOUBLE(parms, MINMAX_OUTPUT_DN, "Minimum and maximum output "
        "values from the resampler",
        IAS_PARM_OPTIONAL, IAS_PARM_ARRAY,  <----
        1, 0.0, 16384.0,
        1, default_min_max_output,
 ---->  min_max_output, sizeof(min_max_output), 2);

    double default_alpha[] = {-0.5};
    IAS_PARM_ADD_DOUBLE(parms, PCCALPHA, "Cubic convolution alpha parameter",
        IAS_PARM_OPTIONAL, IAS_PARM_NOT_ARRAY,  <----
        1, -1.0, 0.0,
        1, default_alpha,
 ---->  &alpha, sizeof(alpha), 0);


*/

#ifndef IAS_PARM_H
#define IAS_PARM_H

#include "ias_logging.h"    /* for the IAS_LOG_ERROR definition */

/* Type of template or help output */
typedef enum
{
    IAS_INPUT_PARAMETERS,
    IAS_OMF_PARAMETERS,
} IAS_PARAMETER_SOURCE;

/* Type of parameter to be read or written */
typedef enum
{
    IAS_PARM_BOOLEAN,
    IAS_PARM_INT,
    IAS_PARM_DOUBLE,
    IAS_PARM_STRING,
} IAS_PARM_TYPE;

/* Union for the parameter values */
typedef union
{
    int *type_int;
    double *type_double;
    char *type_string;
    char **type_string_array;
    char *type_void;
} IAS_PARM_TYPE_UNION;

/* Union for the valid or default parameter values */
typedef union
{
    const int *type_int;
    const double *type_double;
    const char *type_string;
    const char **type_string_array;
    const char *type_void;
} IAS_PARM_TYPE_UNION_CONST;

/* Note:  The number_valids member of ias_parm_parameter_definition 
   represents the number of values the user has entered as valid values.  
   If the parameter is a string, this number represents the number 
   of strings that are legal values.  If it is a number (int or double), 
   The valids are one or more min/max range.  The total number 
   should be used for number_valids, not the number of range pairs. 
   If number_valids is 1, a single pair will be expected. 
   No other odd number will be accepted */

/* This structure contains all (possible) information on a parameter */
typedef struct ias_parm_parameter_definition
{
    /* these fields define information about the parameter */
    const char *name;          /* parameter name */
    const char *description;   /* parameter description */
    int is_required;           /* non-zero if required, zero if optional */
    IAS_PARM_TYPE type;        /* parameter type */
    int is_an_array;           /* is the parameter an array (1) or not (0) */
    int number_valids;         /* number of values entered as constraints */
    IAS_PARM_TYPE_UNION_CONST valid_values; /* possible valid values */
    int number_defaults;       /* number of default values entered */
    IAS_PARM_TYPE_UNION_CONST default_values; /* default parameter values --
                                  used if parameter is optional and not read */
    /* these fields define the information needed to return the value
       assigned to the parameter */
    IAS_PARM_TYPE_UNION value; /* value of the parameter read */
    int value_bytes;           /* bytes pointed to by value */
    int min_count;             /* minimum count of items to read, 0=no minimum*/
    int count_read;            /* number of values read if the parameter is an
                                  array */
} IAS_PARM_PARAMETER_DEFINITION;

/* Defines if a parameter is optional or required */
#define IAS_PARM_OPTIONAL 0
#define IAS_PARM_REQUIRED 1

/* Defines if a parameter is an array or not */
#define IAS_PARM_NOT_ARRAY 0
#define IAS_PARM_ARRAY 1

#define IAS_PARM_MAX_DB_NAMELENGTH 40  /* DB tables define parameter name as
                                          a 40-character string */

/* Various macros used internal to the parameter macro definitions */
#define SET_INT_LIST(x) {.type_int = x}
#define SET_DOUBLE_LIST(x) {.type_double = x}
#define SET_STRING(x) {.type_string = x}
#define SET_STRING_LIST(x) {.type_string_array = x}

/* Sets up the parameter table; needs table name and size */
#define IAS_PARM_DECLARE_TABLE(table, size) \
    struct ias_parm_parameter_definition *table[size]; \
    int size_##table = size; \
    int count_##table = 0

/* Gets the parameter table size */
#define IAS_PARM_GET_TABLE_SIZE(table) (count_##table)

/* Provide access to the count of parameters in the table */
#define IAS_PARM_GET_TABLE_COUNT(table) count_##table

/* Allow the previously added parameter to be "popped" off the end of the 
   parameter table.  This is to facilitate conditionally writing parameters
   to the OMF file. */
#define IAS_PARM_POP_LAST_PARAMETER(table) count_##table--

/* Gets the count of values to read */
#define IAS_PARM_GET_COUNT_READ(table, parm_name) \
        parm_name##_##table.count_read

/* Sets the count of values to write */
#define IAS_PARM_SET_COUNT_WRITE(table, parm_name, count) \
        parm_name##_##table.count_read = count

/* function prototypes */
int ias_parm_read
(
    const char *odl_file_name,   /* I: ODL file name to read */
    struct ias_parm_parameter_definition **odl_list_ptr,
                                 /* I/O : pointer to list of items to read
                                    from the ODL file */
    int list_length              /* I: number of items in the list */
);

int ias_parm_update
(
    const char *odl_file_name,   /* I: ODL file name to update */
    struct ias_parm_parameter_definition **odl_list_ptr,
                                 /* I/O : pointer to list of items to write
                                    from the ODL file */
    int list_length              /* I: number of items in the list */
);

int ias_parm_provide_help
(
    const char *option,          /* I: parameter to show help or template
                                    options available are:
                                    --help, --template, and --loadtable */
    struct ias_parm_parameter_definition **params,
                                 /* I/O : pointer to list of items to read
                                    from the ODL file */
    int count,                   /* I: number of items */
    IAS_PARAMETER_SOURCE file_source_type /* I: Type of file source, OMF and 
                                    Parameter File */
);

/* parameter definition to add a boolean */
#define IAS_PARM_ADD_BOOLEAN(table, parm_name, description, is_required, \
            is_an_array, default_ptr, return_ptr, return_bytes, min_count) \
            int range_##parm_name##_##table[] = {0, 1}; \
            struct ias_parm_parameter_definition parm_name##_##table = \
            {#parm_name, description, is_required, IAS_PARM_BOOLEAN, \
            is_an_array, 2, SET_INT_LIST(range_##parm_name##_##table), \
            sizeof(*default_ptr)/sizeof(int), SET_INT_LIST(default_ptr), \
            SET_INT_LIST(return_ptr), return_bytes, min_count}; \
            if (count_##table < size_##table) \
                table[count_##table++] = &parm_name##_##table; \
            else IAS_LOG_ERROR("Error: %s size exceeded", #table)

/* parameter definition to add an int */
#define IAS_PARM_ADD_INT(table, parm_name, description, is_required, \
            is_an_array, has_range, low, high, \
            num_defaults, default_ptr, return_ptr, return_bytes, min_count) \
            int range_##parm_name##_##table[] = {low, high}; \
            struct ias_parm_parameter_definition parm_name##_##table = \
            {#parm_name, description, is_required, IAS_PARM_INT, is_an_array, \
            has_range, SET_INT_LIST(range_##parm_name##_##table), \
            num_defaults,SET_INT_LIST(default_ptr), \
            SET_INT_LIST(return_ptr), return_bytes, min_count}; \
            if (count_##table < size_##table) \
                table[count_##table++] = &parm_name##_##table; \
            else IAS_LOG_ERROR("Error: %s size exceeded", #table)

/* Count, in this macro, represents the number of valid values in the range, 
   not the number of valid pairs.  The multiple ranges for numbers is used
   to set a list of valid values.  That is, if the values for an integer
   parameter should be 10, 20, and 30, the range array should be set to
   {10, 10, 20, 20, 30, 30} and the count would be 6 */
/* parameter definition to add an int with ranges */
#define IAS_PARM_ADD_INT_WITH_RANGES(table, parm_name, description, \
            is_required, is_an_array, range_count, ranges, \
            num_defaults, default_ptr, return_ptr, return_bytes, min_count) \
            struct ias_parm_parameter_definition parm_name##_##table = \
            {#parm_name, description, is_required, IAS_PARM_INT, is_an_array, \
            range_count, SET_INT_LIST(ranges), \
            num_defaults,SET_INT_LIST(default_ptr), \
            SET_INT_LIST(return_ptr), return_bytes, min_count}; \
            if (count_##table < size_##table) \
                table[count_##table++] = &parm_name##_##table; \
            else IAS_LOG_ERROR("Error: %s size exceeded", #table)

/* parameter definition to add a double */
#define IAS_PARM_ADD_DOUBLE(table, parm_name, description, is_required, \
            is_an_array, has_range, low, high, \
            num_defaults, default_ptr, return_ptr, return_bytes, min_count) \
            double range_##parm_name##_##table[] = {low, high}; \
            struct ias_parm_parameter_definition parm_name##_##table = \
            {#parm_name, description, is_required, IAS_PARM_DOUBLE, \
            is_an_array, \
            has_range, SET_DOUBLE_LIST(range_##parm_name##_##table), \
            num_defaults,SET_DOUBLE_LIST(default_ptr), \
            SET_DOUBLE_LIST(return_ptr), return_bytes, min_count}; \
            if (count_##table < size_##table) \
                table[count_##table++] = &parm_name##_##table; \
            else IAS_LOG_ERROR("Error: %s size exceeded", #table)

/* Count, in this macro, represents the number of valid values in the range, 
   not the number of valid pairs.  The multiple ranges for numbers is used
   to set a list of valid values.  That is, if the values for a double
   parameter should be .5, 1.0, and 1.5, the range array should be set to
   {.5, .5, 1.0, 1.0, 1.5, 1.5} and the count would be 6 */
/* parameter definition to add a double with ranges */
#define IAS_PARM_ADD_DOUBLE_WITH_RANGES(table, parm_name, description,  \
            is_required, is_an_array, range_count, ranges, \
            num_defaults, default_ptr, return_ptr, return_bytes, min_count) \
            struct ias_parm_parameter_definition parm_name##_##table = \
            {#parm_name, description, is_required, IAS_PARM_DOUBLE, \
            is_an_array, \
            range_count, SET_DOUBLE_LIST(ranges), \
            num_defaults,SET_DOUBLE_LIST(default_ptr), \
            SET_DOUBLE_LIST(return_ptr), return_bytes, min_count}; \
            if (count_##table < size_##table) \
                table[count_##table++] = &parm_name##_##table; \
            else IAS_LOG_ERROR("Error: %s size exceeded", #table)

/* Add double taking into account the pass through the processing flow. */
#define IAS_PARM_ADD_DOUBLE_WITH_PASS(table, parm_name, description, \
            is_required, is_an_array, has_range, low, high, \
            num_defaults, default_ptr, return_ptr, return_bytes, min_count, \
            processing_pass) \
            char temp_##table##_##parm_name[200]; \
            int range_##parm_name##_##table[] = {low, high}; \
            if (processing_pass == 0) \
                strcpy(temp_##table##_##parm_name,#parm_name); \
            else \
                sprintf(temp_##table##_##parm_name,"%s_PASS_%d",#parm_name, \
                    processing_pass); \
            struct ias_parm_parameter_definition parm_name##_##table = \
            {temp_##table##_##parm_name, description, is_required, \
            IAS_PARM_DOUBLE, is_an_array, \
            has_range, SET_INT_LIST(range_##parm_name##_##table), \
            num_defaults,SET_DOUBLE_LIST(default_ptr), \
            SET_DOUBLE_LIST(return_ptr), return_bytes, min_count}; \
            if (count_##table < size_##table) \
                table[count_##table++] = &parm_name##_##table; \
            else IAS_LOG_ERROR("Error: %s size exceeded", #table)

/* parameter definition to add a string */
#define IAS_PARM_ADD_STRING(table, parm_name, description, is_required, \
            valid_count, valid_ptr, num_defaults, default_ptr, \
            return_ptr, return_bytes, min_count) \
            IAS_PARM_ADD_STRING_WITH_PASS(table, parm_name, description, \
            is_required, valid_count, valid_ptr, num_defaults, default_ptr, \
            return_ptr, return_bytes, min_count, 0)
           
/* Add string taking into account the pass through the processing flow. */
#define IAS_PARM_ADD_STRING_WITH_PASS(table, parm_name, description, \
            is_required, valid_count, valid_ptr, num_defaults, default_ptr, \
            return_ptr, return_bytes, min_count, processing_pass) \
            char temp_##table##_##parm_name[200]; \
            if (processing_pass == 0) \
                strcpy(temp_##table##_##parm_name,#parm_name); \
            else \
                sprintf(temp_##table##_##parm_name,"%s_PASS_%d",#parm_name, \
                    processing_pass); \
            struct ias_parm_parameter_definition parm_name##_##table = \
            {temp_##table##_##parm_name, description, is_required, \
            IAS_PARM_STRING, IAS_PARM_NOT_ARRAY, valid_count, \
            SET_STRING_LIST(valid_ptr), num_defaults, \
            SET_STRING_LIST(default_ptr), \
            SET_STRING(return_ptr), return_bytes, min_count}; \
            if (count_##table < size_##table) \
                table[count_##table++] = &parm_name##_##table; \
            else IAS_LOG_ERROR("Error: %s size exceeded", #table)

/* Add int taking into account the pass through the processing flow. */
#define IAS_PARM_ADD_INT_WITH_PASS(table, parm_name, description, \
            is_required, is_an_array, has_range, low, high, \
            num_defaults, default_ptr, return_ptr, return_bytes, min_count, \
            processing_pass) \
            char temp_##table##_##parm_name[200]; \
            int range_##parm_name##_##table[] = {low, high}; \
            if (processing_pass == 0) \
                strcpy(temp_##table##_##parm_name,#parm_name); \
            else \
                sprintf(temp_##table##_##parm_name,"%s_PASS_%d",#parm_name, \
                    processing_pass); \
            struct ias_parm_parameter_definition parm_name##_##table = \
            {temp_##table##_##parm_name, description, is_required, \
            IAS_PARM_INT, is_an_array, \
            has_range, SET_INT_LIST(range_##parm_name##_##table), \
            num_defaults,SET_INT_LIST(default_ptr), \
            SET_INT_LIST(return_ptr), return_bytes, min_count}; \
            if (count_##table < size_##table) \
                table[count_##table++] = &parm_name##_##table; \
            else IAS_LOG_ERROR("Error: %s size exceeded", #table)

/* parameter definition to add a string array */
/* The IAS_PARM_ADD_STRING_ARRAY macro takes a char * array for the return_ptr. 
   The developer needs to free the memory pointed to by that array */
#define IAS_PARM_ADD_STRING_ARRAY(table, parm_name, description, is_required, \
            valid_count, valid_ptr, num_defaults, default_ptr, \
            return_ptr, return_bytes, min_count) \
            struct ias_parm_parameter_definition parm_name##_##table = \
            {#parm_name, description, is_required, IAS_PARM_STRING, \
            IAS_PARM_ARRAY, valid_count, SET_STRING_LIST(valid_ptr), \
            num_defaults,SET_STRING_LIST(default_ptr), \
            SET_STRING_LIST(return_ptr), return_bytes, min_count}; \
            if (count_##table < size_##table) \
                table[count_##table++] = &parm_name##_##table; \
            else IAS_LOG_ERROR("Error: %s size exceeded", #table)

#endif
