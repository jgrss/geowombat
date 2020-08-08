#ifndef IAS_PARM_PRIVATE_H
#define IAS_PARM_PRIVATE_H

#include "ias_odl.h"

/* function prototypes */
int ias_parm_check_ranges
(
    IAS_PARM_TYPE_UNION value,      /* I: pointer to table of values */
    IAS_PARM_TYPE_UNION_CONST valid_values,
                                    /* I: pointer to table of valid values */
    int count_read,                 /* I: count of values read */
    int valid_count,                /* I: count of valid values */
    IAS_PARM_TYPE parm_type,        /* I: parameter type */
    int array_flag                  /* I: IAS_PARM_ARRAY / IAS_PARM_NOT_ARRAY  
                                          flag for array, necessary for 
                                          reading the string */
);

int ias_parm_map_odl_type
(
    IAS_PARM_PARAMETER_DEFINITION *pd, /* I: pointer to current parameter
                                             definition */
    IAS_ODL_TYPE *type                 /* O: ODL type to map to */
);

#endif
