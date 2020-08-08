#ifndef GXX_SENSOR_H
#define GXX_SENSOR_H

#ifndef IAS_NO_SENSOR_META_SUPPORT
#include "xxx_MSCD.h"
#endif

typedef struct gxx_func_ptrs
{
    /* CPF */
    
    int (*read_cpf)
    (
        const char l0r_pathname[], /* I: name of Level 0R HDF file */
        const char *cpf_pathname   /* I: name of CPF file if the CPF is not
                                         contained in the Level 0R HDF file
                                         (has been overridden by the parameter
                                         file).  Set to NULL or an empty
                                         string to use the HDF version of the
                                         CPF. */
    );

    int (*get_bumper_array_size)(void);
    
    float *(*get_cpf_scaling_parms)
    (
        int band,        /* I: band number */
        char gain_state  /* I: detector gain state: 'L' or 'H' */
    );

    void (*free_cpf)(void);

    /* MSCD */
#ifndef IAS_NO_SENSOR_META_SUPPORT
    uchar8 *(*copy_mscd_to_buffer)
    (
        uchar8 *p_data,                      /* O: buffer for unpadded MSCD
                                                   data */
        xxx_MSCDControl_TYPE *p_MSCDControl, /* I: contains MSCD data */
        int index                            /* I: MSCD index to write to */
    );

    uchar8 *(*copy_mscd_from_buffer)
    (
        uchar8 *p_data,                      /* O: buffer for unpadded MSCD
                                                   data */
        xxx_MSCDControl_TYPE *p_MSCDControl, /* I: contains MSCD data */
        int index                            /* I: MSCD index to write to */
    );
#endif
} gxx_func_ptrs;


/* Function prototypes */

int gxx_initialize_func_ptr(void);

gxx_func_ptrs *gxx_get_funcs(void);

#endif
