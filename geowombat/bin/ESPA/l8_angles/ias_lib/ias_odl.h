#ifndef IAS_ODL_H
#define IAS_ODL_H

#include <stdio.h>
#include "ias_types.h"
#include "ias_const.h"

/* Warning status codes returned from ias_odl_get_field() */
#define IAS_ODL_NOT_ENOUGH_MEMORY_SUPPLIED 2 
#define IAS_ODL_NOT_FOUND 3
#define IAS_ODL_INVALID_DATA_TYPE 4 

/* Forward reference a couple ODL structures to hide their definition from
   the rest of the software */
typedef struct Object_Structure IAS_OBJ_DESC;
typedef struct Keyword_Structure IAS_ODL_KEYWORD;

/* Flag indicating what function should be performed on the odl field,
   add or replace it. */
typedef enum
{
    IAS_ODL_ADD,
    IAS_ODL_REPLACE
} IAS_ODL_FIELD_FUNCTION;

/* Flag passed to ias_odl_get_field() indicating the ODL keyword's expected 
   data type */
typedef enum  
{
    IAS_ODL_Double,
    IAS_ODL_ArrayOfString,
    IAS_ODL_String,
    IAS_ODL_Int,
    IAS_ODL_Long,
    IAS_ODL_Float,
    IAS_ODL_Sci_Not
} IAS_ODL_TYPE;

/* Type definition for a list of ODL parameters to read */
typedef struct
{
    const char *group_name;     /* group name to locate*/
    char *attribute;            /* attribute name */
    void *parm_ptr;             /* Location for attribute value */
    int parm_size;              /* Number of bytes at param_ptr */
    IAS_ODL_TYPE parm_type;     /* Type of parameter */
    int parm_count;             /* Number attribute values */
} ODL_LIST_TYPE;


/***********************************************************************
  Prototype functions for ODL utility library
***********************************************************************/

IAS_OBJ_DESC *ias_odl_create_tree
(
);

void ias_odl_write_tree
(
    IAS_OBJ_DESC *ODLTree,          /* I: ODL Object Tree to save */
    char *p_ODLFile      /* I: ODL file name (full directory path) */
);

IAS_OBJ_DESC *ias_odl_add_group
(
    IAS_OBJ_DESC *ODLTree,           /* I/O: ODL Object Tree to populate */
    const char *p_ClassName     /* I: Group/Object name */
);

int ias_odl_add_field
(
    IAS_OBJ_DESC *ODLTree,           /* I/O: ODL Object Tree to populate */
    const char *p_LabelName,    /* I: Field to add */
    IAS_ODL_TYPE ValueType,     /* I: What type the field is */
    const int p_MemorySize,     /* I: Total memory size of attribute values */
    void *p_MemoryAddr,         /* I: Pointer to the attribute information */
    const int nelements         /* I: Number of attribute values */
);

int ias_odl_replace_field
(
    IAS_OBJ_DESC *ODLTree,           /* I/O: ODL Object Tree to populate */
    const char *p_LabelName,    /* I: Field to add */
    IAS_ODL_TYPE ValueType,     /* I: What type the field is */
    const int p_MemorySize,     /* I: Total memory size of attribute values */
    void *p_MemoryAddr,         /* I: Pointer to the attribute information */
    const int nelements         /* I: Number of attribute values */
);

int ias_odl_add_field_list
(
    IAS_OBJ_DESC *p_ODLTree,         /* I/O: ODL Object Tree to populate */
    ODL_LIST_TYPE *p_ListParms, /* I: List of attibutes to add */
    const int Count             /* I: Number of items in the ODL_LIST_TYPE */
);

IAS_OBJ_DESC *ias_odl_read_tree 
(
    const char *p_ODLFile       /* I: ODL file name */
);

void ias_odl_free_tree
(
    IAS_OBJ_DESC *p_lp               /* I: ODL tree */
);

int ias_odl_get_field
(
    void *p_MemoryAddr,         /* I: Pointer to the attribute information */
    int MemorySize,             /* I: Total memory size of attribute values */
    IAS_ODL_TYPE ValueType,     /* I: What type the field is */
    IAS_OBJ_DESC *p_ODLTree,         /* I: ODL tree */
    const char *p_ClassName,    /* I: Group/Object name */
    const char *p_LabelName,    /* I: Field to get */
    int *p_Count                /* O: Count the number of values in a array */
);

int ias_odl_get_field_list
(
    IAS_OBJ_DESC *p_ODLTree,         /* I: ODL Object Tree to parse */
    ODL_LIST_TYPE *p_ListParms, /* I/O: List of attibutes to retrieve */
    const int Count             /* I: Number of items in the ODL_LIST_TYPE */
);

IAS_OBJ_DESC *ias_odl_get_group
(
    IAS_OBJ_DESC *ODLTree,      /* I: ODL Object Tree root */
    const char *p_ClassName     /* I: Group/Object name to find */
);

int ias_odl_get_group_names
(
    IAS_OBJ_DESC *p_ODLTree,    /* I: ODL Object Tree to parse */
    char ***p_ClassNames,       /* O: ODL Group/Object names */
    int *p_Count                /* O: Number of group names returned */
);

IAS_OBJ_DESC *ias_odl_parse_file
(
    char *label_fname,                 /* I: File name to read */
    FILE *label_fptr                   /* I: File pointer to read */
);

IAS_OBJ_DESC *ias_odl_parse_label_string
(
    char *odl_string             /* I: ODL string to parse */
);

IAS_OBJ_DESC *ias_odl_find_object_description
(
    IAS_OBJ_DESC *start_object,     /* I: ODL tree to parse */
    const char *object_class,       /* I: class name to search for */
    const char *keyword_name,       /* I: keyword to search for */
    char *keyword_value,            /* I: keyword value to search for */
    unsigned long object_position   /* I: object position to search for */
);

IAS_ODL_KEYWORD *ias_odl_find_keyword
(
    IAS_OBJ_DESC *start_object,     /* I: ODL tree to parse */
    const char *keyword_name,       /* I: keyword to search for */
    char *keyword_value,            /* I: keyword value to search for */
    unsigned long keyword_position  /* I: object position to search for */
);

#endif
