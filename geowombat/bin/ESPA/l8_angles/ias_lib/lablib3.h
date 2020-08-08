/**************************************************************************/
/*                          lablib3.h                                     */
/*                                                                        */
/*  Version:                                                              */
/*                                                                        */
/*      1.0    March 31, 1994                                             */
/*                                                                        */
/*  Change History:                                                       */
/*                                                                        */
/*      03-31-94    Original code                                         */
/*	01-09-95    jsh - Changed OBJECT to OBJDESC                       */
/*      03-11-97    Add ODL_GROUP & ODL_OBJECT,                           */
/*                  change  OdlValidEndObjDesc prototype                  */
/*      04-09-97    Added Landsat 7 Date and Time                         */
/*                  change OdlDataType prototype			  */ 
/*                                                                        */
/**************************************************************************/

#ifndef __LABLIB3_LOADED
#define __LABLIB3_LOADED

#include "toolbox.h"

/**************************************************************************/
/*                         Symbol definitions                             */
/**************************************************************************/

/*  These symbols are used by the label library to determine what, if any,
    type of expanding should be performed on a label.  They are used as masks
    and MUST be in powers of two.
*/

#define ODL_NOEXPAND         0
#define ODL_EXPAND_STRUCTURE 1
#define ODL_EXPAND_CATALOG   2

/*  These symbols are used to restrict the scope of an object search  */

#define ODL_RECURSIVE_DOWN   0
#define ODL_TO_END           1
#define ODL_CHILDREN_ONLY    2
#define ODL_SIBLINGS_ONLY    3
#define ODL_THIS_OBJECT      4

/*  These symbols represent the different data location types  */

#define ODL_RECORD_LOCATION 0
#define ODL_BYTE_LOCATION   1

/*  These symbols represent the data type of a keyword's value  */

#define ODL_UNKNOWN              0
#define ODL_INTEGER              1
#define ODL_REAL                 2
#define ODL_SYMBOL               3
#define ODL_TEXT                 4
#define ODL_DATE                 5
#define ODL_DATE_TIME            6
#define ODL_SEQUENCE             7
#define ODL_SET                  8
#define ODL_L7_DATE_TIME         9
#define ODL_L7_DATE_TIME_FRAC   10

/* Valid is_a_group settings */
#define ODL_OBJECT    0             /* class is OBJECT */
#define ODL_GROUP     1             /* class is GROUP  */

/**************************************************************************/
/*                              Typedefs                                  */
/**************************************************************************/

typedef struct Object_Structure
{
    char *class;
    char *pre_comment;  /* Comments before the OBJECT = line     */
    char *line_comment; /* Comments on the OBJECT = line         */
    char *post_comment; /* Comments before the END_OBJECT = line */
    char *end_comment;  /* Comments on the OBJECT = line         */
    char *file_name;
    char *appl1;   /*  free for your application to use as you see fit  */
    char *appl2;   /*  free for your application to use as you see fit  */
    unsigned short is_a_group;
    unsigned long level;
    unsigned long line_number;
    unsigned long child_count;
    struct Object_Structure *parent;
    struct Object_Structure *left_sibling;
    struct Object_Structure *right_sibling;
    struct Object_Structure *first_child;
    struct Object_Structure *last_child;
    struct Keyword_Structure *first_keyword;
    struct Keyword_Structure *last_keyword;

} OBJDESC;

typedef struct Keyword_Structure
{
    char *name;
    char *file_name;
    char *value;
    unsigned long size;
    char *pre_comment;   /* Comments before the KEYWORD = line */
    char *line_comment;  /* Comments on the KEYWORD = line     */
    char *appl1;   /*  free for your application to use as you see fit  */
    char *appl2;   /*  free for your application to use as you see fit  */
    unsigned short is_a_pointer;
    unsigned short is_a_list;
    unsigned long line_number;
    struct Object_Structure *parent;
    struct Keyword_Structure *left_sibling;
    struct Keyword_Structure *right_sibling;

} KEYWORD;

/**************************************************************************/
/*                         Function Prototypes                            */
/**************************************************************************/

#ifdef _NO_PROTO

OBJDESC *OdlParseLabelFile();
OBJDESC *OdlParseLabelString();
OBJDESC *OdlExpandLabelFile();
unsigned short ExpandIsRecursive();
OBJDESC *OdlFindObjDesc();
OBJDESC *OdlNextObjDesc();
OBJDESC *OdlCutObjDesc();
OBJDESC *OdlPasteObjDesc();
OBJDESC *OdlPasteObjDescBefore();
OBJDESC *OdlPasteObjDescAfter();
OBJDESC *OdlCopyObjDesc();
OBJDESC *OdlNewObjDesc();
char *OdlGetLabelVersion();
char *OdlGetObjDescClassName();
int OdlGetObjDescChildCount();
int OdlGetObjDescLevel();
OBJDESC *OdlGetObjDescParent();
void OdlAdjustObjDescLevel();
KEYWORD *OdlFindKwd();
KEYWORD *OdlNextKwd ();
KEYWORD *OdlCutKwd();
KEYWORD *OdlPasteKwd();
KEYWORD *OdlPasteKwdBefore();
KEYWORD *OdlPasteKwdAfter();
KEYWORD *OdlCopyKwd();
KEYWORD *OdlNewKwd();
KEYWORD *OdlGetFirstKwd ();
KEYWORD *OdlGetNextKwd ();
char *OdlGetKwdValue();
unsigned short OdlGetKwdValueType();
char *OdlGetKwdUnit();
char *OdlGetKwdName();
char *OdlGetFileName();
char *OdlGetFileSpec();
OBJDESC *OdlFreeTree();
KEYWORD *OdlFreeAllKwds();
KEYWORD *OdlFreeKwd();
FILE *OdlLocateStart();
FILE *OdlOpenMessageFile();
short OdlPrintMessage();
char *OdlFormatMessage();
void OdlPrintHierarchy();
void OdlPrintLabel();
void OdlPrintKeywords();
OBJDESC *OdlParseFile();
short OdlNestingLevel();
short OdlValidBraces();
short OdlValidElement();
short OdlValidEndObjDesc();
short OdlValidIdentifier();
short OdlValidKwd();
short OdlValidObjDesc();
short OdlValidValueList();
OBJDESC *OdlTraverseTree();
char *OdlFirstWord();
char *OdlNextWord();
char *OdlValueStart();
char *OdlValueEnd();
char *OdlValueRowStart();
char *OdlValueRowEnd();
unsigned short OdlDataType();
char *OdlTypeString();
TB_STRING_LIST *OdlGetAllKwdValues();
char *OdlTempFname();
unsigned short OdlWildCardCompare();
short CheckBalance();

#else

OBJDESC *OdlParseLabelFile (char *, char *, MASK, int);
OBJDESC *OdlParseLabelString (char *, char *, MASK, int);
OBJDESC *OdlExpandLabelFile (OBJDESC *, char *, MASK, int);
unsigned short ExpandIsRecursive (KEYWORD *, char *);
OBJDESC *OdlFindObjDesc(OBJDESC *, const char *, const char *, char *, 
                       unsigned long, unsigned short);
OBJDESC *OdlNextObjDesc (OBJDESC *, unsigned long, unsigned short *);
OBJDESC *OdlTraverseTree (OBJDESC *, unsigned long);
OBJDESC *OdlCutObjDesc (OBJDESC *);
OBJDESC *OdlPasteObjDesc (OBJDESC *, OBJDESC *);
OBJDESC *OdlPasteObjDescBefore (OBJDESC *, OBJDESC *);
OBJDESC *OdlPasteObjDescAfter (OBJDESC *, OBJDESC *);
OBJDESC *OdlCopyObjDesc (OBJDESC *);
OBJDESC *OdlNewObjDesc (const char *,const char *,const char *,const char *,
        const char *, const char *, short, long);
char *OdlGetLabelVersion (OBJDESC *);
char *OdlGetObjDescClassName (OBJDESC *);
int OdlGetObjDescChildCount (OBJDESC *);
int OdlGetObjDescLevel (OBJDESC *);
OBJDESC *OdlGetObjDescParent (OBJDESC *);
void OdlAdjustObjDescLevel (OBJDESC *);
KEYWORD *OdlFindKwd (OBJDESC *, const char *, char *, unsigned long, unsigned short);
KEYWORD *OdlNextKwd (KEYWORD *, char *, char *, unsigned long, unsigned short);
KEYWORD *OdlCutKwd (KEYWORD *);
KEYWORD *OdlPasteKwd (KEYWORD *, OBJDESC *);
KEYWORD *OdlPasteKwdBefore (KEYWORD *, KEYWORD *);
KEYWORD *OdlPasteKwdAfter (KEYWORD *, KEYWORD *);
KEYWORD *OdlCopyKwd (KEYWORD *);
KEYWORD *OdlNewKwd (char *, char *, char *, char *, char *, long);
KEYWORD *OdlGetFirstKwd (OBJDESC *);
KEYWORD *OdlGetNextKwd (KEYWORD *);
char *OdlGetKwdValue (KEYWORD *);
unsigned short OdlGetKwdValueType (KEYWORD *);
char *OdlGetKwdUnit (KEYWORD *);
char *OdlGetKwdName (KEYWORD *);
char *OdlGetFileName (KEYWORD *, unsigned long *, unsigned short *);
char *OdlGetFileSpec (char *);
OBJDESC *OdlFreeTree (OBJDESC *);
KEYWORD *OdlFreeAllKwds (OBJDESC *);
KEYWORD *OdlFreeKwd (KEYWORD *);
FILE *OdlOpenMessageFile (const char *, FILE *, int);
FILE *OdlLocateStart(char *, unsigned long, unsigned short);
short OdlPrintMessage (const char *, FILE *, long, const char *, int);
char *OdlFormatMessage (char *);
void OdlPrintHierarchy (OBJDESC *, char *, FILE *, int);
void OdlPrintLabel (OBJDESC *, char *, FILE *, unsigned long);
void OdlPrintKeywords (OBJDESC *, char *, FILE *, int);
OBJDESC *OdlParseFile (char *, FILE *, char *, FILE *, 
                      int, unsigned short, unsigned short, unsigned short);
short OdlNestingLevel (char *, long *, long *);
short OdlValidBraces (char *, long, long, char *, FILE *, long, int);
short OdlValidElement (char *, char *, FILE *, long, long, int);
short OdlValidEndObjDesc (OBJDESC *, char *, char *, char *, FILE *,long,unsigned short, int);
short OdlValidIdentifier (const char *, const char *, const char *, FILE *, 
                          long, int);
short OdlValidKwd (OBJDESC *, char *, char *, char *, char *,FILE *,long, int);
short OdlValidObjDesc (OBJDESC *, char *, char *, char *, FILE *,long, int);
short OdlValidValueList (char *, char *, FILE *,long, int);
char *OdlFirstWord (char *);
char *OdlNextWord (char *);
char *OdlValueStart (char *);
char *OdlValueEnd (char *);
char *OdlValueRowStart (char *);
char *OdlValueRowEnd (char *);
unsigned short OdlDataType(char *);
char *OdlTypeString (unsigned short, char *);
TB_STRING_LIST *OdlGetAllKwdValues(KEYWORD *);
char *OdlTempFname();
unsigned short OdlWildCardCompare(const char *, const char *);
short CheckBalance(char *);

#endif  /* _NO_PROTO  */
                              
/**************************************************************************/
/*                       End of lablib3.h stuff                          */
/**************************************************************************/

#endif  /*  __LABLIB3_LOADED  */

