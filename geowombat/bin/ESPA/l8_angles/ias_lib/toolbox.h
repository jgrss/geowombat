
/**************************************************************************/
/*                          toolbox.h                                     */
/*                                                                        */
/*  Version:                                                              */
/*                                                                        */
/*      1.0    March 31, 1994                                             */
/*                                                                        */
/*  Change History:                                                       */
/*                                                                        */
/*      03-31-94    Original code                                         */
/*	01-09-95    jsh - included <stdlib.h>                             */
/*                                                                        */
/**************************************************************************/

#ifndef __TOOLBOX_LOADED
#define __TOOLBOX_LOADED

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

/**************************************************************************/
/*                          Symbol Definitions                            */
/**************************************************************************/

#ifndef NULL
#define NULL 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define TB_MAX_BUFFER  32767
#define TB_MAXLINE     2048   /* was 256 */
#define TB_MAXFNAME    80
#define TB_MAXPATH     256
#define TB_MAXTIME     40
#define TB_MAXNUM      20

/**************************************************************************/
/*                           Macro Definitions                            */
/**************************************************************************/


#ifdef VAX
#define VmsError(zstatz) (!((zstatz)&1) && zstatz != SS$_ENDOFFILE && zstatz != SS$_ENDOFTAPE)
#endif

#define RingThatBell(znumz) {int ziz; for (ziz=0; ziz < znumz; ++ziz) (void)printf("%c", (char) 7);}
#define CloseMe(zfptrz) {if (zfptrz != NULL) {(void)fclose(zfptrz); zfptrz = NULL;}}
#define LemmeGo(zmptrz) {if (zmptrz != NULL) {free(zmptrz); zmptrz = NULL;}}
#define SayGoodbye() {(void)printf("I'm out of memory and I can't get up!\n"); exit(1);}
#define LastChar(zstrz) (zstrz+strlen(zstrz)-1)

#define NewString(zstrz, zsizez) \
            {if (zsizez <= 1) \
                 zstrz = (char *)malloc(1); \
             else \
                 zstrz = (char *)malloc(zsizez); \
             if (zstrz == NULL) \
                 SayGoodbye() \
             else \
                 *zstrz = '\0';}

#define CopyString(zstrz, zoldz) \
            {if (zoldz == NULL) \
                 zstrz = NULL; \
             else \
             { \
                 NewString(zstrz, ((long)(1+strlen(zoldz))))  \
                 (void)strcpy(zstrz,zoldz); \
             }}

#define AppendString(zstrz, znewz) \
            {if (znewz != NULL) \
             { \
                if (zstrz == NULL) \
                    NewString(zstrz, (long)(1+strlen(znewz))) \
                else \
                    zstrz = (char *)realloc(zstrz, 1+strlen(znewz)+strlen(zstrz)); \
                if (zstrz == NULL) \
                    SayGoodbye() \
                else \
                    (void)strcat(zstrz, znewz); \
             }}

#define StripLeading(zstrz, zstripz) \
            {char *zcz; int len;\
             for (zcz=zstrz; ((*zcz != '\0') && (*zcz == zstripz)); ++zcz) ; \
             len = strlen(zcz) + 1; \
             (void)memmove(zstrz, zcz, len);}

#define StripTrailing(zstrz, zstripz) \
            {char *zcz; \
             for (zcz=LastChar(zstrz); ((zcz >= zstrz) && (*zcz == zstripz)); --zcz) \
                 *zcz = '\0';}

#define StripUnprintables(zstrz) \
            {char *zcz; \
             for (zcz=LastChar(zstrz); ((zcz >= zstrz) && ((*zcz < ' ') || (*zcz > '~'))); --zcz) \
                 *zcz = '\0';}

#define ReplaceChar(zstrz, zoldz, znewz) \
            {char *zcz; \
             for (zcz=zstrz; *zcz != '\0'; ++zcz)  \
                 {if (*zcz == zoldz) *zcz = znewz;}}

#define NotOneOfThese(zstrz, zsetz) (zstrz + strspn(zstrz, zsetz))

#define UpperCase(zstrz) \
            {char *zcz; \
             for (zcz=zstrz; *zcz != '\0'; ++zcz) {*zcz=(char)toupper(*zcz);}}

#define LowerCase(zstrz) \
            {char *zcz; \
             for (zcz=zstrz; *zcz != '\0'; ++zcz) {*zcz=(char)tolower(*zcz);}}

#define DateTime(zascz) \
            { \
                struct tm *zptrz = {NULL}; \
                time_t zltz; \
                zltz = time(NULL); \
                zptrz = localtime(&zltz); \
                zascz = asctime(zptrz); \
                zascz[strlen(zascz) -1] = '\0'; \
            }

/*
#define AtTheToneTheTimeWillBe(ztimez) \
            { \
                char *zascz = {NULL}; \
                DateTime(zascz) \
                (void)strcpy(ztimez, zascz); \
                LemmeGo(zascz) \
            }
*/

/**************************************************************************/
/*                              Typedefs                                  */
/**************************************************************************/

typedef unsigned short MASK;

typedef struct tb_string_list
{
    char *text;
    struct tb_string_list *next;

} TB_STRING_LIST;

#define RemoveStringList(zlistz) \
        { \
            TB_STRING_LIST *znz, *znnz=NULL; \
            for (znz=zlistz; znz != NULL; znz=znnz) \
            { \
                znnz = znz->next; \
                LemmeGo(znz->text) \
                LemmeGo(znz) \
            } \
        }

#define NewStringList(zstrz, zlistz) \
        { \
            if (zlistz != NULL) RemoveStringList(zlistz) \
            zlistz = (TB_STRING_LIST *) malloc(sizeof(TB_STRING_LIST)); \
            zlistz->next = NULL; \
            if (zlistz == NULL) SayGoodbye() \
            if (zstrz == NULL) \
                NewString(zlistz->text, 1) \
            else \
                CopyString(zlistz->text, zstrz) \
        }                                               

#define AddStringToList(zstrz, zlistz) \
        { \
            TB_STRING_LIST *znz, *znnz=NULL; \
            if (zlistz == NULL) \
                NewStringList(zstrz, zlistz) \
            else \
            { \
                for (znz=zlistz; znz->next != NULL; znz=znz->next) ; \
                NewStringList(zstrz, znnz) \
                znz->next = znnz; \
            } \
        }

#define AddListToList(zfrom_listz, zto_listz) \
        { \
            TB_STRING_LIST *znz; \
            if (zto_listz == NULL) \
                zto_listz = zfrom_listz; \
            else \
            if (zfrom_listz != NULL) \
            { \
                for (znz=zto_listz; znz->next != NULL; znz=znz->next) ; \
                znz->next = zfrom_listz; \
            } \
        }

/**************************************************************************/
/*                        End of toolbox.h stuff                          */
/**************************************************************************/

#endif  /*  __TOOLBOX_LOADED  */

