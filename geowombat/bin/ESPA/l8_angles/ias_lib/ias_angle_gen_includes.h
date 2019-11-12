/*******************************************************************************
 Name: ias_angle_gen_includes.h
 
 Purpose:  This header is used as a 'dummy' header. The applications that use 
           this library and are part of IAS or LPGS will use this header. Any
           applications that will be distributed to the end user will use a 
           fake header containing the defines needed to replace the header 
           files below.

 ******************************************************************************/
#ifndef _IAS_ANGLE_GEN_INCLUDES_H_
#define _IAS_ANGLE_GEN_INCLUDES_H_

/* COTS Library Includes */
/* Projection codes
   0 = Geographic
   1 = Universal Transverse Mercator (UTM)
   2 = State Plane Coordinates
   3 = Albers Conical Equal Area
   4 = Lambert Conformal Conic
   5 = Mercator
   6 = Polar Stereographic
   7 = Polyconic
   8 = Equidistant Conic
   9 = Transverse Mercator
  10 = Stereographic
  11 = Lambert Azimuthal Equal Area
  12 = Azimuthal Equidistant
  13 = Gnomonic
  14 = Orthographic
  15 = General Vertical Near-Side Perspective
  16 = Sinusiodal
  17 = Equirectangular
  18 = Miller Cylindrical
  19 = Van der Grinten
  20 = (Hotine) Oblique Mercator 
  21 = Robinson
  22 = Space Oblique Mercator (SOM)
  23 = Alaska Conformal
  24 = Interrupted Goode Homolosine 
  25 = Mollweide
  26 = Interrupted Mollweide
  27 = Hammer
  28 = Wagner IV
  29 = Wagner VII
  30 = Oblated Equal Area
  31 = Integerized Sinusiodal
  99 = User defined
*/

/* Define projection codes */
#define GEO 0
#define UTM 1
#define SPCS 2
#define ALBERS 3
#define LAMCC 4
#define MERCAT 5
#define PS 6
#define POLYC 7
#define EQUIDC 8
#define TM 9
#define STEREO 10
#define LAMAZ 11
#define AZMEQD 12
#define GNOMON 13
#define ORTHO 14
#define GVNSP 15
#define SNSOID 16
#define EQRECT 17
#define MILLER 18
#define VGRINT 19
#define HOM 20
#define ROBIN 21
#define SOM 22
#define ALASKA 23
#define GOOD 24
#define MOLL 25
#define IMOLL 26
#define HAMMER 27
#define WAGIV 28
#define WAGVII 29
#define OBEQA 30
#define ISIN 31
#define USDEF 99 

#define MAXPROJ 31		/* largest supported projection number */

/* Define a zone number for all projections except State Plane and UTM. */
#define NULLZONE 62

/* code for a point outside the image   */
#define OUTSIDE 99

#define WGS84_SPHEROID 12

/* Define shape mask value */
#define IAS_GEO_SHAPE_MASK_VALID 0x1

/* Define the number of projection parameters */
#define GCTP_PROJECTION_PARAMETER_COUNT 15

/* Define the return values for error and success */
#define GCTP_ERROR -1
#define GCTP_SUCCESS 0

/* Define the return value for a coordinate that is in a "break" area of an
   interrupted projection like Goode's */
#define GCTP_IN_BREAK -2

/* Define the unit codes */
#define RADIAN  0
#define FEET    1
#define METER   2
#define SECOND  3
#define DEGREE  4
#define DMS     5

#endif
