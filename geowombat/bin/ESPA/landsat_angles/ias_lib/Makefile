#----------------------------------------------------------------------------
# Set up the compiler options.
#----------------------------------------------------------------------------
OPTS = -g -Wall -O2 -march=nocona -mfpmath=sse -msse2  \
       -DIAS_NO_SENSOR_META_SUPPORT -DIAS_NOVAS_NOT_AVAILABLE
CC = gcc
#-------------------------------------------------------
# Define the executable(s), the pdf(s), and the html(s).
#-------------------------------------------------------
LIBS   = liblandsatang.a

INCS = -I ./

#---------------------------------------------------------
# Define the source code object files for each executable.
#---------------------------------------------------------
SRC = \
        gxx_angle_gen_calculate_angles_rpc.c \
        gxx_angle_gen_calculate_vector.c \
        gxx_angle_gen_find_dir.c \
        gxx_angle_gen_geo_utilities.c \
        gxx_angle_gen_initialize.c \
        gxx_angle_gen_initialize_transformation.c \
        gxx_angle_gen_interpolate_ephemeris.c \
        gxx_angle_gen_read_ang.c \
        gxx_angle_gen_write_image.c \
        gxx_dmsdeg.c \
        gxx_dot.c \
        gxx_geod2cart.c \
        gxx_get_units.c \
        gxx_unit.c \
        xxx_Band.c \
        xxx_CloseODL.c \
        xxx_CloseUnmap.c \
        xxx_ConvertString.c \
        xxx_Errno.c \
        xxx_FileLock.c \
        xxx_GetDirFiles.c \
        xxx_GetODLField.c \
        xxx_GetTempName.c \
        xxx_GetTime.c \
        xxx_LogError.c \
        xxx_LogStatus.c \
        xxx_OpenMap.c \
        xxx_OpenODL.c \
        xxx_RecursiveDeletion.c \
        xxx_Sensor.c \
        xxx_strtoupper.c \
        lablib3.c

OBJ = $(SRC:.c=.o)

#------------------------------------------------------------------------------
# Targets for each executable. 
#------------------------------------------------------------------------------
all: $(LIBS) 

liblandsatang.a: $(OBJ)
	ar -r $@ $(OBJ) 

#---------------------------------------
# Target to distribute the executable(s)
#---------------------------------------
.c.o:
	$(CC) $(OPTS) $(INCS) -c $< -o $@

clean:
	/bin/rm -f *.o *.a 
