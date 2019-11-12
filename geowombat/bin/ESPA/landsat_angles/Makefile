#----------------------------------------------------------------------------
# Set up the compiler options.  
#----------------------------------------------------------------------------
OPTS = -g -Wall -O2 -march=nocona -mfpmath=sse -msse2 \
       -DIAS_NO_SENSOR_META_SUPPORT
CC = gcc
#-------------------------------------------------------
# Define the executable(s), the pdf(s), and the html(s).
#-------------------------------------------------------
LIBS = -L./ias_lib/ -llandsatang -lm

INCS =  -I./ias_lib/ -I./

#---------------------------------------------------------
# Define the source code object files for each executable.
#---------------------------------------------------------
SRC = landsat_angles.c

OBJ = $(SRC:.c=.o)

#------------------------------------------------------------------------------
# Targets for each executable. 
#------------------------------------------------------------------------------
all: landsat_angles

landsat_angles: ias_lib/liblandsatang.a $(OBJ)
	$(CC) $(OPTS) $(INCS) -o $@ $^ $(LIBS) 

ias_lib/liblandsatang.a:
	cd ias_lib && make

#---------------------------------------
# Target to distribute the executable(s)
#---------------------------------------
.c.o: $(INCS) $(LIBS)
	$(CC) $(OPTS) $(INCS) -c -o $@ $<
clean:
	/bin/rm -f *.o
