#----------------------------------------------------------------------------
# Set up the compiler options.  
#----------------------------------------------------------------------------
OPTS    = -g -Wall -O2 -march=nocona -mfpmath=sse -msse2 
CC = gcc
#-------------------------------------------------------
# Define the executable(s), the pdf(s), and the html(s).
#-------------------------------------------------------
LIBS = -L./ias_lib/ -ll8ang -lm

INCS =	\
	-I./ias_lib/ \
	-I./

#---------------------------------------------------------
# Define the source code object files for each executable.
#---------------------------------------------------------
SRC	= \
	l8_angles.c \
	angles_api.c 

OBJ	= $(SRC:.c=.o)

#------------------------------------------------------------------------------
# Targets for each executable. 
#------------------------------------------------------------------------------
all: l8_angles

l8_angles: ias_lib/libl8ang.a $(OBJ)
	$(CC) $(OPTS) $(INCS) -o $@ $^ $(LIBS) 

ias_lib/libl8ang.a:
	cd ias_lib && make

#---------------------------------------
# Target to distribute the executable(s)
#---------------------------------------
.c.o: $(INCS) $(LIBS)
	$(CC) $(OPTS) $(INCS) -c -o $@ $<
clean:
	/bin/rm -f *.o
