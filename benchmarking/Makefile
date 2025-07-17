ROOT_DIR =	/work/sc139/sc139/s2767572/LBM-main

INCS	=	-I$(ROOT_DIR)/src
DEFS    =	-g -O3 -Wno-deprecated -Wall -std=c++17
LIBS	=	-lm
COMP	=	g++
DEPS    =       
OBJ     =	objective.cpp

define strip_templates
endef
ifdef strip_templates
STRIP_TEMPLATES = -fdiagnostics-color=always 2>&1 | sed 's/ \[with [^]]*\]//g'
endif

all: run.exe 

%.o: %.cpp $(DEPS)
	$(COMP) -c -o $@ $< $(DEFS) $(LIBS)

run.exe: $(OBJ)
	$(COMP) -o run.exe $(OBJ) $(INCS) $(DEFS) $(LIBS) $(STRIP_TEMPLATES)

clean:
	rm -f *.o run.exe