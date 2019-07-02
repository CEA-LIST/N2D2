EXT=c
BIN=n2d2_test

ifndef CC
  CC=gcc
endif

ifeq ($(CC),stxp70-clang)
  ifdef ASMP
    OPT:=-keep -mcore=stxp70asmp -DNO_DIRENT -D_OPENMP=200505 -fopenmp -Masmp=platform:ASMPCVE,PE_NR:8,lib:assert,pe0_stksize:8192,pen_stksize:4096,heapsize:16k -O2
  else
    OPT:=-keep -mcore=stxp70v4 -Mconfig=mult:yes -DNO_DIRENT -O2
  endif
else
  ifndef DEBUG
    OPT:=-O3 -s -DNDEBUG

    ifndef NOPARALLEL
      OPT:=$(OPT) -fopenmp
    endif

    ifndef NOMARCH
      OPT:=$(OPT) -march=native
    endif
  else
    OPT:=-g -pg -rdynamic
  endif

  OPT:=$(OPT) -std=c99
endif

ifeq ($(notdir $(CC)),$(filter $(notdir $(CC)),gcc cc))
  GCC_VERSION_GE_46=$(shell $(CC) -dumpversion | gawk '{print $$1>=4.6?"1":"0"}')

  ifeq ($(GCC_VERSION_GE_46),1)
    OPT:=$(OPT) -fstack-usage
  endif
endif

ifdef NL
    OPT:=$(OPT) -DNL
endif

ifdef OUTPUTFILE
    OPT:=$(OPT) -DOUTXT
endif

ifdef NRET
    OPT:=$(OPT) -DNRET
endif

ifdef NO_DIRENT
    OPT:=$(OPT) -DNO_DIRENT
endif

override CFLAGS +=-I./include/ -Wall -Wextra -Wno-unused-label -pedantic -Wconversion -fsigned-char $(OPT)
override LDFLAGS +=-lm -Wall -Wextra -Wno-unused-label -pedantic $(OPT)

ifndef BIN_DIR_EXPORT_C
  BIN_DIR_EXPORT_C=bin
endif

OBJ_DIR_EXPORT_C=$(BIN_DIR_EXPORT_C).obj
SRC=$(wildcard src/*.$(EXT))
INCLUDES=$(wildcard include/*.h)

OBJ:=$(SRC:%.$(EXT)=$(OBJ_DIR_EXPORT_C)/%.o)

# $(call make-depend,source-file,object-file,depend-file)
define make-depend
	$(CC) -MM       \
	 -MF $3         \
	 -MP            \
	 -MT $2         \
	 $(CFLAGS)    \
	 $1
endef

all : $(addprefix $(BIN_DIR_EXPORT_C)/, $(BIN))

stxp70 :
	$(MAKE) all "CC=stxp70-clang"

asmp :
	$(MAKE) all "CC=stxp70-clang" "ASMP=1"

run-stxp70 :
	sxrunprof -prof-calltree=prof-calltree.txt -prof-flat=prof-flat.txt -prof-flat-with-inst=prof-flat-with-inst.txt ./bin/n2d2_test $(ARGS)

run-asmp :
	asmp_run -p gepop --vcd --vcd-traces=* -- ./bin/n2d2_test $(ARGS)

debug :
	$(MAKE) all "DEBUG=1"

$(BIN_DIR_EXPORT_C)/% : $(OBJ) $(OBJ_DIR_EXPORT_C)/%.o
	@mkdir -p $(@D)
	$(CC) -o $@ $^ $(LDFLAGS)

ifneq (,$(filter $(MAKECMDGOALS),clean))
  -include $(OBJ:%.o=%.d)
endif

.PRECIOUS : $(OBJ_DIR_EXPORT_C)/%.o
$(OBJ_DIR_EXPORT_C)/%.o :  %.$(EXT) $(INCLUDES)
	@mkdir -p $(@D)
	$(call make-depend,$<,$@,$(patsubst %.o,%.d,$@))
	$(CC) -o $@ -c $< $(CFLAGS)

.PHONY : clean

clean :
	@rm -rf $(OBJ_DIR_EXPORT_C) $(BIN_DIR_EXPORT_C)
