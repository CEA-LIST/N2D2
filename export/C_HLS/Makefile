EXT=c
BIN=n2d2_test

ifndef CC
  CC=gcc
endif

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

ifeq ($(notdir $(CC)),$(filter $(notdir $(CC)),gcc cc))
  GCC_VERSION_GE_46=$(shell $(CC) -dumpversion | gawk '{print $$1>=4.6?"1":"0"}')

  ifeq ($(GCC_VERSION_GE_46),1)
    OPT:=$(OPT) -fstack-usage
  endif
endif

ifdef NL
    OPT:=$(OPT) -DNL
endif

CFLAGS:=-I./include/ -isystem /product/cad/xilinx/2014.4/Vivado_HLS/2014.4/include -DAESL_TB -DHAS_AP_CINT -Wall -Wextra -Wno-unused-label -pedantic -std=c99 -Wconversion -fsigned-char $(OPT)
LDFLAGS:=-lm -Wall -Wextra -Wno-unused-label -pedantic -std=c99 $(OPT)

ifndef BINDIR
  BINDIR=bin
endif

OBJDIR=$(BINDIR).obj
SRC=$(wildcard src/*.$(EXT))
INCLUDES=$(wildcard include/*.h)

OBJ:=$(SRC:%.$(EXT)=$(OBJDIR)/%.o)

# $(call make-depend,source-file,object-file,depend-file)
define make-depend
	$(CC) -MM       \
	 -MF $3         \
	 -MP            \
	 -MT $2         \
	 $(CFLAGS)    \
	 $1
endef

all : $(addprefix $(BINDIR)/, $(BIN))

debug :
	$(MAKE) all "DEBUG=1"

$(BINDIR)/% : $(OBJ) $(OBJDIR)/%.o
	@mkdir -p $(@D)
	$(CC) -o $@ $^ $(LDFLAGS)

ifneq (,$(filter $(MAKECMDGOALS),clean))
  -include $(OBJ:%.o=%.d)
endif

.PRECIOUS : $(OBJDIR)/%.o
$(OBJDIR)/%.o :  %.$(EXT) $(INCLUDES)
	@mkdir -p $(@D)
	$(call make-depend,$<,$@,$(patsubst %.o,%.d,$@))
	$(CC) -o $@ -c $< $(CFLAGS)

.PHONY : clean

clean :
	@rm -rf $(OBJDIR) $(BINDIR)
