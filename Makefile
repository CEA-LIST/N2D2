################################################################################
#    (C) Copyright 2016 CEA LIST. All Rights Reserved.
#    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
#
#    This software is governed by the CeCILL-C license under French law and
#    abiding by the rules of distribution of free software.  You can  use,
#    modify and/ or redistribute the software under the terms of the CeCILL-C
#    license as circulated by CEA, CNRS and INRIA at the following URL
#    "http://www.cecill.info".
#
#    As a counterpart to the access to the source code and  rights to copy,
#    modify and redistribute granted by the license, users are provided only
#    with a limited warranty  and the software's author,  the holder of the
#    economic rights,  and the successive licensors  have only  limited
#    liability.
#
#    The fact that you are presently reading this means that you have had
#    knowledge of the CeCILL-C license and that you accept its terms.
################################################################################

ifndef PARENT
  PARENT=.
endif

EXT=cpp
EXT_CUDA=cu
BIN:=$(foreach path,$(PARENT), \
   $(subst .$(EXT),,$(wildcard $(path)/tests/*.$(EXT))) \
   $(subst .$(EXT),,$(wildcard $(path)/tests/*/*.$(EXT))) \
   $(subst .$(EXT),,$(wildcard $(path)/exec/*.$(EXT))) \
   $(subst .$(EXT),,$(wildcard $(path)/exec/*/*.$(EXT))))

ifndef CXX
  CXX=g++
endif

CPPFLAGS:=`pkg-config opencv --cflags`
LDFLAGS:=`pkg-config opencv --cflags --libs`

ifdef CUDA
  CUDA_PATH=/usr/local/cuda
  CUDA_INC_PATH=$(CUDA_PATH)/include
  CUDA_BIN_PATH=$(CUDA_PATH)/bin
  CUDA_LIB_PATH:=

  ifneq ($(wildcard $(CUDA_PATH)/lib64),)
    CUDA_LIB_PATH:=$(CUDA_LIB_PATH) $(CUDA_PATH)/lib64/
  endif

  CUDA_LIB_PATH:=$(CUDA_LIB_PATH) $(CUDA_PATH)/lib/

  NVCC=$(CUDA_BIN_PATH)/nvcc
  CPPFLAGS:=$(CPPFLAGS) -isystem $(CUDA_INC_PATH) -DCUDA
  LDFLAGS:=$(LDFLAGS) $(foreach lib_dir,$(CUDA_LIB_PATH),-L$(lib_dir)) \
    -lcudart -lcublas -lcudadevrt -lcudnn
  NVFLAGS:=$(CPPFLAGS) -std=c++11 -lcutil -lcudpp -lcudart -lnppi -lnppc \
    -lm -lstdc++
endif

ifdef PUGIXML
  CPPFLAGS:=$(CPPFLAGS) -DPUGIXML
  LDFLAGS:=$(LDFLAGS) -lpugixml
endif

ifdef MONGODB
  CPPFLAGS:=$(CPPFLAGS) -DMONGODB
  LDFLAGS:=$(LDFLAGS) -pthread -lmongoclient

  ifeq ($(shell ldconfig -p | grep libboost_thread-mt),)
    LDFLAGS:=$(LDFLAGS) -lboost_thread
  else
    LDFLAGS:=$(LDFLAGS) -lboost_thread-mt
  endif

  LDFLAGS:=$(LDFLAGS) -lboost_filesystem -lboost_program_options \
    -lboost_system -lssl -lcrypto
endif

ifeq ($(shell pkg-config opencv --modversion),2.0.0)
  # _GLIBCXX_PARALLEL needs to be defined for OpenCV 2.0.0 compiled with OpenMP
  $(info Compiling with _GLIBCXX_PARALLEL flag)
  CPPFLAGS:=$(CPPFLAGS) -D_GLIBCXX_PARALLEL
endif

ifeq ($(CXX),icpc)
  ifndef DEBUG
    OPT:=-O3 -ipo -no-prec-div -DNDEBUG

    ifndef NOPARALLEL
      OPT:=$(OPT) -openmp -parallel
    endif

    ifndef NOMARCH
      OPT:=$(OPT) -xHost
    endif
  else
    OPT:=-O0 -g -traceback -debug all
  endif

  CPPFLAGS:=$(CPPFLAGS) -w2 -Wall -Wcheck $(OPT)
  LDFLAGS:=$(LDFLAGS) -w2 -Wall -Wcheck $(OPT)
else
  ifndef DEBUG
    OPT:=-O3 -s -DNDEBUG -Werror

    ifndef NOPARALLEL
      OPT:=$(OPT) -fopenmp
    endif

    ifndef NOMARCH
      OPT:=$(OPT) -march=native
    endif
  else
    OPT:=-g -pg -rdynamic
  endif

  CPPFLAGS:=$(CPPFLAGS) -Wall -Wextra -pedantic -fsigned-char -std=c++0x -fPIC $(OPT)
  LDFLAGS:=$(LDFLAGS) -Wall -Wextra -pedantic -std=c++0x -fPIC $(OPT)
endif

CPPFLAGS:=$(CPPFLAGS) $(foreach path,$(PARENT),-I$(path)/include/)
NVFLAGS:=$(NVFLAGS) $(foreach path,$(PARENT),-I$(path)/include/)

ifdef DEBUG
  NVFLAGS:=$(NVFLAGS) -G -g
endif

CPPFLAGS:= $(CPPFLAGS) -DN2D2_COMPILE_PATH=\"${CURDIR}\"

ifndef N2D2_BINDIR
  N2D2_BINDIR=bin
endif

OBJDIR=$(N2D2_BINDIR).obj
SRC=$(foreach path,$(PARENT),$(wildcard $(path)/src/*.$(EXT)) \
 $(wildcard $(path)/src/*/*.$(EXT)) \
 $(wildcard $(path)/src/*/*/*.$(EXT)))
SRC_CUDA=$(foreach path,$(PARENT),$(wildcard $(path)/src/*.$(EXT_CUDA)) \
 $(wildcard $(path)/src/*/*.$(EXT_CUDA)) \
 $(wildcard $(path)/src/*/*/*.$(EXT_CUDA)))
INCLUDES=$(foreach path,$(PARENT),$(wildcard $(path)/*.hpp) \
 $(wildcard $(path)/include/*.hpp) \
 $(wildcard $(path)/include/*/*.hpp) \
 $(wildcard $(path)/include/*/*/*.hpp))

OBJ:=$(SRC:%.$(EXT)=$(OBJDIR)/%.o)
ifdef CUDA
  OBJ_CUDA:=$(SRC_CUDA:%.$(EXT_CUDA)=$(OBJDIR)/%.ocu)
endif

# $(call make-depend,source-file,object-file,depend-file)
define make-depend
	$(CXX) -MM       \
	 -MF $3         \
	 -MP            \
	 -MT $2         \
	 $(CPPFLAGS)    \
	 $1
endef

define copy-and-run
	@rsync -av $1/exec $(N2D2_BINDIR)/$1/ --exclude *.cpp \
	    > /dev/null 2>&1 || :
	@rsync -av $1/tests $(N2D2_BINDIR)/$1/ --exclude *.cpp \
	    > /dev/null 2>&1 || :
	@if [ -f "$(N2D2_BINDIR)/$1/tests/run_all.sh" ]; then \
	    $(N2D2_BINDIR)/$1/tests/run_all.sh || exit 1; \
	fi
endef

all : $(addprefix $(N2D2_BINDIR)/, $(BIN))
	$(foreach path,$(PARENT),$(call copy-and-run,$(path));)

debug :
	$(MAKE) all "DEBUG=1"

$(N2D2_BINDIR)/% : $(OBJ) $(OBJ_CUDA) $(OBJDIR)/%.o
	@mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS)
	@if git rev-parse --git-dir > /dev/null 2>&1; then \
        git log -1 > $@.gitrev; \
        git diff HEAD > $@.patch; [ -s $@.patch ] || rm -f $@.patch; \
	fi

ifneq (,$(filter $(MAKECMDGOALS),clean clean-all))
  -include $(OBJ:%.o=%.d)
endif

.PRECIOUS : $(OBJDIR)/%.o
$(OBJDIR)/%.o : %.$(EXT) $(INCLUDES)
	@mkdir -p $(@D)
	$(call make-depend,$<,$@,$(patsubst %.o,%.d,$@))
	$(CXX) -o $@ -c $< $(CPPFLAGS)

ifdef CUDA
  .PRECIOUS : $(OBJDIR)/%.ocu
  $(OBJDIR)/%.ocu : %.$(EXT_CUDA) $(INCLUDES)
	@mkdir -p $(@D)
	$(call make-depend,$<,$@,$(patsubst %.o,%.d,$@))
	$(NVCC) -o $@ -c $< $(NVFLAGS)
endif

doc : $(SRC) $(SRC_CUDA) $(wildcard include/*.hpp) doxygen.cfg
	doxygen doxygen.cfg

.PHONY : clean

clean :
	@rm -rf $(OBJDIR) $(addprefix $(N2D2_BINDIR)/, $(BIN)) doc/

.PHONY : clean-all

clean-all :
	@rm -rf $(OBJDIR) $(N2D2_BINDIR) doc/
