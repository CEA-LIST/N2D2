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

MAKEFILE_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

EXT=cpp
EXT_CUDA=cu

BIN:=$(foreach path, $(PARENT), $(subst .$(EXT),, $(shell find "$(path)/exec/" -name "*.$(EXT)")))
BIN_TESTS:=$(foreach path, $(PARENT), $(subst .$(EXT),, $(shell find "$(path)/tests/" -name "*.$(EXT)")))

ifndef CXX
  CXX=g++
endif

ifndef OPENCV
  OPENCV=opencv
endif

CPPFLAGS:=`pkg-config $(OPENCV) --cflags`
LDFLAGS:=`pkg-config $(OPENCV) --cflags --libs`

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
    -lm -lstdc++ -arch=sm_30 -maxrregcount 64

  NVFLAGS:=$(NVFLAGS) -gencode arch=compute_30,code=sm_30 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_52,code=sm_52 \
    -gencode arch=compute_53,code=sm_53

  CUDA_CAPABILITY_6:= $(shell $(NVCC) --help | grep 'compute_60')

  ifneq ($(CUDA_CAPABILITY_6),)
    $(info Compiling up to CUDA capability 6)
    NVFLAGS:=$(NVFLAGS) \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_62,code=sm_62
  endif
endif

ifdef PUGIXML
  CPPFLAGS:=$(CPPFLAGS) -DPUGIXML
  LDFLAGS:=$(LDFLAGS) -lpugixml
endif

ifdef JSONCPP
  CPPFLAGS:=$(CPPFLAGS) -DJSONCPP
  LDFLAGS:=$(LDFLAGS) -ljsoncpp
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

ifeq ($(shell pkg-config $(OPENCV) --modversion),2.0.0)
  # _GLIBCXX_PARALLEL needs to be defined for OpenCV 2.0.0 compiled with OpenMP
  $(info Compiling with _GLIBCXX_PARALLEL flag)
  CPPFLAGS:=$(CPPFLAGS) -D_GLIBCXX_PARALLEL
endif

OPENCV_USE_OLD_HEADERS:= $(shell expr \
    `pkg-config $(OPENCV) --modversion | sed 's/[.]//g'` \< 220)

ifeq ($(OPENCV_USE_OLD_HEADERS),1)
  CPPFLAGS:=$(CPPFLAGS) -DOPENCV_USE_OLD_HEADERS
endif

ifdef FLEXLM
  CPPFLAGS:=$(CPPFLAGS) -DFLEXLM -isystem ${LM_PATH}/../machind \
    -DNO_ACTIVATION_SUPPORT
  LDFLAGS:=$(LDFLAGS) -l:lm_new_pic.o \
    -L${LM_PATH} -llmgr_pic_trl -lcrvs_pic -lsb_pic -llmgr_dongle_stub_pic \
    -L${LM_PATH}/activation/lib -lnoact_pic -ldl
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
    OPT:=-Werror

    ifdef CHECK_COVERAGE
      OPT:=$(OPT) -O0 -g
    else
      OPT:=$(OPT) -O3 -s -DNDEBUG
    endif

    ifndef NOPARALLEL
      OPT:=$(OPT) -fopenmp
    endif

    ifndef NOMARCH
      OPT:=$(OPT) -march=native
    endif
  else
    # Compile in debug with -O2
    OPT:=-O2 -g -rdynamic

    ifdef PROFILING
      $(info Compiling with -pg flag can cause gdb to hang on __libc_fork call)
      # !!! See note below !!!
      OPT:=$(OPT) -Wl,--no-as-needed -ldl -pg
    endif

    # Debug with OpenMP by default. Call make with NOPARALLEL=1 to avoid OpenMP.
    ifndef NOPARALLEL
      OPT:=$(OPT) -fopenmp
    endif

    ############################################################################
    # Note when using PROFILING:
    # For possible future investigations...
    # Debug with -pg cause gdb to hang over __libc_fork() (called by popen()):

    # #0  0x00007fffe474ee34 in __libc_fork ()
    #     at ../nptl/sysdeps/unix/sysv/linux/x86_64/../fork.c:130
    # #1  0x00007fffe46f9928 in _IO_new_proc_open (fp=fp@entry=0x7ffefe122e90,
    #     command=command@entry=0xde3906 "gnuplot", mode=<optimized out>,
    #     mode@entry=0xe28fe6 "w") at iopopen.c:183
    # #2  0x00007fffe46f9c8c in _IO_new_popen (command=0xde3906 "gnuplot",
    #     mode=0xe28fe6 "w") at iopopen.c:301

    # strace -p `pidof gdb` during hanging returns:
    # (The SIGPROF interrupt signal seems to be related with the -pg option)

    # --- SIGCHLD {si_signo=SIGCHLD, si_code=CLD_TRAPPED, si_pid=20987, si_status=SIGPROF, si_utime=71074, si_stime=4998742} ---
    # rt_sigreturn()                          = -1 EINTR (Interrupted system call)
    # wait4(-1, 0x7ffc778112f4, WNOHANG|__WCLONE, NULL) = 0
    # wait4(-1, [{WIFSTOPPED(s) && WSTOPSIG(s) == SIGPROF}], WNOHANG, NULL) = 20987
    # tkill(20987, SIG_0)                     = 0
    # ptrace(PTRACE_CONT, 20987, 0x1, SIGPROF) = 0
    # wait4(-1, 0x7ffc778112f4, WNOHANG|__WCLONE, NULL) = 0
    # wait4(-1, 0x7ffc778112f4, WNOHANG, NULL) = 0
    # open("/proc/20987/status", O_RDONLY|O_CLOEXEC) = 25
    # fstat(25, {st_mode=S_IFREG|0444, st_size=0, ...}) = 0
    # mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f71844ce000
    # read(25, "Name:\tn2d2\nState:\tR (running)\nTg"..., 1024) = 834
    # close(25)                               = 0
    # munmap(0x7f71844ce000, 4096)            = 0
    # rt_sigsuspend([])                       = ? ERESTARTNOHAND (To be restarted if no handler)

    # Tested environments:
    # gcc version 4.8.4 (Ubuntu 4.8.4-2ubuntu1~14.04.4)
    # gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.10)
    ############################################################################
  endif

  CPPFLAGS:=$(CPPFLAGS) -Wall -Wextra -pedantic -fsigned-char -std=c++0x -fPIC $(OPT)
  LDFLAGS:=$(LDFLAGS) -Wall -Wextra -pedantic -std=c++0x -fPIC $(OPT)

  ifdef CHECK_COVERAGE
    CPPFLAGS:=$(CPPFLAGS) -fprofile-arcs -ftest-coverage
    LDFLAGS:=$(LDFLAGS) -lgcov
  endif
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
SRC=$(foreach path, $(PARENT), $(shell find "$(path)/src/" -name "*.$(EXT)"))
SRC_CUDA=$(foreach path, $(PARENT), $(shell find "$(path)/src/" -name "*.$(EXT_CUDA)"))
INCLUDES=$(foreach path, $(PARENT), $(shell find "$(path)/include/" -name "*.hpp"))

PCH_SRC=$(MAKEFILE_DIR)/include/Precompiled.hpp
PCH_OUT=$(PCH_SRC).gch

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

define copy-to-bin
	@mkdir -p $(N2D2_BINDIR)/$2
	@cp $1 $(N2D2_BINDIR)/$2/ > /dev/null 2>&1 || :
endef

define copy-resources-to-bin
	@rsync -av $1/$2 $(N2D2_BINDIR)/$1/ --exclude *.cpp \
	    > /dev/null 2>&1 || :
endef

define run-if-exists
	if [ -f "$(N2D2_BINDIR)/$1" ]; then \
	    MAKEFLAGS=; \
	    $(N2D2_BINDIR)/$1 || exit 1; \
	fi
endef

exec : flexlm $(addprefix $(N2D2_BINDIR)/, $(BIN))
ifdef FLEXLM
	$(shell rm -f ${LM_PATH}/lm_new_pic.o)
endif
	$(foreach path,$(PARENT),$(call copy-resources-to-bin,$(path),exec);)

tests : flexlm $(addprefix $(N2D2_BINDIR)/, $(BIN_TESTS))
ifdef FLEXLM
	$(shell rm -f ${LM_PATH}/lm_new_pic.o)
endif
	$(foreach path,$(PARENT),$(call copy-resources-to-bin,$(path),tests);)
	@$(foreach path,$(PARENT),$(call run-if-exists,$(path)/tests/run_all.sh);)

all : exec tests

debug :
	$(MAKE) all "DEBUG=1"

flexlm:
ifdef FLEXLM
	$(info Compiling with FlexLM license manager)
	${LM_PATH}/lmcrypt $(FLEXLM)
	$(MAKE) -s -C ${LM_PATH} PIE=1 lm_new_pic.o
	$(call copy-to-bin,${LM_PATH}/ceatech,flexlm)
	$(call copy-to-bin,README-FLEXLM,flexlm)
	$(call copy-to-bin,${LM_PATH}/lmgrd,flexlm)
	$(call copy-to-bin,${LM_PATH}/lmstat,flexlm)
	$(call copy-to-bin,${LM_PATH}/lmdown,flexlm)
	$(call copy-to-bin,${LM_PATH}/lmdiag,flexlm)
	$(call copy-to-bin,${LM_PATH}/lmhostid,flexlm)
	$(call copy-to-bin,$(FLEXLM),flexlm)
endif

$(N2D2_BINDIR)/% : $(OBJ) $(OBJ_CUDA) $(OBJDIR)/%.o
	@mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS)
	@if git rev-parse --git-dir > /dev/null 2>&1; then \
        git log -1 -p --submodule > $@.gitrev; \
        git submodule foreach --recursive git log -1 >> $@.gitrev; \
        git diff HEAD > $@.patch; \
        git submodule foreach --recursive git diff HEAD >> $@.patch; \
            [ -s $@.patch ] || rm -f $@.patch; \
	fi

ifneq (,$(filter $(MAKECMDGOALS),clean clean-all))
  -include $(OBJ:%.o=%.d)
endif

$(PCH_OUT): $(PCH_SRC)
	$(CXX) $(CPPFLAGS) -o $@ $<

.PRECIOUS : $(OBJDIR)/%.o
$(OBJDIR)/%.o : %.$(EXT) $(INCLUDES) $(PCH_OUT)
	@mkdir -p $(@D)
	$(call make-depend,$<,$@,$(patsubst %.o,%.d,$@))
	$(CXX) -o $@ -c $< $(CPPFLAGS) -include $(PCH_SRC)

ifdef CUDA
  .PRECIOUS : $(OBJDIR)/%.ocu
  $(OBJDIR)/%.ocu : %.$(EXT_CUDA) $(INCLUDES) $(PCH_OUT)
	@mkdir -p $(@D)
	$(call make-depend,$<,$@,$(patsubst %.o,%.d,$@))
	$(NVCC) -o $@ -c $< $(NVFLAGS) -include $(PCH_SRC)
endif

doc : $(SRC) $(SRC_CUDA) $(wildcard include/*.hpp) doxygen.cfg
	doxygen doxygen.cfg

.PHONY : clean

clean :
	@rm -rf $(OBJDIR) $(addprefix $(N2D2_BINDIR)/, $(BIN)) $(addprefix $(N2D2_BINDIR)/, $(BIN_TESTS)) doc/ $(PCH_OUT)

.PHONY : clean-all

clean-all :
	@rm -rf $(OBJDIR) $(N2D2_BINDIR) doc/

