EXT_CPP=cpp

TARGET=n2d2_cuda_test

CPP_FILES=$(wildcard src/*.$(EXT_CPP)) $(wildcard *.$(EXT_CPP))
INCLUDES=$(wildcard dnn/include/network.hpp) $(wildcard include/*.h) $(wildcard include/*.hpp) $(wildcard *.hpp)

ifndef DNNLIB
  DNNLIB=dnn
endif

ifndef CXX
  CXX=g++
endif

ifdef OUTPUTFILE
  CPPFLAGS:=$(CPPFLAGS) -DOUTXT
endif

ifdef NRET
  CPPFLAGS:=$(CPPFLAGS) -DNRET
endif

ifndef CXX
  CXX=g++
endif

ifndef DEBUG
  OPT:=-O3 -s -DNDEBUG

  ifndef NOMARCH
    OPT:=$(OPT) -march=native
  endif
else
  OPT:=-g -pg -rdynamic
endif

CPPFLAGS:=$(CPPFLAGS) -I./include/ -I./include/dnn/ -I./include/utils/
LPPFLAGS=-std=c++0x -L./$(DNNLIB)/bin_dnn/ -lnetworktCUDA "-Wl,-rpath,$(DNNLIB)/bin_dnn/"
CPPFLAGS:=$(CPPFLAGS) -std=c++0x


ifndef BIN_DIR_EXPORT_CUDA
  BIN_DIR_EXPORT_CUDA=bin
endif


OBJ_DIR_EXPORT_CUDA=$(BIN_DIR_EXPORT_CUDA).obj


OBJ_FILES = $(addprefix $(OBJ_DIR_EXPORT_CUDA)/,$(CPP_FILES:.$(EXT_CPP)=.o))

$(BIN_DIR_EXPORT_CUDA)/$(TARGET): $(OBJ_FILES)
	$(CXX) -o $@ $^ $(LPPFLAGS)

$(OBJ_DIR_EXPORT_CUDA)/%.o: %.$(EXT_CPP) $(INCLUDES)
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CPPFLAGS)


.PHONY: dnnlib
dnnlib:
	$(MAKE) -C $(DNNLIB)

all: $(OBJ_FILES)

$(OBJ_FILES): dnnlib | $(OBJ_DIR_EXPORT_CUDA)

$(OBJ_DIR_EXPORT_CUDA):
	mkdir -p $(OBJ_DIR_EXPORT_CUDA)
	mkdir -p $(BIN_DIR_EXPORT_CUDA)


clean:
	rm -rf $(OBJ_DIR_EXPORT_CUDA) $(BIN_DIR_EXPORT_CUDA)
	$(MAKE) clean -C $(DNNLIB)
