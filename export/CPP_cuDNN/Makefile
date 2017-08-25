EXT_CPP=cpp

TARGET=n2d2_cudnn_test

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

CPPFLAGS:=$(CPPFLAGS) -I./include/ -I./include/dnn/ -I./include/utils/
LPPFLAGS=-std=c++0x -L./$(DNNLIB)/bin_dnn/ -lnetworktCUDNN "-Wl,-rpath,$(DNNLIB)/bin_dnn/"
CPPFLAGS:=$(CPPFLAGS) -std=c++0x

ifndef BIN_DIR_EXPORT_CUDNN
  BIN_DIR_EXPORT_CUDNN=bin
endif


OBJ_DIR_EXPORT_CUDNN=$(BIN_DIR_EXPORT_CUDNN).obj


OBJ_FILES = $(addprefix $(OBJ_DIR_EXPORT_CUDNN)/,$(CPP_FILES:.$(EXT_CPP)=.o))

$(BIN_DIR_EXPORT_CUDNN)/$(TARGET): $(OBJ_FILES)
	$(CXX) -o $@ $^ $(LPPFLAGS)

$(OBJ_DIR_EXPORT_CUDNN)/%.o: %.$(EXT_CPP) $(INCLUDES)
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CPPFLAGS)


.PHONY: dnnlib
dnnlib:
	$(MAKE) -C $(DNNLIB)

all: $(OBJ_FILES)

$(OBJ_FILES): dnnlib | $(OBJ_DIR_EXPORT_CUDNN)

$(OBJ_DIR_EXPORT_CUDNN):
	mkdir -p $(OBJ_DIR_EXPORT_CUDNN)
	mkdir -p $(BIN_DIR_EXPORT_CUDNN)


clean:
	rm -rf $(OBJ_DIR_EXPORT_CUDNN) $(BIN_DIR_EXPORT_CUDNN)
	$(MAKE) clean -C $(DNNLIB)
