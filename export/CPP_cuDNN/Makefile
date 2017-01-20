EXT_CPP=cpp
EXT_CUDA=cu

CUDA_PATH=/usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH=$(CUDA_PATH)/bin

ifdef ARCH32
	CUDA_LIB_PATH = $(CUDA_PATH)/lib
else
	CUDA_LIB_PATH = $(CUDA_PATH)/lib64
endif

TARGET=n2d2_cudnn_test
BIN_DIR=bin
OBJ_DIR=obj

CPP_FILES=$(wildcard src/*.$(EXT_CPP)) $(wildcard *.$(EXT_CPP))
CUDA_FILES=$(wildcard src/*.$(EXT_CUDA)) $(wildcard *.$(EXT_CUDA))
INCLUDES=$(wildcard *.hpp) $(wildcard include/*.hpp) $(wildcard include/*.h)

ifndef CXX
  CXX=g++
endif

ifndef NVXX
  NVXX=$(CUDA_BIN_PATH)/nvcc
endif

ifdef PROFILING
  CPPFLAGS:=$(CPPFLAGS) -DPROFILING
endif

CPPFLAGS:=$(CPPFLAGS) -I./include/ -I$(CUDA_INC_PATH) -IUtilNPP -std=c++11
LIBS=-lcudart -lnppi -lnppc -lcublas -lcudnn
LPPFLAGS=-L$(CUDA_LIB_PATH) $(LIBS)
NVFLAGS:=$(CPPFLAGS) -lcutil -lcudpp -lcudart -lnppi -lnppc -lm -lstdc++

OBJ_CUDA = $(addprefix $(OBJ_DIR)/,$(CUDA_FILES:.$(EXT_CUDA)=.ocu))
OBJ_FILES = $(addprefix $(OBJ_DIR)/,$(CPP_FILES:.$(EXT_CPP)=.o))

$(BIN_DIR)/$(TARGET):  $(OBJ_FILES) $(OBJ_CUDA)
	$(CXX) -o $@ $^ $(LPPFLAGS)

$(OBJ_DIR)/%.ocu : %.$(EXT_CUDA) $(INCLUDES)
	@mkdir -p $(@D)
	$(NVXX) -c -o $@ $< $(NVFLAGS)

$(OBJ_DIR)/%.o: %.$(EXT_CPP) $(INCLUDES)
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

all: $(OBJ_CUDA) $(OBJ_FILES)

$(OBJ_CUDA):  | $(OBJ_DIR)
$(OBJ_FILES): | $(OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
