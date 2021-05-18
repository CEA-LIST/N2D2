EXT_CPP=cpp
EXT_CUDA=cu

CUDA_PATH=/usr/local/cuda
CUDA_INC_PATH=$(CUDA_PATH)/include
CUDA_BIN_PATH=$(CUDA_PATH)/bin

ifdef ARCH32
	CUDA_LIB_PATH = $(CUDA_PATH)/lib
else
	CUDA_LIB_PATH = $(CUDA_PATH)/lib64
endif

TARGET=libnetworktCUDA.so

CPP_FILES=$(wildcard src/*.$(EXT_CPP))
CUDA_FILES=$(wildcard src/*.$(EXT_CUDA))
INCLUDES=$(wildcard include/*.hpp) $(wildcard include/*.h)

ifndef CXX
  CXX=g++
endif

ifndef NVXX
  NVXX=$(CUDA_BIN_PATH)/nvcc
endif

ifdef OUTPUTFILE
  CPPFLAGS:=$(CPPFLAGS) -DOUTXT
endif

ifdef NRET
  CPPFLAGS:=$(CPPFLAGS) -DNRET
endif

NVCPPFLAGS:=$(CPPFLAGS) -I./include/ -I$(CUDA_INC_PATH) -IUtilNPP\
 -gencode arch=compute_50,code=sm_50\
 -gencode arch=compute_52,code=sm_52\
 -gencode arch=compute_53,code=sm_53

CUDA_CAPABILITY_6:= $(shell $(NVXX) --help | grep 'compute_60')
ifneq ($(CUDA_CAPABILITY_6),)
  $(info Compiling up to CUDA capability 6)
  NVCPPFLAGS:=$(NVCPPFLAGS) \
  -gencode arch=compute_60,code=sm_60\
  -gencode arch=compute_61,code=sm_61
endif

CUDA_CAPABILITY_7:= $(shell $(NVXX) --help | grep 'compute_70')
ifneq ($(CUDA_CAPABILITY_7),)
  $(info Compiling up to CUDA capability 7)
  NVCPPFLAGS:=$(NVCPPFLAGS) \
  -gencode arch=compute_70,code=sm_70\
  -gencode arch=compute_72,code=sm_72\
  -gencode arch=compute_75,code=sm_75
endif

CPPFLAGS:=$(CPPFLAGS) -I./include/ -I$(CUDA_INC_PATH) -IUtilNPP -std=c++0x -fPIC
LIBS=-lcudart -lnppc -lcublas
LPPFLAGS=-L$(CUDA_LIB_PATH) $(LIBS)
NVFLAGS:=$(NVCPPFLAGS) -lcutil -lcudpp -lcudart -lnppc -lm -lstdc++ -Xcompiler -fPIC

ifndef BIN_DIR_EXPORT_CUDA
  BIN_DIR_EXPORT_CUDA=bin_dnn
endif

OBJ_DIR_EXPORT_CUDA=$(BIN_DIR_EXPORT_CUDA).obj

OBJ_CUDA = $(addprefix $(OBJ_DIR_EXPORT_CUDA)/,$(CUDA_FILES:.$(EXT_CUDA)=.ocu))
OBJ_FILES = $(addprefix $(OBJ_DIR_EXPORT_CUDA)/,$(CPP_FILES:.$(EXT_CPP)=.o))

$(BIN_DIR_EXPORT_CUDA)/$(TARGET):  $(OBJ_FILES) $(OBJ_CUDA)
	$(CXX) -shared -o $@ $^ $(LPPFLAGS)

$(OBJ_DIR_EXPORT_CUDA)/%.ocu : %.$(EXT_CUDA) $(INCLUDES)
	@mkdir -p $(@D)
	$(NVXX) -c -o $@ $< $(NVFLAGS)

$(OBJ_DIR_EXPORT_CUDA)/%.o: %.$(EXT_CPP) $(INCLUDES)
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

all: $(OBJ_CUDA) $(OBJ_FILES)

$(OBJ_CUDA):  | $(OBJ_DIR_EXPORT_CUDA)
$(OBJ_FILES): | $(OBJ_DIR_EXPORT_CUDA)

$(OBJ_DIR_EXPORT_CUDA):
	mkdir -p $(OBJ_DIR_EXPORT_CUDA)
	mkdir -p $(BIN_DIR_EXPORT_CUDA)

clean:
	rm -rf $(OBJ_DIR_EXPORT_CUDA) $(BIN_DIR_EXPORT_CUDA)
