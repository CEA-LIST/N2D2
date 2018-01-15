EXT_CPP=cpp
EXT_CUDA=cu

CUDA_PATH=/usr/local/cuda
CUDA_INC_PATH=$(CUDA_PATH)/include
CUDA_BIN_PATH=$(CUDA_PATH)/bin

ifneq ($(wildcard $(CUDA_PATH)/lib64),)
CUDA_LIB_PATH:=$(CUDA_LIB_PATH) $(CUDA_PATH)/lib64/
endif

CUDA_LIB_PATH:=$(CUDA_LIB_PATH) $(CUDA_PATH)/lib/

TARGET=libnetworktCUDNN.so

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

CPPFLAGS:=$(CPPFLAGS) -I./include/ -isystem $(CUDA_INC_PATH)
LIBS=-lcudart -lcublas -lcudnn
LPPFLAGS=$(foreach lib_dir,$(CUDA_LIB_PATH),-L$(lib_dir)) $(LIBS)
NVFLAGS:=$(CPPFLAGS) -lcutil -lcudpp -lcudart -lm -lstdc++ \
    -std=c++11 -Xcompiler -fPIC
CPPFLAGS:=$(CPPFLAGS) -std=c++0x -fPIC -fsigned-char

ifndef BIN_DIR_EXPORT_CUDNN
  BIN_DIR_EXPORT_CUDNN=bin_dnn
endif

OBJ_DIR_EXPORT_CUDNN=$(BIN_DIR_EXPORT_CUDNN).obj

OBJ_CUDA = $(addprefix $(OBJ_DIR_EXPORT_CUDNN)/,$(CUDA_FILES:.$(EXT_CUDA)=.ocu))
OBJ_FILES = $(addprefix $(OBJ_DIR_EXPORT_CUDNN)/,$(CPP_FILES:.$(EXT_CPP)=.o))

$(BIN_DIR_EXPORT_CUDNN)/$(TARGET):  $(OBJ_FILES) $(OBJ_CUDA)
	$(CXX) -shared -o $@ $^ $(LPPFLAGS)

$(OBJ_DIR_EXPORT_CUDNN)/%.ocu : %.$(EXT_CUDA) $(INCLUDES)
	@mkdir -p $(@D)
	$(NVXX) -c -o $@ $< $(NVFLAGS)

$(OBJ_DIR_EXPORT_CUDNN)/%.o: %.$(EXT_CPP) $(INCLUDES)
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

all: $(OBJ_CUDA) $(OBJ_FILES)

$(OBJ_CUDA):  | $(OBJ_DIR_EXPORT_CUDNN)
$(OBJ_FILES): | $(OBJ_DIR_EXPORT_CUDNN)

$(OBJ_DIR_EXPORT_CUDNN):
	mkdir -p $(OBJ_DIR_EXPORT_CUDNN)
	mkdir -p $(BIN_DIR_EXPORT_CUDNN)

clean:
	rm -rf $(OBJ_DIR_EXPORT_CUDNN) $(BIN_DIR_EXPORT_CUDNN)
