EXT_CPP=cpp

EXT_CUDA=cu
CUDA_PATH=/usr/local/cuda
CUDA_INC_PATH=$(CUDA_PATH)/include
CUDA_BIN_PATH=$(CUDA_PATH)/bin
CUDA_LIB_PATH:=

ifneq ($(wildcard $(CUDA_PATH)/lib64),)
CUDA_LIB_PATH:=$(CUDA_LIB_PATH) $(CUDA_PATH)/lib64/
endif

CUDA_LIB_PATH:=$(CUDA_LIB_PATH) $(CUDA_PATH)/lib/


ifndef WRAPPER_PYTHON
	TARGET=libnetworktRT.so
else
	#for boost
	TARGET=libnetworktRT.so
	#TARGET=networktRT.o
endif

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

ifdef TF
  CPPFLAGS:=$(CPPFLAGS) -DTF
endif

ifndef DEBUG
  ifndef NOPARALLEL
    LIBS:=$(LIBS) -fopenmp -lpthread
    NVFLAGS:=$(NVFLAGS) -fopenmp -lpthread
  endif

else
  NVFLAGS:=$(NVFLAGS) -G -g
  CPPFLAGS:=$(CPPFLAGS) -g -pg -O0
  LPPFLAGS:=$(LPPFLAGS) -g -pg -O0
endif

CPPFLAGS:=$(CPPFLAGS) -I./include/ -isystem $(CUDA_INC_PATH)
LIBS:=$(LIBS) -lcudart -lcublas -lcudnn -lnvinfer
LPPFLAGS=$(foreach lib_dir,$(CUDA_LIB_PATH),-L$(lib_dir)) $(LIBS)
NVFLAGS:=$(CPPFLAGS) -lcutil -lcudpp -lcudart -lm -lstdc++ \
    -std=c++11 -Xcompiler -fPIC
CPPFLAGS:=$(CPPFLAGS) -std=c++11 -fPIC

ifndef BIN_DIR_EXPORT_TENSORRT
  BIN_DIR_EXPORT_TENSORRT=bin_dnn
endif


OBJ_DIR_EXPORT_TENSORRT=$(BIN_DIR_EXPORT_TENSORRT).obj

OBJ_CUDA = $(addprefix $(OBJ_DIR_EXPORT_TENSORRT)/,$(CUDA_FILES:.$(EXT_CUDA)=.ocu))
OBJ_FILES = $(addprefix $(OBJ_DIR_EXPORT_TENSORRT)/,$(CPP_FILES:.$(EXT_CPP)=.o))

$(BIN_DIR_EXPORT_TENSORRT)/$(TARGET):  $(OBJ_FILES) $(OBJ_CUDA)
	$(CXX) -shared -o $@ $^ $(LPPFLAGS)

$(OBJ_DIR_EXPORT_TENSORRT)/%.ocu : %.$(EXT_CUDA) $(INCLUDES)
	@mkdir -p $(@D)
	$(NVXX) -c -o $@ $< $(NVFLAGS)

$(OBJ_DIR_EXPORT_TENSORRT)/%.o: %.$(EXT_CPP) $(INCLUDES)
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

all: $(OBJ_CUDA) $(OBJ_FILES)

$(OBJ_CUDA):  | $(OBJ_DIR_EXPORT_TENSORRT)
$(OBJ_FILES): | $(OBJ_DIR_EXPORT_TENSORRT)

$(OBJ_DIR_EXPORT_TENSORRT):
	mkdir -p $(OBJ_DIR_EXPORT_TENSORRT)
	mkdir -p $(BIN_DIR_EXPORT_TENSORRT)

clean:
	rm -rf $(OBJ_DIR_EXPORT_TENSORRT) $(BIN_DIR_EXPORT_TENSORRT)