EXT_CPP=cpp
TARGET=n2d2_tensorRT_test

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
  ifndef WRAPPER_CPP
	  ifndef MAIN
	    MAIN=$(TARGET)
	  else
	    MAIN=$(MAIN)
	  endif
  else
	  TARGET=libn2d2_tensorRT_inference.so
  endif

else
	TARGET=N2D2.so
  PYTHON_VERSION = $(WRAPPER_PYTHON)
	PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
	# location of the Boost Python include files and library
	BOOST_INC = /usr/local/include
	BOOST_LIB = /usr/local/lib
ifndef ARM
		ARCHLIB=/usr/lib/x86_64-linux-gnu/
else
		ARCHLIB=/usr/lib/aarch64-linux-gnu/
endif

ifdef BOOST_NATIVE
		CPPFLAGS:=$(CPPFLAGS) -DBOOST_NATIVE=1
endif

endif




CPP_FILES= $(wildcard dnn/src/*.$(EXT_CPP)) $(wildcard src/*.$(EXT_CPP))
CUDA_FILES=$(wildcard src/*.$(EXT_CUDA))
INCLUDES=$(wildcard dnn/include/*.hpp) $(wildcard include/plugins/*.hpp) $(wildcard include/*.hpp) $(wildcard include/*.h)

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

ifeq ($(WRAPPER_PYTHON),3.6m)
		CPPFLAGS:=$(CPPFLAGS) -DWRAPPER_PYTHON=1 -I$(PYTHON_INCLUDE)
ifndef BOOST_NATIVE
		LPPFLAGS:=$(LPPFLAGS) -std=c++0x -L$(BOOST_LIB) -lboost_python3 -L$(BOOST_LIB) -lboost_numpy3 -L$(ARCHLIB) -lpython$(PYTHON_VERSION)
else
		LPPFLAGS:=$(LPPFLAGS) -std=c++0x -L$(BOOST_LIB) -lboost_python -L$(BOOST_LIB) -lboost_numpy -L$(ARCHLIB) -lpython$(PYTHON_VERSION)
endif
endif

ifndef WRAPPER_PYTHON
	CPP_FILES:= $(CPP_FILES) $(wildcard $(MAIN).$(EXT_CPP))
endif
CPPFLAGS:=$(CPPFLAGS) -I./include/ -I./include/plugins/ -isystem $(CUDA_INC_PATH)
LIBS:=$(LIBS) -lcudart -lcublas -lcudnn -lnvinfer
LPPNVFLAGS=$(foreach lib_dir,$(CUDA_LIB_PATH),-L$(lib_dir)) $(LIBS)
LPPFLAGS:=$(LPPFLAGS) $(LPPNVFLAGS)

NVFLAGS:=$(CPPFLAGS) -lcutil -lcudpp -lcudart -lm -lstdc++ \
    -std=c++11 -Xcompiler -fPIC
CPPFLAGS:=$(CPPFLAGS) -std=c++11 -fPIC

ifndef BIN_DIR_EXPORT_TENSORRT
  BIN_DIR_EXPORT_TENSORRT=bin
endif

ifdef WRAPPER_PYTHON
  CPPFLAGS:=$(CPPFLAGS) -DWRAPPER_PYTHON=1
	SHARE_OBJ := -shared -Wl,-soname,$(TARGET).so
endif

ifdef WRAPPER_CPP
	SHARE_OBJ := -shared
endif

OBJ_DIR_EXPORT_TENSORRT=$(BIN_DIR_EXPORT_TENSORRT).obj

OBJ_CUDA = $(addprefix $(OBJ_DIR_EXPORT_TENSORRT)/,$(CUDA_FILES:.$(EXT_CUDA)=.ocu))
OBJ_FILES = $(addprefix $(OBJ_DIR_EXPORT_TENSORRT)/,$(CPP_FILES:.$(EXT_CPP)=.o))

$(BIN_DIR_EXPORT_TENSORRT)/$(TARGET):  $(OBJ_FILES) $(OBJ_CUDA)
	$(CXX) $(SHARE_OBJ) -o $@ $^ $(LPPFLAGS)

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