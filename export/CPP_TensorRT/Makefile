EXT_CPP=cpp
TARGET=n2d2_tensorRT_test

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
	TARGET=n2d2_tensorRT_inference.so
endif

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

ifdef CV_SUPPORT
  CPPFLAGS:=$(CPPFLAGS)`pkg-config opencv --cflags` -DCV_SUPPORT
  LPPFLAGS:=$(LPPFLAGS)`pkg-config opencv --cflags --libs` -lpthread
endif

ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
DNN_INSTALLL_PATH=$(ROOT_DIR)/$(DNNLIB)/bin_dnn/libnetworktRT.so

ifdef WRAPPER_PYTHON
	# location of the Python header files
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

ifeq ($(WRAPPER_PYTHON),3.6m)
		CPPFLAGS:=$(CPPFLAGS) -DWRAPPER_PYTHON=1 -fPIC -I$(PYTHON_INCLUDE)
ifndef BOOST_NATIVE
		LPPFLAGS:=$(LPPFLAGS) -std=c++0x -L$(BOOST_LIB) -lboost_python3 -L$(BOOST_LIB) -lboost_numpy3 -L$(ARCHLIB) -lpython$(PYTHON_VERSION)
else
		LPPFLAGS:=$(LPPFLAGS) -std=c++0x -L$(BOOST_LIB) -lboost_python -L$(BOOST_LIB) -lboost_numpy -L$(ARCHLIB) -lpython$(PYTHON_VERSION)
endif
else ifeq ($(WRAPPER_PYTHON),2.7)
		CPPFLAGS:=$(CPPFLAGS) -DWRAPPER_PYTHON=1 -fPIC -I/usr/include/python$(PYTHON_VERSION)
		LPPFLAGS:=$(LPPFLAGS) -std=c++0x -L$(BOOST_LIB) -lboost_python -L$(BOOST_LIB) -lboost_numpy -L$(ARCHLIB) -lpython$(PYTHON_VERSION)
endif

else
  ifndef WRAPPER_CPP
		  LPPFLAGS:=$(LPPFLAGS) -std=c++0x -L./$(DNNLIB)/bin_dnn/ -lnetworktRT "-Wl,-rpath,$(DNNLIB)/bin_dnn/"
  else
		  CPPFLAGS:=$(CPPFLAGS) -DWRAPPER_CPP=1 -fPIC
		  LPPFLAGS:=$(LPPFLAGS) -std=c++0x -L./$(DNNLIB)/bin_dnn/ -lnetworktRT "-Wl,-rpath,$(DNNLIB)/bin_dnn/"
  endif
endif


CPP_FILES=$(wildcard src/*.$(EXT_CPP))
INCLUDES=$(wildcard dnn/include/network.hpp) $(wildcard include/*.h) $(wildcard include/*.hpp) $(wildcard *.hpp)

ifndef WRAPPER_PYTHON
	CPP_FILES:= $(CPP_FILES) $(wildcard $(MAIN).$(EXT_CPP))
endif


CPPFLAGS:=$(CPPFLAGS) -I./include/ -I./include/dnn/ -I./include/utils/ -std=c++0x


ifdef WRAPPER_PYTHON
    PYTHON_DEFAULT_VERSION = 2.7
    CPPFLAGS:=$(CPPFLAGS) -DWRAPPER_PYTHON=1 -fPIC
    LPPFLAGS:=$(LPPFLAGS) -std=c++0x -L./$(DNNLIB)/bin_dnn/ -lnetworktRT "-Wl,-rpath,$(DNNLIB)/bin_dnn/"
endif

ifndef BIN_DIR_EXPORT_TENSORRT
  BIN_DIR_EXPORT_TENSORRT=bin
endif

ifdef DEBUG
  CPPFLAGS:=$(CPPFLAGS) -g -pg -O2 -rdynamic
  LPPFLAGS:=$(LPPFLAGS) -g -pg -O2 -rdynamic
endif

export DEBUG

OBJ_DIR_EXPORT_TENSORRT=$(BIN_DIR_EXPORT_TENSORRT).obj

OBJ_FILES = $(addprefix $(OBJ_DIR_EXPORT_TENSORRT)/,$(CPP_FILES:.$(EXT_CPP)=.o))

ifdef WRAPPER_PYTHON
	#DNNFLAGS:= ./$(DNNLIB)/bin_dnn/networktRT.o
	DNNFLAGS:= $(DNN_INSTALLL_PATH)
	OBJ_FILES := $(OBJ_FILES) $(DNNFLAGS)
	SHARE_OBJ := -shared
endif

ifdef WRAPPER_CPP
	#DNNFLAGS:= ./$(DNNLIB)/bin_dnn/networktRT.o
	DNNFLAGS:= $(DNN_INSTALLL_PATH)
	OBJ_FILES := $(OBJ_FILES) $(DNNFLAGS)
	SHARE_OBJ := -shared
endif

#for boost add share
$(BIN_DIR_EXPORT_TENSORRT)/$(TARGET):  $(OBJ_FILES)
		$(CXX) $(SHARE_OBJ) -o $@ $^ $(LPPFLAGS)

$(OBJ_DIR_EXPORT_TENSORRT)/%.o: %.$(EXT_CPP) $(INCLUDES)
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

.PHONY: dnnlib
dnnlib:
	$(MAKE) -C $(DNNLIB)

all: $(OBJ_FILES)

$(OBJ_FILES): dnnlib | $(OBJ_DIR_EXPORT_TENSORRT)

$(OBJ_DIR_EXPORT_TENSORRT):
	mkdir -p $(OBJ_DIR_EXPORT_TENSORRT)
	mkdir -p $(BIN_DIR_EXPORT_TENSORRT)

clean:
	rm -rf $(OBJ_DIR_EXPORT_TENSORRT) $(BIN_DIR_EXPORT_TENSORRT)
	$(MAKE) clean -C $(DNNLIB)