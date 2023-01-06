
TARGET := ${BINDIR}/n2d2_stm32_emulator

CXX := g++
CXXFLAGS := ${CXXFLAGS} -std=c++14 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -DN2D2_STM32_EMULATOR -DSTIMULI_DIRECTORY="\"stimuli\"" -MMD

INCLUDE_DIRS := -I./include -I. -I./dnn/include -I./targets/emulator
CXX_SRCS := $(shell find src dnn/src targets/emulator -iname "*.cpp")
CXX_OBJS := $(patsubst %.cpp, ${OBJDIR}/%.o, ${CXX_SRCS})
DEPENDENCIES := $(patsubst %.o, %.d, ${CXX_OBJS})

ifdef SAVE_OUTPUTS
	CXXFLAGS := ${CXXFLAGS} -DSAVE_OUTPUTS
endif

all: build

build: ${CXX_OBJS}
	@mkdir -p $(dir ${TARGET})
	${CXX} ${CXX_OBJS} ${LDFLAGS} -o ${TARGET}

${OBJDIR}/%.o: %.cpp
	@mkdir -p $(dir $@)
	${CXX} ${CXXFLAGS} ${INCLUDE_DIRS} -c $< -o $@ 

-include $(DEPENDENCIES)