CXX=g++
override INCLUDE_DIRS += -I. -I../include -I../dnn/include
override CXXFLAGS += -std=c++14 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable $(INCLUDE_DIRS) -DN2D2_STM32_EMULATOR -DSTIMULI_DIRECTORY="\"../stimuli\""

EXEC=stm32_emulator
SRC_FILES:=$(shell find . ../src ../dnn/src -iname "*.cpp")
OBJECTS:=$(patsubst %.cpp, %.o, $(SRC_FILES))
DEPENDENCIES:=$(patsubst %.cpp, %.d, $(SRC_FILES))

all: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CXX) -o $(EXEC) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

clean:
	rm -f $(DEPENDENCIES)
	rm -f $(OBJECTS)
	rm -f $(EXEC)

-include $(DEPENDENCIES)
