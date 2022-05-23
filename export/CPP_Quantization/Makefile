CXX=g++
override INCLUDE_DIRS += -I. -I./include -I./dnn/include
override CXXFLAGS += -std=c++11 -O3 -march=native -fopenmp -Wall -Wextra -pedantic -fsigned-char -std=c++0x -fPIC $(INCLUDE_DIRS) -DSTIMULI_DIRECTORY="\"./stimuli\""
override LDFLAGS += -fopenmp

EXEC=run_export
SRC_FILES:=$(shell find . -iname "*.cpp")
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
