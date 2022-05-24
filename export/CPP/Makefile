CXX=g++
override INCLUDE_DIRS += -I. -I./include -I./dnn/include
override CXXFLAGS += -std=c++11 -O3 -march=native -fopenmp -Wall -Wextra -pedantic -fsigned-char -std=c++0x -fPIC $(INCLUDE_DIRS) -DSTIMULI_DIRECTORY="\"./stimuli\""
override LDFLAGS += -fopenmp
OBJDIR := build
BINDIR := bin
TARGET := $(BINDIR)/run_export

ifdef SAVE_OUTPUTS
	override CXXFLAGS += -DSAVE_OUTPUTS
endif


SRC_FILES:=$(shell find . -iname "*.cpp")
OBJECTS:=$(patsubst %.cpp, $(OBJDIR)/%.o, $(SRC_FILES))
DEPENDENCIES:=$(patsubst %.cpp, $(OBJDIR)/%.d, $(SRC_FILES))

all: build

build: $(OBJECTS)
	@mkdir -p $(dir $(TARGET))
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $(TARGET)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@


.PHONY: clean save_outputs

clean:
	if [ -d "$(OBJDIR)" ]; then rm -rf $(OBJDIR); fi
	if [ -d "$(BINDIR)" ]; then rm -rf $(BINDIR); fi

save_outputs:
	$(MAKE) all "SAVE_OUTPUTS=1"

-include $(DEPENDENCIES)
