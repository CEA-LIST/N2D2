##################################################
##: Different options to compile the export
##: Usage :
##:
##:    make help
##:         display the different options available
##:    make / make build
##:         compile the export without specific option
##:    make save_outputs
##:         compile the export with the option to save 
##:         the outputs of all layers 
##:         each time there is a network forward 
##:    make clean
##:         clean up the build and bin folders
##:
##################################################


CXX=g++
override INCLUDE_DIRS += -I. -I./include -I./dnn/include
override CXXFLAGS += -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -pedantic -fsigned-char -std=c++0x -fPIC $(INCLUDE_DIRS) -DSTIMULI_DIRECTORY="\"./stimuli\""
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


.PHONY: clean save_outputs help

clean:
	if [ -d "$(OBJDIR)" ]; then rm -rf $(OBJDIR); fi
	if [ -d "$(BINDIR)" ]; then rm -rf $(BINDIR); fi

save_outputs:
	$(MAKE) all "SAVE_OUTPUTS=1"

help:
	@grep -e "^##:"  Makefile;

-include $(DEPENDENCIES)
