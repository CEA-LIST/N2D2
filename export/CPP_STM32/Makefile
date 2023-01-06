###########################################################
##: Different options to compile the export
##: Usage :
##:
##:    make / make help
##:         display the different options available
##:    make emulator
##:         compile the export for emulation
##:    make export_h7
##:         compile the export for STM32H7
##:    make export_l4
##:         compile the export for STM32L4
##:    make clean
##:         clean up the build and bin folders
##:
###########################################################

OBJDIR := build
BINDIR := bin
MAKEFLAGS := --no-print-directory

# For SubMakefiles
export OBJDIR BINDIR

all: help

help:
	@grep -e "^##:"  Makefile;

clean:
	if [ -d "$(OBJDIR)" ]; then rm -rf $(OBJDIR); fi
	if [ -d "$(BINDIR)" ]; then rm -rf $(BINDIR); fi

emulator:
	${MAKE} -f targets/emulator/emulator.mak

export_h7:
	${MAKE} -f targets/stm32_h7/stm32h7.mak

export_l4:
	${MAKE} -f targets/stm32_l4/stm32l4.mak
