# This makefile does nothing but delegating the actual building to cmake
BUILDDIR := build
MAKEFLAGS := --no-print-directory

all: local

local:
	@mkdir -p ${BUILDDIR} && cd ${BUILDDIR} && cmake .. && ${MAKE} ${MAKEFLAGS}

debug:
	@mkdir -p ${BUILDDIR} && cd ${BUILDDIR} && cmake -DCMAKE_BUILD_TYPE=Debug .. && ${MAKE} ${MAKEFLAGS}

clean:
	if [ -d "${BUILDDIR}" ]; then rm -rf ${BUILDDIR}; fi
