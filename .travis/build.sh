#!/bin/sh
################################################################################
#    (C) Copyright 2016 CEA LIST. All Rights Reserved.
#    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
#
#    This software is governed by the CeCILL-C license under French law and
#    abiding by the rules of distribution of free software.  You can  use,
#    modify and/ or redistribute the software under the terms of the CeCILL-C
#    license as circulated by CEA, CNRS and INRIA at the following URL
#    "http://www.cecill.info".
#
#    As a counterpart to the access to the source code and  rights to copy,
#    modify and redistribute granted by the license, users are provided only
#    with a limited warranty  and the software's author,  the holder of the
#    economic rights,  and the successive licensors  have only  limited
#    liability.
#
#    The fact that you are presently reading this means that you have had
#    knowledge of the CeCILL-C license and that you accept its terms.
################################################################################

if [ -n "$USE_CMAKE" ] ; then
    mkdir build
    cd build
    if [ -n "$CHECK_COVERAGE" ] ; then
        cmake -DCMAKE_BUILD_TYPE=Coverage -DCMAKE_CXX_FLAGS="-Werror" ..
        make -j $NUM_THREADS
        make -j $NUM_THREADS tests
        ctest -j 4 --output-on-failure
    else
        cmake -DCMAKE_CXX_FLAGS="-Werror" ..
        make -j $NUM_THREADS
        make -j $NUM_THREADS tests
        ctest -j 4 --output-on-failure
    fi
else
    ARGS="ONNX=1"

    if [ -n "$USE_CUDA" ] ; then
        ARGS="$ARGS CUDA=1"
    fi

    if [ -n "$CHECK_COVERAGE" ] ; then
        ARGS="$ARGS CHECK_COVERAGE=1"
    fi

    make all -j $NUM_THREADS $ARGS
fi

