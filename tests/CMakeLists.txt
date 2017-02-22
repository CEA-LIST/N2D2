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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)

SET(TESTS "")
GET_DIRECTORIES(tests_dirs ./*.cpp)
FOREACH(tests_dir ${tests_dirs})
    AUX_SOURCE_DIRECTORY(${tests_dir} TESTS)

    GET_FILENAME_COMPONENT(tests_bin ${tests_dir} NAME_WE)
    FILE(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${tests_bin})
ENDFOREACH()

N2D2_COPY_DIRECTORY(copy-tests_data-n2d2
    ${CMAKE_CURRENT_SOURCE_DIR}/tests_data
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests_data)

foreach(f ${TESTS})
    GET_FILENAME_COMPONENT(bin ${f} NAME_WE)
    ADD_EXECUTABLE(${bin} ${f})
    TARGET_LINK_LIBRARIES(${bin} -Wl,-whole-archive ${N2D2_LIB} -Wl,-no-whole-archive)

    # Workaround for CMake bug (fixed in 3.3.0)
    # http://stackoverflow.com/questions/41597612/add-custom-command-does-not-re-run-on-failure
    ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bin}.passed
        COMMAND ${bin} # > ${bin}.log 2>&1
        COMMAND ${CMAKE_COMMAND} -E touch ${bin}.passed
        DEPENDS ${bin} copy-tests_data-n2d2
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
    ADD_CUSTOM_TARGET(
        run_${bin} ALL
        DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bin}.passed
    )
endforeach(f)
