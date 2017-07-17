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

SET(EXEC "")
GET_DIRECTORIES(exec_dirs ./*.cpp)
FOREACH(exec_dir ${exec_dirs})
    AUX_SOURCE_DIRECTORY(${exec_dir} EXEC)

    GET_FILENAME_COMPONENT(exec_bin ${exec_dir} NAME_WE)
    FILE(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${exec_bin})
ENDFOREACH()

foreach(f ${EXEC})
    GET_FILENAME_COMPONENT(bin ${f} NAME_WE)
    ADD_EXECUTABLE(${bin} ${f})
    TARGET_LINK_LIBRARIES(${bin} -Wl,-whole-archive ${N2D2_LIB} -Wl,-no-whole-archive)

    INSTALL(TARGETS ${bin}
        RUNTIME DESTINATION bin
    )
endforeach(f)
