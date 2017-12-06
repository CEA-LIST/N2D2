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

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")

OPTION(CHECK_COVERAGE "Generate coveralls data" OFF)

if (NOT CMAKE_BUILD_TYPE)
    MESSAGE(STATUS "No build type selected, default to Release")
    SET(CMAKE_BUILD_TYPE "Release")
endif()

# Find required & optional packages
if(MSVC)
    if (EXISTS "$ENV{DIRENT_INCLUDE_DIR}")
        SET(CMAKE_REQUIRED_INCLUDES
            "${CMAKE_REQUIRED_INCLUDES} $ENV{DIRENT_INCLUDE_DIR}")
        INCLUDE_DIRECTORIES(SYSTEM $ENV{DIRENT_INCLUDE_DIR})
    endif()

    INCLUDE(CheckIncludeFile)
    CHECK_INCLUDE_FILE(dirent.h HAVE_DIRENT_H)

    if (NOT HAVE_DIRENT_H)
        MESSAGE(FATAL_ERROR "dirent.h required - you can download it and"
            " install it from http://www.softagalleria.net/dirent.php")
    endif()
endif()

FIND_PACKAGE(Gnuplot REQUIRED)

# Define environment variable OpenCV_DIR to point to for example
# "C:\OpenCV\opencv\build"
if (EXISTS "$ENV{OpenCV_DIR}")
    INCLUDE("$ENV{OpenCV_DIR}/OpenCVConfig.cmake")
endif()

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

FIND_PACKAGE(OpenMP)
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")

FIND_PACKAGE(CUDA)
if (CUDA_FOUND)
    INCLUDE_DIRECTORIES(SYSTEM ${CUDA_INCLUDE_DIRS})
    GET_FILENAME_COMPONENT(CUDA_LIB_DIR ${CUDA_CUDART_LIBRARY} PATH)

    message(STATUS "CUDA library status:")
    message(STATUS "    version: ${CUDA_VERSION_STRING}")
    message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "    libraries: ${CUDA_LIBRARIES}")

    if(MSVC)
        set(CUDNN_LIB_NAME "cudnn.lib")
    else()
        set(CUDNN_LIB_NAME "libcudnn.so")
    endif()

    FIND_PATH(CUDNN_INCLUDE_DIR cudnn.h
        PATHS ${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_INCLUDE}
        DOC "Path to CuDNN include directory." )
    FIND_LIBRARY(CUDNN_LIB_DIR NAMES ${CUDNN_LIB_NAME}
        PATHS ${CUDNN_INCLUDE_DIR} ${CUDA_LIB_DIR}
        DOC "Path to CuDNN library.")

    if (CUDNN_INCLUDE_DIR AND CUDNN_LIB_DIR)
        file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_FILE_CONTENTS)

        string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
               CUDNN_VERSION_MAJOR "${CUDNN_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
        string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
               CUDNN_VERSION_MINOR "${CUDNN_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
               CUDNN_VERSION_PATCH "${CUDNN_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
               CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

        if(NOT CUDNN_VERSION_MAJOR)
            set(CUDNN_VERSION "?")
        else()
            set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
        endif()

        message(STATUS "CuDNN library status:")
        message(STATUS "    version: ${CUDNN_VERSION}")
        message(STATUS "    include path: ${CUDNN_INCLUDE_DIR}")
        message(STATUS "    libraries: ${CUDNN_LIB_DIR}")

        INCLUDE_DIRECTORIES(SYSTEM ${CUDNN_INCLUDE_DIR})

        SET(CUDA_LIBS "cudart;cublas;cudadevrt")
        SET(CUDNN_LIBS "cudnn")

        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA=1")
        SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -arch=sm_30")

        if (MSVC)
            SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -MD")
        else()
            SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
        endif()
    else()
        MESSAGE(WARNING "CUDA found but CuDNN seems to be missing - you can"
            " download it and install it from http://www.nvidia.com")
    endif()
endif()

FIND_PACKAGE(PugiXML)
if (PUGIXML_FOUND)
    INCLUDE_DIRECTORIES(SYSTEM ${PUGIXML_INCLUDE_DIR})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPUGIXML=1")
endif()

FIND_PACKAGE(MongoDB)
if (MongoDB_FOUND)
    FIND_PACKAGE(Boost COMPONENTS thread system filesystem program_options
        REQUIRED)
    FIND_PACKAGE(OpenSSL REQUIRED)

    INCLUDE_DIRECTORIES(SYSTEM ${MongoDB_INCLUDE_DIR})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMONGODB=1")
endif()

# Compiler flags
if(MSVC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    ADD_DEFINITIONS(-D_CONSOLE -D_VISUALC_ -DNeedFunctionPrototypes)
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS -D_VARIADIC_MAX=10)
    # /wd4250 disable 'class1' : inherits 'class2::member' via dominance
    # /wd4512 disable 'class' : assignment operator could not be generated
    ADD_DEFINITIONS(/wd4250 /wd4512)
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char -std=c++0x -fPIC")

    if(${OpenCV_VERSION} VERSION_EQUAL "2.0.0")
        MESSAGE(WARNING "Compiling with _GLIBCXX_PARALLEL flag")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_PARALLEL")
    endif()

    if(${OpenCV_VERSION} VERSION_LESS "2.2.0")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV_USE_OLD_HEADERS")
    endif()

    if(CHECK_COVERAGE)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
    else()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -s -DNDEBUG")
    endif()
endif()

SET(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -DN2D2_COMPILE_PATH=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")

MACRO(GET_DIRECTORIES return_list exp)
    FILE(GLOB_RECURSE new_list ${exp})
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

MACRO(N2D2_INCLUDE inc_path)
    GET_DIRECTORIES(headers_dirs ${inc_path}/*.hpp)
    INCLUDE_DIRECTORIES(${headers_dirs} ${inc_path})
    INSTALL(DIRECTORY ${inc_path} DESTINATION include)
ENDMACRO()

SET(SRC "")
SET(CU_SRC "")

MACRO(N2D2_AUX_SOURCE src_path)
    GET_DIRECTORIES(sources_dirs ${src_path}/*.cpp)
    FOREACH(source_dir ${sources_dirs})
        AUX_SOURCE_DIRECTORY(${source_dir} SRC)
    ENDFOREACH()

    if (CUDA_FOUND)
        FILE(GLOB_RECURSE CU_SRC_PATH "${src_path}/*.cu")
        LIST(APPEND CU_SRC "${CU_SRC_PATH}")
    endif()
ENDMACRO()

SET(N2D2_LIB "")

MACRO(N2D2_MAKE_LIBRARY name)
    SET(N2D2_LIB ${name})
    LINK_DIRECTORIES(${OpenCV_LIB_DIR})

    if (CUDA_FOUND)
        if (NOT "${CU_SRC}" STREQUAL "")
            CUDA_ADD_LIBRARY(${name} STATIC ${SRC} ${CU_SRC})
        else()
            ADD_LIBRARY(${name} STATIC ${SRC})
        endif()

        LINK_DIRECTORIES(${CUDA_LIB_DIR})
        LINK_DIRECTORIES(${CUDNN_LIB_DIR})
        TARGET_LINK_LIBRARIES(${name} ${CUDA_LIBS})
        TARGET_LINK_LIBRARIES(${name} ${CUDNN_LIBS})
    else()
        ADD_LIBRARY(${name} STATIC ${SRC})
    endif()

    if (PUGIXML_FOUND)
        TARGET_LINK_LIBRARIES(${name} ${PUGIXML_LIBRARIES})
    endif()

    if (MongoDB_FOUND)
        TARGET_LINK_LIBRARIES(${name} ${MongoDB_LIBRARIES})
        TARGET_LINK_LIBRARIES(${name} ${Boost_THREAD_LIBRARY}
            ${Boost_FILESYSTEM_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY}
            ${Boost_SYSTEM_LIBRARY} ${OPENSSL_LIBRARIES})
    endif()

    TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})

    INSTALL(TARGETS ${name}
       ARCHIVE DESTINATION lib
    )
ENDMACRO()

MACRO(N2D2_COPY_DIRECTORY target src dst)
    FILE(GLOB_RECURSE allfiles RELATIVE "${src}" FOLLOW_SYMLINKS "${src}/*")

    SET(file_targets "")
    FOREACH(each_file ${allfiles})
        if(NOT IS_DIRECTORY "${src}/${each_file}")
            ADD_CUSTOM_COMMAND(OUTPUT "${dst}/${each_file}" PRE_BUILD
                COMMAND ${CMAKE_COMMAND}
                ARGS -E copy "${src}/${each_file}" "${dst}/${each_file}")

            SET(file_targets ${file_targets} "${dst}/${each_file}")
        endif()
    ENDFOREACH(each_file)

    ADD_CUSTOM_TARGET(${target} ALL DEPENDS ${file_targets})
ENDMACRO()
