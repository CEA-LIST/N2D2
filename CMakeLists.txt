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
cmake_minimum_required(VERSION 3.1)

macro(set_globals)
    set(CMAKE_BUILD_TYPE_INIT Release)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage")
    set(CMAKE_MODULE_LINKER_FLAGS_COVERAGE "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} --coverage")
endmacro()

function(target_link_libraries_whole_archive target linked_libs)
    if(MSVC)
        foreach(lib IN LISTS  linked_libs ARGN)
            set_property(TARGET ${target} APPEND PROPERTY LINK_FLAGS "/WHOLEARCHIVE:${lib}")
            target_link_libraries(${target} PUBLIC ${lib})
        endforeach()
    else()
        target_link_libraries(${target} PUBLIC -Wl,--whole-archive ${linked_libs} ${ARGN} -Wl,--no-whole-archive)
    endif()
endfunction()

function(add_n2d2_test file_path target linked_libs)
    get_filename_component(file_name ${file_path} NAME_WE)

    add_executable(${file_name} EXCLUDE_FROM_ALL ${file_path})
    set_target_properties(${file_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    target_link_libraries_whole_archive(${file_name} ${linked_libs} ${ARGN})
    add_test(NAME "${file_name}" COMMAND ${file_name} WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_dependencies(${target} ${file_name})
endfunction()

function(add_n2d2_executable file_path target linked_libs)
    get_filename_component(file_name ${file_path} NAME_WE)

    add_executable(${file_name} EXCLUDE_FROM_ALL ${file_path})
    target_link_libraries_whole_archive(${file_name} ${linked_libs} ${ARGN})
    add_dependencies(${target} ${file_name})
endfunction()

# Must be called after find_package(CUDA) as the find_package call resets CUDA_NVCC_FLAGS
macro(set_nvcc_flags)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    # Must be replace by sm_80 for A100 architecture
    list(APPEND CUDA_NVCC_FLAGS -std=c++11;-arch=sm_30;)
    if (MSVC)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-MD;)
    else()
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC;)
    endif()
endmacro()


set_globals()
include("${CMAKE_CURRENT_LIST_DIR}/cmake/modules/cotire.cmake")
project(N2D2)




# n2d2_lib target
file(GLOB_RECURSE src_files "src/*.cpp")
file(GLOB_RECURSE src_files_python "src/python/*.cpp")
foreach(item ${src_files_python})
    list(REMOVE_ITEM src_files ${item})
endforeach()
add_library(n2d2_lib STATIC ${src_files})

target_include_directories(n2d2_lib PUBLIC "include")
target_compile_definitions(n2d2_lib PUBLIC N2D2_COMPILE_PATH=\"${CMAKE_SOURCE_DIR}\")

if(MSVC)
    # TODO clean-up warnings
    target_compile_options(n2d2_lib PUBLIC /W3 /wd4250 /wd4512 /wd4267 /wd4244 /wd4800 /wd4297)
    target_compile_definitions(n2d2_lib PUBLIC _CONSOLE _VISUALC_ NeedFunctionPrototypes _CRT_SECURE_NO_WARNINGS _VARIADIC_MAX=10)
else()
    target_compile_options(n2d2_lib PUBLIC -std=c++0x -Wall -Wextra -pedantic -Wno-unknown-pragmas -fsigned-char -fPIC)
endif()


find_package(OpenCV REQUIRED)
target_link_libraries(n2d2_lib PUBLIC ${OpenCV_LIBS})
target_include_directories(n2d2_lib PUBLIC ${OpenCV_INCLUDE_DIRS})

find_package(Threads REQUIRED)
target_link_libraries(n2d2_lib PUBLIC Threads::Threads)

find_package(Gnuplot REQUIRED)

find_package(OpenMP QUIET)
if(OpenMP_FOUND)
    target_compile_options(n2d2_lib PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(n2d2_lib PUBLIC ${OpenMP_CXX_FLAGS})

    if(NOT MSVC)
        # Fix for missing atomic lib
        target_link_libraries(n2d2_lib PUBLIC -Wl,--as-needed -Wl,-latomic -Wl,--no-as-needed)
    endif()

    target_compile_definitions(n2d2_lib PUBLIC OPENMP=1)
endif()

find_package(PugiXML QUIET)
if(PUGIXML_FOUND)
    target_include_directories(n2d2_lib PUBLIC ${PUGIXML_INCLUDE_DIR})
    target_link_libraries(n2d2_lib PUBLIC ${PUGIXML_LIBRARY})
    target_compile_definitions(n2d2_lib PUBLIC PUGIXML=1)
endif()

find_package(JsonCpp QUIET)
if(JsonCpp_FOUND)
    target_include_directories(n2d2_lib PUBLIC ${JsonCpp_INCLUDE_DIR})
    target_link_libraries(n2d2_lib PUBLIC ${JsonCpp_LIBRARY})
    target_compile_definitions(n2d2_lib PUBLIC JSONCPP=1)
endif()

find_package(MongoDB QUIET)
if (MongoDB_FOUND)
    find_package(Boost REQUIRED COMPONENTS filesystem program_options system thread)
    target_link_libraries(n2d2_lib PUBLIC ${Boost_LIBRARIES})

    find_package(OpenSSL REQUIRED)
    target_link_libraries(n2d2_lib PUBLIC ${OPENSSL_LIBRARIES})

    target_include_directories(n2d2_lib PUBLIC ${MongoDB_INCLUDE_DIR})
    target_link_libraries(n2d2_lib PUBLIC ${MongoDB_LIBRARIES})
    target_compile_definitions(n2d2_lib PUBLIC MONGODB=1)
endif()

INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf QUIET)
if (Protobuf_FOUND)
    file(DOWNLOAD https://raw.githubusercontent.com/onnx/onnx/master/onnx/onnx.proto3
        ${CMAKE_BINARY_DIR}/onnx.proto)

    INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
    PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER ${CMAKE_BINARY_DIR}/onnx.proto)
    ADD_LIBRARY(onnx_proto_lib ${PROTO_HEADER} ${PROTO_SRC})
    target_compile_options(onnx_proto_lib PUBLIC -std=c++0x -fPIC)

    get_filename_component("PROTO_HEADER_DIR" ${PROTO_HEADER} DIRECTORY)
    target_include_directories(n2d2_lib PUBLIC ${PROTO_HEADER_DIR})
    target_link_libraries(n2d2_lib PUBLIC onnx_proto_lib ${PROTOBUF_LIBRARY})
    target_compile_definitions(n2d2_lib PUBLIC ONNX=1)
endif()

if(MSVC)
    if (EXISTS "$ENV{DIRENT_INCLUDE_DIR}")
        set(CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES} $ENV{DIRENT_INCLUDE_DIR}")
        target_include_directories(n2d2_lib SYSTEM PUBLIC $ENV{DIRENT_INCLUDE_DIR})
    endif()

    include(CheckIncludeFile)
    check_include_file(dirent.h HAVE_DIRENT_H)

    if (NOT HAVE_DIRENT_H)
        message(FATAL_ERROR "dirent.h required - you can download it and "
                            "install it from http://www.softagalleria.net/dirent.php")
    endif()
endif()

# Python binding
# Requires PyBind >= 2.2, because previous versions had a bug with virtual 
# inherance: https://github.com/pybind/pybind11/issues/865
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG 2.2)
if(NOT pybind11_FOUND)
    find_package(pybind11 2.2)
endif()

if(pybind11_FOUND)
    message(STATUS "System pybind11 found.")
else()
    message(STATUS "Using third_party/pybind11.")
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/pybind11)
endif()

pybind11_add_module(N2D2 MODULE ${src_files_python})
message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
target_include_directories(N2D2 PUBLIC ${pybind11_INCLUDE_DIRS})
target_compile_definitions(N2D2 PUBLIC PYBIND=1)
target_link_libraries_whole_archive(N2D2 n2d2_lib)



set_target_properties(n2d2_lib PROPERTIES COTIRE_ADD_UNITY_BUILD FALSE)
set_target_properties(n2d2_lib PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "${CMAKE_CURRENT_LIST_DIR}/include/Precompiled.hpp")
cotire(n2d2_lib)


# n2d2_lib_cuda target
if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
    # For CMake < 3.9.0, use our own provided FindCUDA.cmake
    # in order to support "CUDA_LINK_LIBRARIES_KEYWORD"
    find_package(CUDA)
else()
    # For CMake >= 3.9.0, use the default FindCUDA provided by CMake
    # => in this case, we need to prevent find_package() to use our own.
    list(REMOVE_ITEM CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")
    find_package(CUDA)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")
endif()

if(CUDA_FOUND)
    set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
    find_package(CuDNN REQUIRED)

    file(GLOB_RECURSE src_files_cuda "src/*.cu")
    set_nvcc_flags()
    cuda_include_directories("${CMAKE_CURRENT_LIST_DIR}/include/")
    cuda_add_library(n2d2_lib_cuda STATIC ${src_files_cuda})

    # CUDA
    target_include_directories(n2d2_lib_cuda PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(n2d2_lib_cuda PUBLIC ${CUDA_LIBRARIES})

    # CuDNN
    target_include_directories(n2d2_lib_cuda PUBLIC ${CUDNN_INCLUDE_DIRS})
    target_link_libraries(n2d2_lib_cuda PUBLIC ${CUDNN_LIBRARY})

    # Cublas
    # Work-around due to a bug in CMake < 3.12.2
    if(CUDA_VERSION VERSION_GREATER 9.1 AND CMAKE_VERSION VERSION_LESS 3.12.2)
         list(REMOVE_ITEM CUDA_CUBLAS_LIBRARIES "CUDA_cublas_device_LIBRARY-NOTFOUND")
    endif()

    target_link_libraries(n2d2_lib_cuda PUBLIC ${CUDA_CUBLAS_LIBRARIES})

    # OpenCV
    find_package(OpenCV REQUIRED)
    target_link_libraries(n2d2_lib_cuda PUBLIC ${OpenCV_LIBS})

    # NVML
    find_package(NVML)
    if(NVML_FOUND)
        message(STATUS "NVML Version : ${NVML_API_VERSION}")
        target_include_directories(n2d2_lib_cuda PUBLIC ${NVML_INCLUDE_DIRS})
        target_link_libraries(n2d2_lib_cuda PUBLIC ${NVML_LIBRARIES})
        target_compile_definitions(n2d2_lib_cuda PUBLIC NVML=1)
    else()
        message(STATUS "NVML package not found")
    endif()

    target_compile_definitions(n2d2_lib_cuda PUBLIC CUDA=1)
    target_link_libraries(n2d2_lib PUBLIC n2d2_lib_cuda)
endif()

# The CMake is used directly and not through add_subdirectory
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # n2d2 target
    add_executable(n2d2 "exec/n2d2.cpp")
    target_link_libraries_whole_archive(n2d2 n2d2_lib)


    # exec target
    add_custom_target(exec)
    file(GLOB_RECURSE src_exec "exec/*.cpp")
    list(REMOVE_ITEM src_exec "${CMAKE_CURRENT_LIST_DIR}/exec/n2d2.cpp")
    foreach(file ${src_exec})
        add_n2d2_executable(${file} exec n2d2_lib)
    endforeach()


    # tests target
    enable_testing()
    add_custom_target(tests)
    add_custom_command(TARGET tests PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/tests/tests_data ${CMAKE_BINARY_DIR}/tests/tests_data)

    file(GLOB_RECURSE src_tests "tests/*.cpp")
    foreach(file ${src_tests})
        add_n2d2_test(${file} tests n2d2_lib)
    endforeach()
endif()

find_package(Doxygen)
if(Doxygen_FOUND)
    # Find all the public headers
    get_target_property(N2D2_PUBLIC_HEADER_DIR n2d2_lib INTERFACE_INCLUDE_DIRECTORIES)
    file(GLOB_RECURSE N2D2_PUBLIC_HEADERS ${N2D2_PUBLIC_HEADER_DIR}/*.hpp)

    set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR})
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_LIST_DIR}/docs/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
    set(DOXYFILE_IN ${CMAKE_CURRENT_LIST_DIR}/Doxyfile.in)
    set(DOXYFILE_OUT ${CMAKE_CURRENT_LIST_DIR}/Doxyfile)
    
    #Replace variables inside @@ with the current values
    configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                       DEPENDS ${CAT_CUTIFIER_PUBLIC_HEADERS}
                       COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                       MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                       COMMENT "Generating docs")

    add_custom_target(doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

    find_package(Sphinx)
    if(Sphinx_FOUND)
        set(SPHINX_SOURCE ${CMAKE_CURRENT_LIST_DIR}/docs/)
        set(SPHINX_BUILD ${CMAKE_CURRENT_LIST_DIR}/docs/_build/html)
        set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

        # Only regenerate Sphinx when:
        # - Doxygen has rerun
        # - Our doc files have been updated
        # - The Sphinx config has been updated
        add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                           COMMAND 
                             ${SPHINX_EXECUTABLE} -b html
                             # Tell Breathe where to find the Doxygen output
                             -Dbreathe_projects.N2D2=${DOXYGEN_OUTPUT_DIR}/xml
                           ${SPHINX_SOURCE} ${SPHINX_BUILD}
                           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                           DEPENDS
                           # Other docs files you want to track should go here (or in some variable)
                           ${SPHINX_SOURCE}/*
                           ${DOXYGEN_INDEX_FILE}
                           MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                           COMMENT "Generating documentation with Sphinx")
        
        # Nice named target so we can run the job easily
        add_custom_target(sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})
    endif()
endif()
