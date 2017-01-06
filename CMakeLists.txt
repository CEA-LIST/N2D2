CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(N2D2)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

if (NOT CMAKE_BUILD_TYPE)
    MESSAGE(STATUS "No build type selected, default to Release")
    SET(CMAKE_BUILD_TYPE "Release")
endif()

# Find required & optional packages
if(MSVC)
    INCLUDE(CheckIncludeFile)
    CHECK_INCLUDE_FILE(dirent.h HAVE_DIRENT_H)

    if (NOT HAVE_DIRENT_H)
        MESSAGE(FATAL_ERROR "dirent.h required - you can download it and install it from http://www.softagalleria.net/dirent.php")
    endif()
endif()

FIND_PACKAGE(Gnuplot REQUIRED)

# Define environment variable OpenCV_DIR to point to for example "C:\OpenCV\opencv\build"
if (EXISTS "$ENV{OpenCV_DIR}")
    INCLUDE("$ENV{OpenCV_DIR}/OpenCVConfig.cmake")
endif()

FIND_PACKAGE(OpenCV 2.0.0 REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${OpenCV_INCLUDE_DIR})

FIND_PACKAGE(OpenMP)
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")

FIND_PACKAGE(CUDA)
if (CUDA_FOUND)
    INCLUDE_DIRECTORIES(SYSTEM ${CUDA_INCLUDE_DIRS})

    if (EXISTS "${CUDA_INCLUDE_DIRS}/cudnn.h")
        GET_FILENAME_COMPONENT(CUDNN_LIB_DIR ${CUDA_CUDART_LIBRARY} PATH)
        SET(CUDA_LIBS "cuda;cudart;cublas;cudadevrt")
        SET(CUDNN_LIBS "cudnn")

        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA=1")
        SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    else()
        MESSAGE(WARNING "CUDA found but CuDNN seems to be missing - you can download it and install it from http://www.nvidia.com")
    endif()
endif()

FIND_PACKAGE(PugiXML)
if (PUGIXML_FOUND)
    INCLUDE_DIRECTORIES(SYSTEM ${PUGIXML_INCLUDE_DIR})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPUGIXML=1")
endif()

FIND_PACKAGE(MongoDB)
if (MongoDB_FOUND)
    FIND_PACKAGE(Boost COMPONENTS thread system filesystem program_options REQUIRED)
    FIND_PACKAGE(OpenSSL REQUIRED)

    INCLUDE_DIRECTORIES(SYSTEM ${MongoDB_INCLUDE_DIR})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMONGODB=1")
endif()

# Compiler flags
if(MSVC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    # /wd4250 disable 'class1' : inherits 'class2::member' via dominance
    # /wd4512 disable 'class' : assignment operator could not be generated
    ADD_DEFINITIONS(-D_CONSOLE -D_VISUALC_ -DNeedFunctionPrototypes -D_CRT_SECURE_NO_WARNINGS -D_VARIADIC_MAX=10 /wd4250 /wd4512)
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror -std=c++0x -O3 -s -DNDEBUG")

    if(${OpenCV_VERSION} EQUAL "2.0.0")
        MESSAGE(WARNING "Compiling with _GLIBCXX_PARALLEL flag")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_PARALLEL")
    endif()
endif()

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

# N2D2_Core
GET_DIRECTORIES(headers_dirs include/*.h)
INCLUDE_DIRECTORIES(${headers_dirs})

SET(SRC "")
GET_DIRECTORIES(sources_dirs src/*.cpp)
FOREACH(source_dir ${sources_dirs})
    AUX_SOURCE_DIRECTORY(${source_dir} SRC)
ENDFOREACH()

LINK_DIRECTORIES(${OpenCV_LIB_DIR})

if (CUDA_FOUND)
    SET(CU_SRC "")
    FILE(GLOB_RECURSE CU_SRC "src/*.cu")

    if (NOT "${CU_SRC}" STREQUAL "")
        CUDA_ADD_LIBRARY(N2D2_Core STATIC ${SRC} ${CU_SRC})
    else()
        ADD_LIBRARY(N2D2_Core STATIC ${SRC})
    endif()

    LINK_DIRECTORIES(${CUDNN_LIB_DIR})
    TARGET_LINK_LIBRARIES(N2D2_Core ${CUDA_LIBS})
    TARGET_LINK_LIBRARIES(N2D2_Core ${CUDNN_LIBS})
else()
    ADD_LIBRARY(N2D2_Core STATIC ${SRC})
endif()

if (PUGIXML_FOUND)
    TARGET_LINK_LIBRARIES(N2D2_Core ${PUGIXML_LIBRARIES})
endif()

if (MongoDB_FOUND)
    TARGET_LINK_LIBRARIES(N2D2_Core ${MongoDB_LIBRARIES})
    TARGET_LINK_LIBRARIES(N2D2_Core ${Boost_THREAD_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${Boost_SYSTEM_LIBRARY} ${OPENSSL_LIBRARIES})
endif()

TARGET_LINK_LIBRARIES(N2D2_Core ${OpenCV_LIBS})

# Binaries directory (if you want them in the project file, use ${PROJECT_SOURCE_DIR})
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Simulations
AUX_SOURCE_DIRECTORY("simus" SIMUS)

foreach(f ${SIMUS})
    GET_FILENAME_COMPONENT(bin ${f} NAME_WE)

    ADD_EXECUTABLE(${bin} ${f})
    TARGET_LINK_LIBRARIES(${bin} -Wl,-whole-archive N2D2_Core -Wl,-no-whole-archive)
endforeach(f)

# Unit Tests
SET(UNIT_TESTS "")
GET_DIRECTORIES(unit_tests_dirs unit_tests/*.cpp)
FOREACH(unit_tests_dir ${unit_tests_dirs})
    AUX_SOURCE_DIRECTORY(${unit_tests_dir} UNIT_TESTS)

    GET_FILENAME_COMPONENT(unit_tests_bin ${unit_tests_dir} NAME_WE)
    FILE(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/${unit_tests_bin})
ENDFOREACH()

foreach(f ${UNIT_TESTS})
    GET_FILENAME_COMPONENT(bin ${f} NAME_WE)
    SET(bin "unit_tests_${bin}")

    ADD_EXECUTABLE(${bin} ${f})
    TARGET_LINK_LIBRARIES(${bin} -Wl,-whole-archive N2D2_Core -Wl,-no-whole-archive)

    ADD_CUSTOM_COMMAND(
        OUTPUT ${bin}.log
        COMMAND ${bin}
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    )
    ADD_CUSTOM_TARGET(
        run_${bin} ALL
        DEPENDS ${bin}.log
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    )
endforeach(f)

