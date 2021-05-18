CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(N2D2_export_C)

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

FIND_PACKAGE(OpenMP)
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")

# Compiler flags
if(MSVC)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    # /wd4250 disable 'class1' : inherits 'class2::member' via dominance
    # /wd4512 disable 'class' : assignment operator could not be generated
    ADD_DEFINITIONS(-D_CONSOLE -D_VISUALC_ -DNeedFunctionPrototypes -D_CRT_SECURE_NO_WARNINGS -D_VARIADIC_MAX=10 /wd4250 /wd4512)
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-label -pedantic -std=c99 -Wconversion -fsigned-char -O3 -s -DNDEBUG")
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

GET_DIRECTORIES(headers_dirs include/*.h)
INCLUDE_DIRECTORIES(${headers_dirs})

SET(SRC "")
GET_DIRECTORIES(sources_dirs src/*.c)
FOREACH(source_dir ${sources_dirs})
    AUX_SOURCE_DIRECTORY(${source_dir} SRC)
ENDFOREACH()

ADD_LIBRARY(N2D2_export_C_Core STATIC ${SRC})

# Binaries directory (if you want them in the project file, use ${PROJECT_SOURCE_DIR})
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Binaries
AUX_SOURCE_DIRECTORY("." BINARIES)

foreach(f ${BINARIES})
    GET_FILENAME_COMPONENT(bin ${f} NAME_WE)

    ADD_EXECUTABLE(${bin} ${f})
    TARGET_LINK_LIBRARIES(${bin} N2D2_export_C_Core)
endforeach(f)
