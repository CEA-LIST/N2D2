# Version 1.2
# Public Domain
# Written by Maxime SCHMITT <maxime.schmitt@etu.unistra.fr>
# Adapted by Vincent TEMPLIER <vincent.templier@cea.fr>

#/////////////////////////////////////////////////////////////////////////////#
#                                                                             #
# Search for Nvidia nvml library on the system                                #
# Call with find_package(NVML)                                                #
# The module defines:                                                         #
#   - NVML_FOUND        - If NVML was found                                   #
#   - NVML_INCLUDE_DIRS - the NVML include directories                        #
#   - NVML_LIBRARIES    - the NVML library directories                        #
#   - NVML_API_VERSION  - the NVML api version                                #
#                                                                             #
#/////////////////////////////////////////////////////////////////////////////#

if (NVML_INCLUDE_DIRS AND NVML_LIBRARIES)
  set(NVML_FIND_QUIETLY TRUE)
endif()

# windows, including both 32-bit and 64-bit
if(WIN32)

    if(${CUDA_VERSION_STRING} VERSION_LESS "8.0")
        file(GLOB nvml_header_path_hint "C:/Program Files/NVIDIA Corporation/GDK/nvml/include")
        file(GLOB nvml_lib_path_hint "C:/Program Files/NVIDIA Corporation/GDK/nvml/lib")
    else()
        file(GLOB nvml_header_path_hint ${CUDA_INCLUDE_DIRS})
        file(GLOB nvml_lib_path_hint "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
    endif()

    # Headers
    find_path(NVML_INCLUDE_DIRS NAMES nvml.h
    PATHS ${nvml_header_path_hint} ${PROJECT_BINARY_DIR}/include)

    # library
    find_library(NVML_LIBRARIES NAMES nvml
    PATHS ${nvml_lib_path_hint})

# linux
elseif(UNIX AND NOT APPLE)

    # Headers
    file(GLOB nvml_header_path_hint "${CUDA_INCLUDE_DIRS}" "${CUDA_INCLUDE_DIRS}/targets/*/include" "${CUDA_TOOLKIT_ROOT_DIR}/include")
    find_path(NVML_INCLUDE_DIRS NAMES nvml.h
    PATHS ${nvml_header_path_hint} ${PROJECT_BINARY_DIR}/include)

    # library
    if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8") # 64bit
        file(GLOB nvml_lib_path_hint "${CUDA_INCLUDE_DIRS}/targets/*/lib/stubs/" "${CUDA_TOOLKIT_ROOT_DIR}/lib64/"
            "${CUDA_TOOLKIT_ROOT_DIR}/lib/")
    else() # assume 32bit
        file(GLOB nvml_lib_path_hint "${CUDA_INCLUDE_DIRS}/targets/*/lib/stubs/" "${CUDA_TOOLKIT_ROOT_DIR}/lib/")
    endif()

    find_library(NVML_LIBRARIES NAMES nvidia-ml libnvidia-ml.so.1
    PATHS ${nvml_lib_path_hint})

else()
    message(STATUS "Unsupported platform for NVML library")

endif()

# Version
set(filename "${NVML_INCLUDE_DIRS}/nvml.h")
if (EXISTS ${filename})
  file(READ "${filename}" nvml_header)
  set(nvml_api_version_match "NVML_API_VERSION")

  string(REGEX REPLACE ".*#[ \t]*define[ \t]*${nvml_api_version_match}[ \t]*([0-9]+).*"
    "\\1" nvml_api_version "${nvml_header}")

  if (nvml_api_version STREQUAL nvml_header AND NOT quiet)
    message(AUTHOR_WARNING "Unable to find nvml api version")
  else()
    set(NVML_API_VERSION   "${nvml_api_version}")
  endif()
endif(EXISTS ${filename})

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(NVML
  FOUND_VAR NVML_FOUND
  REQUIRED_VARS NVML_INCLUDE_DIRS NVML_LIBRARIES
  VERSION_VAR NVML_API_VERSION)

mark_as_advanced(NVML_INCLUDE_DIRS NVML_LIBRARIES NVML_API_VERSION)