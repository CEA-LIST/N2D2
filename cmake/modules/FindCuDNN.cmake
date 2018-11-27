# Set CUDNN_FOUND, CUDNN_INCLUDE_DIRS, CUDNN_LIBRARY, CUDNN_VERSION_MAJOR, CUDNN_VERSION_MINOR, CUDNN_VERSION_PATCH and CUDNN_VERSION.
include(FindPackageHandleStandardArgs)

find_path(CUDNN_INCLUDE_DIRS cudnn.h HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES include)
find_library(CUDNN_LIBRARY NAMES cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64 lib/x64)

find_package_handle_standard_args(CuDNN DEFAULT_MSG CUDNN_INCLUDE_DIRS CUDNN_LIBRARY)
    
if (CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARY)
    file(READ ${CUDNN_INCLUDE_DIRS}/cudnn.h CUDNN_FILE_CONTENTS)

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
    message(STATUS "    include path: ${CUDNN_INCLUDE_DIRS}")
    message(STATUS "    libraries: ${CUDNN_LIBRARY}")
endif()