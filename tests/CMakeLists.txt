set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)

SET(TESTS "")
GET_DIRECTORIES(tests_dirs ./*.cpp)
FOREACH(tests_dir ${tests_dirs})
    AUX_SOURCE_DIRECTORY(${tests_dir} TESTS)

    GET_FILENAME_COMPONENT(tests_bin ${tests_dir} NAME_WE)
    FILE(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/${tests_bin})
ENDFOREACH()

foreach(f ${TESTS})
    GET_FILENAME_COMPONENT(bin ${f} NAME_WE)
    ADD_EXECUTABLE(${bin} ${f})
    TARGET_LINK_LIBRARIES(${bin} -Wl,-whole-archive N2D2 -Wl,-no-whole-archive)

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

