SET(EXEC "")
GET_DIRECTORIES(exec_dirs ./*.cpp)
FOREACH(exec_dir ${exec_dirs})
    AUX_SOURCE_DIRECTORY(${exec_dir} EXEC)

    GET_FILENAME_COMPONENT(exec_bin ${exec_dir} NAME_WE)
    FILE(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/${exec_bin})
ENDFOREACH()

foreach(f ${EXEC})
    GET_FILENAME_COMPONENT(bin ${f} NAME_WE)
    ADD_EXECUTABLE(${bin} ${f})
    TARGET_LINK_LIBRARIES(${bin} -Wl,-whole-archive N2D2 -Wl,-no-whole-archive)
endforeach(f)

