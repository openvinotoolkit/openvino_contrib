cmake_minimum_required(VERSION 3.16)
project(openvino_bevpool_extension)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenVINO via Python
if(NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c
                    "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
            OUTPUT_VARIABLE OpenVINO_DIR_PY
            ERROR_QUIET)
        if(OpenVINO_DIR_PY)
            message(STATUS "Found OpenVINO via Python: ${OpenVINO_DIR_PY}")
        endif()
    endif()
endif()

find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}" "${OpenVINO_DIR}")
message(STATUS "OpenVINO version: ${OpenVINO_VERSION}")

set(TARGET_NAME "openvino_bevpool_extension")
add_library(${TARGET_NAME} MODULE bev_pool_op.cpp ov_extension.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib RUNTIME DESTINATION lib)
install(FILES bev_pool_gpu.cl bev_pool_gpu.xml DESTINATION lib)
