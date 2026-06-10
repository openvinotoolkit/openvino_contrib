cmake_minimum_required(VERSION 3.16)
project(openvino_sparse_encoder_extension)

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

# Find OpenCL
find_package(OpenCL QUIET)
if(NOT OpenCL_FOUND)
    # Manual fallback
    find_path(OpenCL_INCLUDE_DIR CL/cl.h PATHS /usr/include /usr/local/include)
    find_library(OpenCL_LIBRARY OpenCL PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
    if(OpenCL_INCLUDE_DIR AND OpenCL_LIBRARY)
        set(OpenCL_FOUND TRUE)
        message(STATUS "Found OpenCL manually: ${OpenCL_LIBRARY}")
    endif()
endif()

if(NOT OpenCL_FOUND)
    message(FATAL_ERROR "OpenCL not found — required for sparse encoder GPU kernels")
endif()

# Find OpenMP
find_package(OpenMP)

set(TARGET_NAME "openvino_sparse_encoder_extension")
add_library(${TARGET_NAME} MODULE sparse_encoder_op.cpp ov_extension.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime ${OpenCL_LIBRARY})
target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_INCLUDE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for neighbor map construction")
endif()

target_compile_options(${TARGET_NAME} PRIVATE -O3 -march=native)

install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib RUNTIME DESTINATION lib)
install(FILES sparse_conv.cl DESTINATION lib)
