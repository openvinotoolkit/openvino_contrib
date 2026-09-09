# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(ov_vox2seq LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Find OpenVINO via Python-reported CMake path ──────────────────────
execute_process(
    COMMAND python3 -c
        "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
    OUTPUT_VARIABLE OV_CMAKE_DIR
    RESULT_VARIABLE _ov_rc)
if(NOT _ov_rc EQUAL 0)
    message(FATAL_ERROR "Cannot locate OpenVINO cmake path via Python")
endif()
list(APPEND CMAKE_PREFIX_PATH "${OV_CMAKE_DIR}")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# ── Find OpenCL (optional — GPU dispatch) ─────────────────────────────
find_package(OpenCL QUIET)

# ── Build shared library ──────────────────────────────────────────────
add_library(ov_vox2seq SHARED
    vox2seq_op.cpp
    ov_extension.cpp)

target_link_libraries(ov_vox2seq PRIVATE openvino::runtime)

if(OpenCL_FOUND)
    target_link_libraries(ov_vox2seq PRIVATE OpenCL::OpenCL)
    target_compile_definitions(ov_vox2seq PRIVATE HAVE_OPENCL=1)
endif()

install(TARGETS ov_vox2seq LIBRARY DESTINATION lib)
