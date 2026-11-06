# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Top-level CMake: PointPillars OpenVINO Extension.
#
# pp_cpu/     contains the C++ extension.
# pp_gpu/     kernels are compiled by the GPU plugin at runtime from
#             pp_custom_gpu_kernels.xml.
# pp_runtime/ contains the optional OpenCL runtime library.
#
# Produces:
#   build/pp_extension.so
#   build/libpp_runtime.so (when OpenCL is available)

cmake_minimum_required(VERSION 3.13)
project(PointPillarsExtension)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# ov_export.py writes the generated configuration used by the C++ and OpenCL code.

set(EXTENSION_SOURCES
    pp_cpu/src/pp_voxelization.cpp
    pp_cpu/src/pp_scatter.cpp
    pp_cpu/src/pp_postprocess.cpp
    pp_cpu/src/ov_extension.cpp
)

add_library(pp_extension MODULE ${EXTENSION_SOURCES})
set_target_properties(pp_extension PROPERTIES PREFIX "")
target_include_directories(pp_extension PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/pp_cpu/include
)
target_link_libraries(pp_extension PRIVATE openvino::runtime)

# The runtime is built when OpenCL is available. It launches the kernels on the 
# same queue as the network, so the tensors between them stay in device memory.
find_package(OpenCL)
if(OpenCL_FOUND)
    add_library(pp_runtime SHARED pp_runtime/src/pp_runtime.cpp)
    target_include_directories(pp_runtime PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/pp_runtime/include
        ${CMAKE_CURRENT_SOURCE_DIR}/pp_cpu/include
    )
    # Configure the OpenCL headers used by the runtime.
    target_compile_definitions(pp_runtime PUBLIC
        CL_TARGET_OPENCL_VERSION=300
        OV_GPU_USE_OPENCL_HPP
    )
    target_link_libraries(pp_runtime PUBLIC openvino::runtime OpenCL::OpenCL)

    # OpenVINO's Python-package libraries must be discoverable at runtime.
    if(PP_RUNTIME_RPATH)
        set_target_properties(pp_runtime PROPERTIES
            BUILD_RPATH "${PP_RUNTIME_RPATH}" INSTALL_RPATH "${PP_RUNTIME_RPATH}")
        target_link_options(pp_runtime PRIVATE -Wl,--disable-new-dtags)
    endif()
endif()
