# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# CDPN OpenVINO Extension - top-level CMake.
#
#   cdpn_cpu/  - C++ extension .so (ov::Op subclasses with evaluate())
#   cdpn_gpu/  - OpenCL kernels (loaded via SimpleGPU XML at runtime)
#
# Only cdpn_cpu needs to be built. cdpn_gpu kernels are compiled by the
# GPU plugin at runtime from the .cl source files referenced in
# cdpn_custom_gpu_kernels.xml.
#
# Produces:
#   build/cdpn_extension.so

cmake_minimum_required(VERSION 3.13)
project(CdpnExtension)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set(SRC
    cdpn_cpu/src/cdpn_preprocess.cpp
    cdpn_cpu/src/cdpn_pnp_solve.cpp
    cdpn_cpu/src/ov_extension.cpp
)

add_library(cdpn_extension MODULE ${SRC})
set_target_properties(cdpn_extension PROPERTIES PREFIX "")
target_include_directories(cdpn_extension PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cdpn_cpu/include
)
target_link_libraries(cdpn_extension PRIVATE openvino::runtime)
