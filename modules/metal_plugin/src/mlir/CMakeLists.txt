# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(METAL_MLIR_SOURCES
    mlir_builder.cpp
    mlir_passes.cpp
    mlir_utils.cpp)

add_library(metal_mlir STATIC ${METAL_MLIR_SOURCES})

target_include_directories(metal_mlir
    PUBLIC
        ${MLIR_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(metal_mlir
    PUBLIC
        openvino::runtime::dev
        MLIRIR
        MLIRSupport
        MLIRPass
        MLIRTransforms
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRTensorDialect
        MLIRArithDialect
        LLVMSupport)

set_target_properties(metal_mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
