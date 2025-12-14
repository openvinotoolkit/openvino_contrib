# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(METAL_MLIR_SOURCES
    mlir_builder_matmul.cpp
    mlir_builder_unary.cpp
    mlir_builder_add.cpp
    mlir_builder_sub.cpp
    mlir_builder_mul.cpp
    mlir_builder_div.cpp
    mlir_builder_pow.cpp
    mlir_builder_mod.cpp
    mlir_builder_floor_mod.cpp
    mlir_builder_batchnorm.cpp
    mlir_builder_conv2d.cpp
    mlir_builder_conv3d.cpp
    mlir_builder_concat.cpp
    mlir_builder_split.cpp
    mlir_builder_convert.cpp
    mlir_builder_interpolate.cpp
    mlir_builder_maxpool.cpp
    mlir_builder_avgpool.cpp
    mlir_builder_softmax.cpp
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
        MLIRBufferizationDialect
        MLIRBufferizationTransforms
        MLIRFuncTransforms
        MLIRMemRefTransforms
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRTensorDialect
        MLIRArithDialect
        MLIRSCFDialect
        LLVMSupport)

if(DEFINED METAL_LLVM_EXTERNAL_TARGET)
    add_dependencies(metal_mlir ${METAL_LLVM_EXTERNAL_TARGET})
endif()

set_target_properties(metal_mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
