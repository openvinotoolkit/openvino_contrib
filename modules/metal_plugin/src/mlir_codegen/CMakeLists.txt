# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(METAL_MLIR_CODEGEN_SOURCES
    codegen_common.hpp
    codegen_dispatcher.cpp
    matmul_codegen.cpp
    conv2d_codegen.cpp
    conv3d_codegen.cpp
    conv2d_codegen.cpp
    eltwise_codegen.cpp
    pool_max_codegen.cpp
    pool_avg_codegen.cpp
    softmax_codegen.cpp
    batchnorm_codegen.cpp
    concat_codegen.cpp
    interpolate_codegen.cpp
    split_codegen.cpp
    transpose_codegen.cpp
    slice_generic_codegen.cpp
    convert_codegen.cpp
    unary_codegen.cpp)

add_library(metal_mlir_codegen STATIC ${METAL_MLIR_CODEGEN_SOURCES})

target_include_directories(metal_mlir_codegen
    PUBLIC
        ${MLIR_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(metal_mlir_codegen
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
        MLIRSCFDialect
        MLIRLinalgTransforms
        LLVMSupport)

if(DEFINED METAL_LLVM_EXTERNAL_TARGET)
    add_dependencies(metal_mlir_codegen ${METAL_LLVM_EXTERNAL_TARGET})
endif()

set_target_properties(metal_mlir_codegen PROPERTIES POSITION_INDEPENDENT_CODE ON)
