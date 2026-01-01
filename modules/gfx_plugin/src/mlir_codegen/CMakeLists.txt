# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(GFX_MLIR_CODEGEN_SOURCES
    codegen_common.hpp
    msl_codegen.cpp
    spirv_codegen.cpp
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
    gather_codegen.cpp
    gathernd_codegen.cpp
    gather_elements_codegen.cpp
    depth_to_space_codegen.cpp
    space_to_depth_codegen.cpp
    scatter_elements_update_codegen.cpp
    scatter_nd_update_codegen.cpp
    shapeof_codegen.cpp
    convert_codegen.cpp
    unary_codegen.cpp
    select_codegen.cpp
    reduce_codegen.cpp
    pad_codegen.cpp
    tile_codegen.cpp
    broadcast_codegen.cpp
    range_codegen.cpp
    reverse_codegen.cpp
    topk_codegen.cpp)

add_library(gfx_mlir_codegen STATIC ${GFX_MLIR_CODEGEN_SOURCES})

target_include_directories(gfx_mlir_codegen
    PUBLIC
        ${MLIR_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_link_libraries(gfx_mlir_codegen
    PUBLIC
        openvino::runtime::dev
        MLIRIR
        MLIRSupport
        MLIRPass
        MLIRTransforms
        MLIRLinalgUtils
        MLIRAnalysis
        MLIRArithTransforms
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRTensorDialect
        MLIRArithDialect
        MLIRSCFDialect
        MLIRLinalgTransforms
        LLVMSupport
        LLVMDemangle)

if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
    add_dependencies(gfx_mlir_codegen ${GFX_LLVM_EXTERNAL_TARGET})
endif()

set_target_properties(gfx_mlir_codegen PROPERTIES POSITION_INDEPENDENT_CODE ON)
