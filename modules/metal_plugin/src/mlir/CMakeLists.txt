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
    mlir_builder_prelu.cpp
    mlir_builder_minmax.cpp
    mlir_builder_logical.cpp
    mlir_builder_compare.cpp
    mlir_builder_select.cpp
    mlir_builder_reduce.cpp
    mlir_builder_pad.cpp
    mlir_builder_tile.cpp
    mlir_builder_broadcast.cpp
    mlir_builder_range.cpp
    mlir_builder_topk.cpp
    mlir_builder_reverse.cpp
    mlir_builder_batchnorm.cpp
    mlir_builder_conv2d.cpp
    mlir_builder_conv3d.cpp
    mlir_builder_maxpool.cpp
    mlir_builder_avgpool.cpp
    mlir_builder_softmax.cpp
    mlir_builder_convert.cpp
    mlir_builder_transpose.cpp
    mlir_builder_slice.cpp
    mlir_builder_concat.cpp
    mlir_builder_split.cpp
    mlir_builder_interpolate.cpp
    mlir_builder_gather.cpp
    mlir_builder_gathernd.cpp
    mlir_builder_gather_elements.cpp
    mlir_builder_depth_to_space.cpp
    mlir_builder_space_to_depth.cpp
    mlir_builder_scatter_elements_update.cpp
    mlir_builder_scatter_nd_update.cpp
    mlir_builder_shapeof.cpp
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
        MLIRLinalgUtils
        MLIRCallInterfaces
        MLIRValueBoundsOpInterface
        MLIRAnalysis
        MLIRArithTransforms
        MLIRComplexDialect
        MLIRMathDialect
        MLIRMathTransforms
        MLIRSparseTensorDialect
        MLIRSparseTensorTransforms
        MLIRTensorDialect
        MLIRArithDialect
        MLIRSCFDialect
        LLVMSupport
        LLVMDemangle)

if(DEFINED METAL_LLVM_EXTERNAL_TARGET)
    add_dependencies(metal_mlir ${METAL_LLVM_EXTERNAL_TARGET})
endif()

set_target_properties(metal_mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
