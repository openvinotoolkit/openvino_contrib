# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(GFX_MLIR_SOURCES
    mlir_builder.cpp
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
    mlir_builder_group_conv2d.cpp
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
    ../transforms/fusion_pass.cpp
    ../transforms/fusion_conv_activation.cpp
    ../transforms/fusion_conv_batchnorm.cpp
    ../transforms/fusion_conv_batchnorm_act.cpp
    ../transforms/fusion_eltwise_activation.cpp
    ../transforms/fusion_matmul_activation.cpp
    ../transforms/fusion_matmul_bias_activation.cpp
    ../transforms/fusion_matmul_bias.cpp
    ../transforms/fusion_conv_bias_activation.cpp
    ../transforms/fusion_conv_bias.cpp
    ../transforms/fusion_eltwise_bias_activation.cpp
    ../transforms/fusion_eltwise_bias.cpp
    ../transforms/fusion_attention.cpp
    ../transforms/fusion_attention_scale_mask.cpp
    ../transforms/conv_parallel_lowering.cpp
    ../transforms/parallel_fill_fusion.cpp
    mlir_passes.cpp
    mlir_utils.cpp)

add_library(gfx_mlir STATIC ${GFX_MLIR_SOURCES})

target_include_directories(gfx_mlir
    PUBLIC
        ${MLIR_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_link_libraries(gfx_mlir
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
        MLIRGPUDialect
        MLIRGPUTransforms
        MLIRSCFToGPU
        MLIRVectorDialect
        MLIRVectorInterfaces
        MLIRSubsetOpInterface
        MLIRSCFTransforms
        MLIRDLTIDialect
        MLIRReconcileUnrealizedCasts
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

if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
    add_dependencies(gfx_mlir ${GFX_LLVM_EXTERNAL_TARGET})
endif()

set_target_properties(gfx_mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
