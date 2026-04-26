# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(GFX_MLIR_SOURCES
    mlir_builder.cpp
    rms_codegen.cpp
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
    mlir_builder_rms.cpp
    mlir_builder_pad.cpp
    mlir_builder_tile.cpp
    mlir_builder_broadcast.cpp
    mlir_builder_range.cpp
    mlir_builder_topk.cpp
    mlir_builder_reverse.cpp
    mlir_builder_batchnorm.cpp
    mlir_builder_conv2d.cpp
    mlir_builder_conv2d_vulkan.cpp
    mlir_builder_group_conv2d.cpp
    mlir_builder_group_conv2d_vulkan.cpp
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
    mlir_builder_scatter_update.cpp
    mlir_builder_scatter_nd_update.cpp
    mlir_builder_shapeof.cpp
    ../transforms/fusion_pass.cpp
    ../transforms/fusion_conv_activation.cpp
    ../transforms/fusion_conv_swish.cpp
    ../transforms/fusion_conv_batchnorm.cpp
    ../transforms/fusion_conv_batchnorm_act.cpp
    ../transforms/fusion_conv_batchnorm_swish.cpp
    ../transforms/fusion_eltwise_activation.cpp
    ../transforms/fusion_matmul_activation.cpp
    ../transforms/fusion_matmul_swish.cpp
    ../transforms/fusion_matmul_bias_activation.cpp
    ../transforms/fusion_matmul_bias_swish.cpp
    ../transforms/fusion_matmul_bias.cpp
    ../transforms/fusion_conv_bias_activation.cpp
    ../transforms/fusion_conv_bias_swish.cpp
    ../transforms/fusion_conv_bias.cpp
    ../transforms/fusion_conv_scale_activation.cpp
    ../transforms/fusion_conv_scale.cpp
    ../transforms/fusion_eltwise_bias_activation.cpp
    ../transforms/fusion_eltwise_bias.cpp
    ../transforms/mlir_fused_ops.cpp
    ../transforms/fusion_attention.cpp
    ../transforms/fusion_attention_scale_mask.cpp
    ../transforms/conv_im2col_matmul_rewrite.cpp
    ../transforms/conv_im2col_parallel_lowering.cpp
    ../transforms/conv_parallel_lowering.cpp
    ../transforms/conv3d_parallel_lowering.cpp
    ../transforms/matmul_parallel_lowering.cpp
    ../transforms/parallel_fill_fusion.cpp
    ../transforms/parallel_post_fusion.cpp
    gfx_mlir_transform_pipeline.cpp
    mlir_passes.cpp
    mlir_utils.cpp)

function(_gfx_resolve_mlir_libs out_var)
    set(_gfx_resolved_libs "")
    set(_gfx_mlir_lib_dir "")
    if(DEFINED MLIR_DIR)
        get_filename_component(_gfx_mlir_cmake_dir "${MLIR_DIR}" DIRECTORY)
        get_filename_component(_gfx_mlir_lib_dir "${_gfx_mlir_cmake_dir}" DIRECTORY)
    endif()
    foreach(_gfx_mlir_lib IN LISTS ARGN)
        if(TARGET ${_gfx_mlir_lib})
            list(APPEND _gfx_resolved_libs ${_gfx_mlir_lib})
        elseif(_gfx_mlir_lib_dir)
            set(_gfx_mlir_lib_path "${_gfx_mlir_lib_dir}/lib${_gfx_mlir_lib}.a")
            if(EXISTS "${_gfx_mlir_lib_path}")
                list(APPEND _gfx_resolved_libs "${_gfx_mlir_lib_path}")
            else()
                list(APPEND _gfx_resolved_libs ${_gfx_mlir_lib})
            endif()
            unset(_gfx_mlir_lib_path)
        else()
            list(APPEND _gfx_resolved_libs ${_gfx_mlir_lib})
        endif()
    endforeach()
    set(${out_var} "${_gfx_resolved_libs}" PARENT_SCOPE)
    unset(_gfx_mlir_lib)
    unset(_gfx_mlir_cmake_dir)
    unset(_gfx_mlir_lib_dir)
    unset(_gfx_resolved_libs)
endfunction()

set(GFX_COMMON_MLIR_LIBS
    ${GFX_MLIR_COMPONENT_LIBS}
    ${GFX_LLVM_COMPONENT_LIBS})
_gfx_resolve_mlir_libs(GFX_COMMON_MLIR_LINK_LIBS ${GFX_COMMON_MLIR_LIBS})
if(UNIX AND NOT APPLE AND NOT (DEFINED LLVM_FOUND AND LLVM_FOUND AND DEFINED MLIR_FOUND AND MLIR_FOUND))
    list(JOIN GFX_COMMON_MLIR_LINK_LIBS "," GFX_COMMON_MLIR_LIBS_JOINED)
    set(GFX_COMMON_MLIR_LINK_LIBS "$<LINK_GROUP:RESCAN,${GFX_COMMON_MLIR_LIBS_JOINED}>")
endif()

add_library(gfx_common_mlir_deps INTERFACE)
target_link_libraries(gfx_common_mlir_deps INTERFACE ${GFX_COMMON_MLIR_LINK_LIBS})

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
        gfx_common_mlir_deps)

if(DEFINED GFX_LLVM_EXTERNAL_TARGET)
    add_dependencies(gfx_mlir ${GFX_LLVM_EXTERNAL_TARGET})
endif()

set_target_properties(gfx_mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
