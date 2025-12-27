# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include_guard(GLOBAL)

set(_gfx_src_dir "${CMAKE_CURRENT_LIST_DIR}/../src")

set(GFX_PLUGIN_SOURCES
    ${_gfx_src_dir}/plugin/compiled_model.cpp
    ${_gfx_src_dir}/plugin/gfx_device_info.cpp
    ${_gfx_src_dir}/plugin/infer_request.mm
    ${_gfx_src_dir}/plugin/plugin.cpp
    ${_gfx_src_dir}/plugin/remote_context_support.cpp
    ${_gfx_src_dir}/plugin/remote_stub.mm
)

set(GFX_PLUGIN_HEADERS
    ${_gfx_src_dir}/plugin/compiled_model.hpp
    ${_gfx_src_dir}/plugin/gfx_device_info.hpp
    ${_gfx_src_dir}/plugin/gfx_profiling_utils.hpp
    ${_gfx_src_dir}/plugin/gfx_remote_properties.hpp
    ${_gfx_src_dir}/plugin/gfx_remote_utils.hpp
    ${_gfx_src_dir}/plugin/gfx_property_utils.hpp
    ${_gfx_src_dir}/plugin/infer_request.hpp
    ${_gfx_src_dir}/plugin/infer_pipeline.hpp
    ${_gfx_src_dir}/backends/metal/plugin/properties.hpp
    ${_gfx_src_dir}/plugin/plugin.hpp
    ${_gfx_src_dir}/plugin/remote_context_support.hpp
    ${_gfx_src_dir}/plugin/remote_stub.hpp
    ${_gfx_src_dir}/transforms/conv_relu_fusion.hpp
    ${_gfx_src_dir}/transforms/pipeline.hpp
)

set(GFX_RUNTIME_COMMON_HEADERS
    ${_gfx_src_dir}/runtime/gfx_activation.hpp
    ${_gfx_src_dir}/runtime/gfx_backend_caps.hpp
    ${_gfx_src_dir}/runtime/gfx_backend_utils.hpp
    ${_gfx_src_dir}/runtime/gfx_op_support.hpp
    ${_gfx_src_dir}/runtime/gpu_backend.hpp
    ${_gfx_src_dir}/compiler/gfx_codegen_backend.hpp
    ${_gfx_src_dir}/runtime/gpu_buffer.hpp
    ${_gfx_src_dir}/runtime/gpu_buffer_pool.hpp
    ${_gfx_src_dir}/runtime/gpu_stage.hpp
    ${_gfx_src_dir}/runtime/gpu_stage_factory.hpp
    ${_gfx_src_dir}/runtime/gpu_tensor.hpp
    ${_gfx_src_dir}/runtime/gpu_types.hpp
    ${_gfx_src_dir}/runtime/gfx_kernel_dispatch.hpp
    ${_gfx_src_dir}/compiler/gfx_kernel_plan.hpp
    ${_gfx_src_dir}/compiler/gfx_kernel_spec.hpp
    ${_gfx_src_dir}/compiler/mlir/gfx_mlir_kernel_builder.hpp
    ${_gfx_src_dir}/compiler/mlir_support.hpp
    ${_gfx_src_dir}/runtime/gfx_logger.hpp
    ${_gfx_src_dir}/runtime/gfx_op_utils.hpp
    ${_gfx_src_dir}/runtime/profiling/gfx_profiler_config.hpp
    ${_gfx_src_dir}/backends/vulkan/runtime/memory_api.hpp
)

set(GFX_RUNTIME_COMMON_SOURCES
    ${_gfx_src_dir}/runtime/gfx_backend_caps.cpp
    ${_gfx_src_dir}/runtime/gfx_op_support.cpp
    ${_gfx_src_dir}/runtime/gpu_stage_factory.cpp
    ${_gfx_src_dir}/runtime/gpu_memory.cpp
    ${_gfx_src_dir}/compiler/mlir/gfx_mlir_kernel_builder.cpp
    ${_gfx_src_dir}/compiler/mlir_support.cpp
    ${_gfx_src_dir}/runtime/gfx_logger.cpp
    ${_gfx_src_dir}/runtime/gfx_op_utils.cpp
)

set(GFX_RUNTIME_METAL_SOURCES
    ${_gfx_src_dir}/backends/metal/runtime/stage.cpp
    ${_gfx_src_dir}/backends/metal/runtime/gpu_memory.mm
    ${_gfx_src_dir}/backends/metal/codegen/kernel_compiler_common.mm
    ${_gfx_src_dir}/backends/metal/runtime/backend.mm
    ${_gfx_src_dir}/backends/metal/runtime/dtype.cpp
    ${_gfx_src_dir}/backends/metal/runtime/memory.mm
    ${_gfx_src_dir}/backends/metal/runtime/op.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_activations.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_batchnorm.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_broadcast.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_concat.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_conv.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_conv3d.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_convert.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_depth_to_space.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_elementwise.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_factory.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_gather.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_gather_elements.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_gathernd.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_group_conv.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_interpolate.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_matmul.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_pad.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_pooling.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_range.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_reduce.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_reshape.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_reverse.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_scatter_elements_update.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_scatter_nd_update.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_select.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_shapeof.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_slice.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_softmax.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_space_to_depth.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_split.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_tile.mm
    ${_gfx_src_dir}/backends/metal/runtime/op_topk.mm
    ${_gfx_src_dir}/backends/metal/memory/allocator.mm
    ${_gfx_src_dir}/backends/metal/memory/allocator_core.mm
    ${_gfx_src_dir}/backends/metal/memory/const_cache.mm
    ${_gfx_src_dir}/backends/metal/memory/device_caps.mm
    ${_gfx_src_dir}/backends/metal/memory/heap_pool.mm
    ${_gfx_src_dir}/backends/metal/memory/memory_session.mm
    ${_gfx_src_dir}/backends/metal/memory/staging_pool.mm
    ${_gfx_src_dir}/backends/metal/profiling/gpu_timestamps.mm
    ${_gfx_src_dir}/backends/metal/profiling/profiler.mm
    ${_gfx_src_dir}/backends/metal/profiling/profiling_report.cpp
)

set(GFX_RUNTIME_METAL_HEADERS
    ${_gfx_src_dir}/backends/metal/codegen/kernel_compiler.hpp
    ${_gfx_src_dir}/backends/metal/runtime/backend.hpp
    ${_gfx_src_dir}/backends/metal/runtime/dtype.hpp
    ${_gfx_src_dir}/backends/metal/runtime/logger.hpp
    ${_gfx_src_dir}/backends/metal/runtime/memory.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_activations.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_batchnorm.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_broadcast.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_concat.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_conv.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_conv3d.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_convert.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_depth_to_space.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_elementwise.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_factory.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_gather.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_gather_elements.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_gathernd.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_group_conv.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_interpolate.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_kinds.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_matmul.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_pad.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_pooling.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_range.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_reduce.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_reshape.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_reverse.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_scatter_elements_update.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_scatter_nd_update.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_select.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_shapeof.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_slice.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_softmax.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_space_to_depth.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_split.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_tile.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_topk.hpp
    ${_gfx_src_dir}/backends/metal/runtime/op_utils.hpp
    ${_gfx_src_dir}/backends/metal/memory/allocator.hpp
    ${_gfx_src_dir}/backends/metal/memory/allocator_core.hpp
    ${_gfx_src_dir}/backends/metal/memory/buffer.hpp
    ${_gfx_src_dir}/backends/metal/memory/const_cache.hpp
    ${_gfx_src_dir}/backends/metal/memory/device_caps.hpp
    ${_gfx_src_dir}/backends/metal/memory/freelist.hpp
    ${_gfx_src_dir}/backends/metal/memory/heap_pool.hpp
    ${_gfx_src_dir}/backends/metal/memory/memory_session.hpp
    ${_gfx_src_dir}/backends/metal/memory/memory_stats.hpp
    ${_gfx_src_dir}/backends/metal/memory/staging_pool.hpp
    ${_gfx_src_dir}/backends/metal/profiling/gpu_timestamps.hpp
    ${_gfx_src_dir}/backends/metal/profiling/profiler.hpp
    ${_gfx_src_dir}/backends/metal/profiling/profiler_config.hpp
    ${_gfx_src_dir}/backends/metal/profiling/profiling_report.hpp
    ${_gfx_src_dir}/backends/metal/runtime/stage.hpp
)

# TODO: add Vulkan backend sources once they are introduced.
set(GFX_RUNTIME_VULKAN_SOURCES
    ${_gfx_src_dir}/backends/vulkan/runtime/backend.cpp
    ${_gfx_src_dir}/backends/vulkan/runtime/stage.cpp
    ${_gfx_src_dir}/backends/vulkan/runtime/memory.cpp
    ${_gfx_src_dir}/backends/vulkan/runtime/gpu_memory.cpp
    ${_gfx_src_dir}/backends/vulkan/profiling/profiler.cpp
)

set(GFX_RUNTIME_VULKAN_HEADERS
    ${_gfx_src_dir}/backends/vulkan/runtime/backend.hpp
    ${_gfx_src_dir}/backends/vulkan/runtime/stage.hpp
    ${_gfx_src_dir}/backends/vulkan/runtime/memory.hpp
    ${_gfx_src_dir}/backends/vulkan/profiling/profiler.hpp
)

set(GFX_HAS_METAL_SOURCES OFF)
if(GFX_RUNTIME_METAL_SOURCES)
    set(GFX_HAS_METAL_SOURCES ON)
endif()

set(GFX_HAS_VULKAN_SOURCES OFF)
if(GFX_RUNTIME_VULKAN_SOURCES)
    set(GFX_HAS_VULKAN_SOURCES ON)
endif()

unset(_gfx_src_dir)
