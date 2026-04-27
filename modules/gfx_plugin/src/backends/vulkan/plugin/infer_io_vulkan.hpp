// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "plugin/infer_io_utils.hpp"
#include "runtime/gfx_profiler.hpp"

namespace ov {
namespace gfx_plugin {

struct VulkanOutputBindingResult {
    OutputBindingResult binding;
    GpuBuffer readback_buffer;
    size_t readback_bytes = 0;
};

GpuTensor bind_host_input_vulkan(const ov::Tensor& host,
                                 GpuBufferPool* pool,
                                 BufferHandle* device_handle,
                                 BufferHandle* staging_handle,
                                 GpuCommandBufferHandle command_buffer,
                                 GfxProfiler* profiler,
                                 const char* error_prefix);

VulkanOutputBindingResult prepare_host_output_vulkan(const GpuTensor& dev,
                                                     const OutputViewInfo& info,
                                                     const ov::Tensor* host_override,
                                                     ov::Tensor* reusable_host,
                                                     GpuBufferPool* pool,
                                                     BufferHandle* staging_handle,
                                                     GpuCommandBufferHandle command_buffer,
                                                     GfxProfiler* profiler,
                                                     const char* error_prefix);

void finalize_host_output_vulkan(VulkanOutputBindingResult& result,
                                 GfxProfiler* profiler,
                                 const char* error_prefix);

}  // namespace gfx_plugin
}  // namespace ov
