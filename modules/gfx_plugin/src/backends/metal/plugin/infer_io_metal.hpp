// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "plugin/infer_io_utils.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_metal(const ov::Tensor& host,
                                IGpuAllocator* allocator,
                                const char* error_prefix);

OutputBindingResult bind_host_output_metal(const GpuTensor& dev,
                                          const OutputViewInfo& info,
                                          const ov::Tensor* host_override,
                                          const ov::Tensor* reusable_host,
                                          IGpuAllocator* allocator,
                                          GpuBufferPool* pool,
                                          BufferHandle* staging_handle,
                                          GpuCommandQueueHandle metal_queue,
                                          const char* error_prefix);

}  // namespace gfx_plugin
}  // namespace ov
