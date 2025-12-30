// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "plugin/infer_io_utils.hpp"
#include "backends/metal/runtime/metal_memory.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_metal(const ov::Tensor& host,
                                MetalAllocatorCore* metal_core,
                                const char* error_prefix);

OutputBindingResult bind_host_output_metal(const GpuTensor& dev,
                                          const OutputViewInfo& info,
                                          const ov::Tensor* host_override,
                                          MetalAllocatorCore* metal_core,
                                          MetalAllocator* metal_allocator,
                                          GpuCommandQueueHandle metal_queue,
                                          const char* error_prefix);

}  // namespace gfx_plugin
}  // namespace ov
