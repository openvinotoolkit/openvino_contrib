// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "plugin/infer_io_utils.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_vulkan(const ov::Tensor& host,
                                 GpuBufferPool* pool,
                                 BufferHandle* device_handle,
                                 BufferHandle* staging_handle,
                                 const char* error_prefix);

OutputBindingResult bind_host_output_vulkan(const GpuTensor& dev,
                                            const OutputViewInfo& info,
                                            const ov::Tensor* host_override,
                                            const ov::Tensor* reusable_host,
                                            GpuBufferPool* pool,
                                            BufferHandle* staging_handle,
                                            const char* error_prefix);

}  // namespace gfx_plugin
}  // namespace ov
