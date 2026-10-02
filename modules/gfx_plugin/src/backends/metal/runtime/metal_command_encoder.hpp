// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <vector>

#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

GpuCommandEncoderHandle metal_get_or_create_compute_encoder(GpuCommandBufferHandle command_buffer,
                                                            bool* created = nullptr);
bool metal_set_compute_pipeline_if_needed(GpuCommandBufferHandle command_buffer,
                                          GpuCommandEncoderHandle encoder,
                                          void* pipeline);
size_t metal_bind_compute_buffers_if_needed(GpuCommandBufferHandle command_buffer,
                                            GpuCommandEncoderHandle encoder,
                                            const std::vector<void*>& buffers,
                                            const std::vector<size_t>& offsets);
void metal_end_compute_encoder(GpuCommandBufferHandle command_buffer);

}  // namespace gfx_plugin
}  // namespace ov
