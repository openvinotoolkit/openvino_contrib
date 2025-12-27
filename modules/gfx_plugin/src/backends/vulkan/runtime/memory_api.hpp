// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

// Vulkan memory helpers that do not require Vulkan types in the signature.
void* vulkan_map_buffer(const GpuBuffer& buf);
void vulkan_unmap_buffer(const GpuBuffer& buf);
void vulkan_flush_buffer(const GpuBuffer& buf, size_t bytes, size_t offset = 0);
void vulkan_invalidate_buffer(const GpuBuffer& buf, size_t bytes, size_t offset = 0);
void vulkan_free_buffer(GpuBuffer& buf);
void vulkan_copy_buffer(const GpuBuffer& src, const GpuBuffer& dst, size_t bytes);

}  // namespace gfx_plugin
}  // namespace ov
