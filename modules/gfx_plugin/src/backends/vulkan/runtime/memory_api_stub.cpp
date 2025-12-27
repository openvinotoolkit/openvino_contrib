// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/memory_api.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
[[noreturn]] void throw_vulkan_unavailable() {
    OPENVINO_THROW("GFX: Vulkan backend is not available in this build");
}
}  // namespace

void* vulkan_map_buffer(const GpuBuffer& buf) {
    if (!buf.buffer) {
        return nullptr;
    }
    throw_vulkan_unavailable();
}

void vulkan_unmap_buffer(const GpuBuffer& buf) {
    if (!buf.buffer) {
        return;
    }
    throw_vulkan_unavailable();
}

void vulkan_flush_buffer(const GpuBuffer& buf, size_t bytes, size_t /*offset*/) {
    if (!buf.buffer || bytes == 0) {
        return;
    }
    throw_vulkan_unavailable();
}

void vulkan_invalidate_buffer(const GpuBuffer& buf, size_t bytes, size_t /*offset*/) {
    if (!buf.buffer || bytes == 0) {
        return;
    }
    throw_vulkan_unavailable();
}

void vulkan_free_buffer(GpuBuffer& buf) {
    if (!buf.buffer) {
        return;
    }
    throw_vulkan_unavailable();
}

void vulkan_copy_buffer(const GpuBuffer& src, const GpuBuffer& dst, size_t bytes) {
    if (!src.buffer || !dst.buffer || bytes == 0) {
        return;
    }
    throw_vulkan_unavailable();
}

}  // namespace gfx_plugin
}  // namespace ov
