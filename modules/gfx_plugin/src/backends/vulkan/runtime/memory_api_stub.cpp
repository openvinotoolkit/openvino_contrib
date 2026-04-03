// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/memory_api.hpp"
#include "runtime/gpu_memory_ops.hpp"

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

void vulkan_copy_buffer(GpuCommandQueueHandle /*execution_context*/,
                        const GpuBuffer& src,
                        const GpuBuffer& dst,
                        size_t bytes) {
    if (!src.buffer || !dst.buffer || bytes == 0) {
        return;
    }
    throw_vulkan_unavailable();
}

void vulkan_copy_buffer_regions(GpuCommandQueueHandle /*execution_context*/,
                                const GpuBuffer& src,
                                const GpuBuffer& dst,
                                const GpuBufferCopyRegion* regions,
                                size_t region_count) {
    if (!src.buffer || !dst.buffer || !regions || region_count == 0) {
        return;
    }
    throw_vulkan_unavailable();
}

const GpuMemoryOps& vulkan_memory_ops() {
    static const GpuMemoryOps ops{
        /*map*/ [](const GpuBuffer& buf) -> void* { return vulkan_map_buffer(buf); },
        /*unmap*/ [](const GpuBuffer& buf) { vulkan_unmap_buffer(buf); },
        /*flush*/ [](const GpuBuffer& buf, size_t bytes, size_t offset) { vulkan_flush_buffer(buf, bytes, offset); },
        /*invalidate*/ [](const GpuBuffer& buf, size_t bytes, size_t offset) {
            vulkan_invalidate_buffer(buf, bytes, offset);
        },
        /*copy*/ [](GpuCommandQueueHandle execution_context,
                    const GpuBuffer& src,
                    const GpuBuffer& dst,
                    size_t bytes) { vulkan_copy_buffer(execution_context, src, dst, bytes); },
        /*copy_regions*/ [](GpuCommandQueueHandle execution_context,
                            const GpuBuffer& src,
                            const GpuBuffer& dst,
                            const GpuBufferCopyRegion* regions,
                            size_t region_count) {
            vulkan_copy_buffer_regions(execution_context, src, dst, regions, region_count);
        }};
    return ops;
}

void ensure_vulkan_memory_ops_registered() {
    static const bool registered = register_memory_ops(GpuBackend::Vulkan, &vulkan_memory_ops);
    (void)registered;
}

namespace {
const bool kRegistered = (ensure_vulkan_memory_ops_registered(), true);
}  // namespace

}  // namespace gfx_plugin
}  // namespace ov
