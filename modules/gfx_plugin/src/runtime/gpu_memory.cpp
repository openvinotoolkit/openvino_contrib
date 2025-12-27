// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gpu_memory.hpp"

#include <cstring>

#include "openvino/core/except.hpp"
#include "backends/metal/runtime/memory.hpp"
#include "backends/vulkan/runtime/memory_api.hpp"

namespace ov {
namespace gfx_plugin {

bool gpu_host_visible(const GpuBuffer& buf) {
    return buf.host_visible;
}

void* gpu_map_buffer(const GpuBuffer& buf) {
    if (!buf.buffer) {
        return nullptr;
    }
    switch (buf.backend) {
        case GpuBackend::Metal:
            return metal_map_buffer(buf);
        case GpuBackend::Vulkan:
            return vulkan_map_buffer(buf);
        default:
            return nullptr;
    }
}

void gpu_unmap_buffer(const GpuBuffer& buf) {
    if (!buf.buffer) {
        return;
    }
    switch (buf.backend) {
        case GpuBackend::Metal:
            metal_unmap_buffer(buf);
            return;
        case GpuBackend::Vulkan:
            vulkan_unmap_buffer(buf);
            return;
        default:
            return;
    }
}

void gpu_copy_from_host(GpuBuffer& buf, const void* src, size_t bytes, size_t offset) {
    OPENVINO_ASSERT(src || bytes == 0, "GFX: source pointer is null");
    OPENVINO_ASSERT(buf.buffer || bytes == 0, "GFX: destination buffer is null");
    OPENVINO_ASSERT(offset + bytes <= buf.size, "GFX: copy_from_host out of bounds");
    if (bytes == 0) {
        return;
    }
    if (!gpu_host_visible(buf)) {
        OPENVINO_THROW("GFX: buffer is not host-visible for copy_from_host");
    }
    auto* mapped = static_cast<uint8_t*>(gpu_map_buffer(buf));
    OPENVINO_ASSERT(mapped, "GFX: failed to map buffer for host copy");
    std::memcpy(mapped + offset, src, bytes);
    if (buf.backend == GpuBackend::Vulkan) {
        vulkan_flush_buffer(buf, bytes, offset);
    }
    gpu_unmap_buffer(buf);
}

void gpu_copy_to_host(const GpuBuffer& buf, void* dst, size_t bytes, size_t offset) {
    OPENVINO_ASSERT(dst || bytes == 0, "GFX: destination pointer is null");
    OPENVINO_ASSERT(buf.buffer || bytes == 0, "GFX: source buffer is null");
    OPENVINO_ASSERT(offset + bytes <= buf.size, "GFX: copy_to_host out of bounds");
    if (bytes == 0) {
        return;
    }
    if (!gpu_host_visible(buf)) {
        OPENVINO_THROW("GFX: buffer is not host-visible for copy_to_host");
    }
    auto* mapped = static_cast<const uint8_t*>(gpu_map_buffer(buf));
    OPENVINO_ASSERT(mapped, "GFX: failed to map buffer for host copy");
    if (buf.backend == GpuBackend::Vulkan) {
        vulkan_invalidate_buffer(buf, bytes, offset);
    }
    std::memcpy(dst, mapped + offset, bytes);
    gpu_unmap_buffer(buf);
}

void gpu_copy_buffer(GpuCommandQueueHandle queue,
                     const GpuBuffer& src,
                     const GpuBuffer& dst,
                     size_t bytes) {
    if (!src.buffer || !dst.buffer || bytes == 0) {
        return;
    }
    OPENVINO_ASSERT(src.backend == dst.backend, "GFX: cannot copy between different backends");
    switch (src.backend) {
        case GpuBackend::Metal:
            metal_copy_buffer(queue, src, dst, bytes);
            return;
        case GpuBackend::Vulkan:
            vulkan_copy_buffer(src, dst, bytes);
            return;
        default:
            return;
    }
}

}  // namespace gfx_plugin
}  // namespace ov
