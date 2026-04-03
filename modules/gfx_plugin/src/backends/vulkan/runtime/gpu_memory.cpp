// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/gpu_memory.hpp"

#include "openvino/core/except.hpp"
#include "backends/vulkan/runtime/memory_api.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"

namespace ov {
namespace gfx_plugin {

VulkanGpuAllocator::VulkanGpuAllocator(VkBufferUsageFlags usage) : m_usage(usage) {
    ensure_vulkan_memory_ops_registered();
}

GpuBuffer VulkanGpuAllocator::allocate(const GpuBufferDesc& desc) {
    validate_gpu_buffer_desc(desc, "GFX Vulkan");
    VkMemoryPropertyFlags props = 0;
    if (desc.cpu_read || desc.cpu_write) {
        props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else {
        props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    GpuBuffer buf = vulkan_allocate_buffer(desc.bytes, desc.type, m_usage, props);
    if (!buf.buffer && desc.bytes > 0) {
        OPENVINO_THROW("GFX Vulkan: failed to allocate buffer");
    }
    buf.backend = GpuBackend::Vulkan;
    buf.host_visible = (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    return buf;
}

GpuBuffer VulkanGpuAllocator::wrap_shared(void* /*ptr*/, size_t /*bytes*/, ov::element::Type /*type*/) {
    OPENVINO_THROW("GFX Vulkan: wrap_shared is not supported");
}

void VulkanGpuAllocator::release(GpuBuffer&& buf) {
    vulkan_free_buffer(buf);
}

GpuBuffer VulkanGpuAllocator::ensure_handle(BufferHandle& handle, const GpuBufferDesc& desc) {
    const size_t target = desc.bytes;
    const bool want_host_visible = desc.cpu_read || desc.cpu_write;
    if (handle.valid() && handle.capacity >= target && handle.buf.host_visible == want_host_visible) {
        handle.buf.from_handle = true;
        handle.buf.type = desc.type;
        return handle.buf;
    }

    if (handle.buf.valid()) {
        release(std::move(handle.buf));
    }

    GpuBuffer buf = allocate(desc);
    buf.from_handle = true;
    handle.buf = buf;
    handle.capacity = buf.size;
    return handle.buf;
}

}  // namespace gfx_plugin
}  // namespace ov
