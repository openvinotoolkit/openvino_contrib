// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_remote_context.hpp"

#include "openvino/core/except.hpp"

#include "plugin/gfx_remote_utils.hpp"
#include "backends/vulkan/plugin/vulkan_properties.hpp"
#include "runtime/memory_manager.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"

namespace ov {
namespace gfx_plugin {

RemoteTensorCreateResult create_vulkan_remote_tensor(const ov::element::Type& type,
                                                     const ov::Shape& shape,
                                                     const ov::AnyMap& params,
                                                     GpuDeviceHandle /*device*/,
                                                     size_t bytes) {
    RemoteTensorCreateResult result;
    GpuTensor tensor;
    tensor.shape = shape;
    tensor.expected_type = type;
    tensor.buf.type = type;
    tensor.buf.backend = GpuBackend::Vulkan;

    bool owns_buffer = false;
    void* external_buf = find_any_ptr(params,
                                      {kVkBufferProperty, kVulkanBufferProperty, kGfxBufferProperty});
    void* external_mem = find_any_ptr(params,
                                      {kVkMemoryProperty, kVulkanMemoryProperty, kGfxMemoryProperty});

    bool host_visible = find_any_bool(params, {kHostVisibleProperty, kGfxHostVisibleProperty}, false);

    if (external_buf) {
        tensor.buf.buffer = external_buf;
        tensor.buf.heap = external_mem;
        tensor.buf.size = bytes;
        tensor.buf.host_visible = host_visible;
        tensor.buf.external = true;
        tensor.buf.from_handle = true;
    } else {
        VulkanGpuAllocator allocator;
        GpuBufferDesc desc;
        desc.bytes = bytes;
        desc.type = type;
        desc.usage = BufferUsage::IO;
        desc.cpu_read = host_visible;
        desc.cpu_write = host_visible;
        desc.prefer_device_local = !host_visible;
        tensor.buf = allocator.allocate(desc);
        owns_buffer = true;
    }

    result.tensor = tensor;
    result.owns_buffer = owns_buffer;
    return result;
}

void release_vulkan_remote_tensor(GpuTensor& tensor, bool owns_buffer) {
    if (!owns_buffer || tensor.buf.backend != GpuBackend::Vulkan || !tensor.buf.buffer) {
        return;
    }
    vulkan_free_buffer(tensor.buf);
    tensor.buf.buffer = nullptr;
    tensor.buf.size = 0;
}

}  // namespace gfx_plugin
}  // namespace ov
