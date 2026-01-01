// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/plugin/remote_tensor.hpp"

#include "openvino/core/except.hpp"

#include "plugin/gfx_remote_utils.hpp"
#include "backends/vulkan/plugin/vulkan_properties.hpp"
#include "runtime/memory_manager.hpp"
#include "backends/vulkan/runtime/gpu_memory.hpp"
#include "backends/vulkan/runtime/vulkan_memory.hpp"

namespace ov {
namespace gfx_plugin {

static void release_vulkan_remote_tensor(GpuTensor& tensor) {
    if (!tensor.buf.owned || tensor.buf.backend != GpuBackend::Vulkan || !tensor.buf.buffer) {
        return;
    }
    vulkan_free_buffer(tensor.buf);
    tensor.buf.buffer = nullptr;
    tensor.buf.size = 0;
}

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

    void* external_buf = find_any_ptr(params,
                                      {kVkBufferProperty, kVulkanBufferProperty, kGfxBufferProperty});
    void* external_mem = find_any_ptr(params,
                                      {kVkMemoryProperty, kVulkanMemoryProperty, kGfxMemoryProperty});

    const size_t declared_bytes = find_any_size(params, {kGfxBufferBytesProperty}, 0);

    bool host_visible = find_any_bool(params, {kHostVisibleProperty, kGfxHostVisibleProperty}, false);

    if (external_buf) {
        OPENVINO_ASSERT(declared_bytes > 0,
                        "GFX Vulkan: remote buffer size must be provided via ",
                        kGfxBufferBytesProperty);
        OPENVINO_ASSERT(declared_bytes >= bytes,
                        "GFX Vulkan: remote buffer is smaller than required (",
                        declared_bytes,
                        " < ",
                        bytes,
                        ")");
        tensor.buf.buffer = external_buf;
        tensor.buf.heap = external_mem;
        tensor.buf.size = declared_bytes;
        tensor.buf.host_visible = host_visible;
        tensor.buf.external = true;
        tensor.buf.from_handle = true;
        tensor.buf.owned = false;
    } else {
        if (declared_bytes) {
            OPENVINO_ASSERT(declared_bytes == bytes,
                            "GFX Vulkan: remote tensor bytes mismatch (declared ",
                            declared_bytes,
                            ", required ",
                            bytes,
                            ")");
        }
        VulkanGpuAllocator allocator;
        GpuBufferDesc desc;
        desc.bytes = bytes;
        desc.type = type;
        desc.usage = BufferUsage::IO;
        desc.cpu_read = host_visible;
        desc.cpu_write = host_visible;
        desc.prefer_device_local = !host_visible;
        tensor.buf = allocator.allocate(desc);
        tensor.buf.owned = true;
    }

    result.tensor = tensor;
    result.properties[kGfxBufferProperty] = tensor.buf.buffer;
    result.properties[kGfxMemoryProperty] = tensor.buf.heap;
    result.properties[kGfxBufferBytesProperty] = tensor.buf.size;
    result.properties[kGfxHostVisibleProperty] = tensor.buf.host_visible;
    result.properties[kVkBufferProperty] = tensor.buf.buffer;
    result.properties[kVulkanBufferProperty] = tensor.buf.buffer;
    result.properties[kVkMemoryProperty] = tensor.buf.heap;
    result.properties[kVulkanMemoryProperty] = tensor.buf.heap;
    result.properties[kHostVisibleProperty] = tensor.buf.host_visible;
    result.release_fn = &release_vulkan_remote_tensor;
    return result;
}

}  // namespace gfx_plugin
}  // namespace ov
