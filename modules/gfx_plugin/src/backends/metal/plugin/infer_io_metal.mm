// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/plugin/infer_io_metal.hpp"

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_metal(const ov::Tensor& host,
                                IGpuAllocator* allocator,
                                const char* error_prefix) {
    HostInputBinding binding = prepare_host_input_binding(host, GpuBackend::Metal, error_prefix);
    GpuTensor tensor = binding.tensor;
    const size_t bytes = binding.bytes;
    if (bytes == 0) {
        return tensor;
    }

    OPENVINO_ASSERT(allocator, error_prefix, ": GPU allocator is null");
    tensor.buf = allocator->wrap_shared(const_cast<void*>(host.data()), bytes, host.get_element_type());
    tensor.buf.backend = allocator->backend();
    tensor.buf.host_visible = true;
    tensor.prefer_private = false;
    return tensor;
}

OutputBindingResult bind_host_output_metal(const GpuTensor& dev,
                                          const OutputViewInfo& info,
                                          const ov::Tensor* host_override,
                                          const ov::Tensor* reusable_host,
                                          IGpuAllocator* allocator,
                                          GpuBufferPool* pool,
                                          BufferHandle* staging_handle,
                                          GpuCommandQueueHandle metal_queue,
                                          const char* error_prefix) {
    OutputBindingResult result{};
    result.device_tensor = dev;

    OPENVINO_ASSERT(allocator, error_prefix, ": GPU allocator is not available");
    OPENVINO_ASSERT(pool, error_prefix, ": GPU buffer pool is not available");
    OPENVINO_ASSERT(staging_handle, error_prefix, ": staging handle is not available");

    if (host_override && *host_override) {
        HostOutputBinding host_binding = prepare_host_output_binding(info, host_override, reusable_host);
        const size_t bytes = host_binding.bytes;
        pool->release(*staging_handle);
        GpuBuffer shared = allocator->wrap_shared(const_cast<void*>(host_override->data()), bytes, info.type);
        if (bytes && dev.buf.buffer != shared.buffer) {
            gpu_copy_buffer(metal_queue, dev.buf, shared, bytes);
        }
        result.device_tensor = dev;
        result.device_tensor.buf = shared;
        result.device_tensor.expected_type = info.type;
        result.device_tensor.shape = info.shape;
        result.device_tensor.prefer_private = false;
        result.host_tensor = host_binding.host;
        return result;
    }

    if (dev.buf.host_visible && dev.buf.buffer) {
        pool->release(*staging_handle);
        void* ptr = gpu_map_buffer(dev.buf);
        OPENVINO_ASSERT(ptr, error_prefix, ": shared output buffer has no CPU pointer");
        result.host_tensor = ov::Tensor(info.type, info.shape, ptr);
        result.device_tensor.expected_type = info.type;
        if (result.device_tensor.shape.empty()) {
            result.device_tensor.shape = info.shape;
        }
        result.device_tensor.prefer_private = false;
        return result;
    }

    HostOutputBinding host_binding = prepare_host_output_binding(info, host_override, reusable_host);
    size_t bytes = host_binding.bytes;
    if (bytes) {
        GpuBufferDesc desc;
        desc.bytes = bytes;
        desc.type = info.type;
        desc.usage = BufferUsage::IO;
        desc.cpu_read = true;
        desc.cpu_write = true;
        desc.prefer_device_local = false;
        GpuBuffer shared = pool->ensure(*staging_handle, desc);
        gpu_copy_buffer(metal_queue, dev.buf, shared, bytes);
        result.device_tensor = dev;
        result.device_tensor.buf = shared;
        result.device_tensor.expected_type = info.type;
        result.device_tensor.shape = info.shape;
        result.device_tensor.prefer_private = false;
        void* ptr = gpu_map_buffer(shared);
        OPENVINO_ASSERT(ptr, error_prefix, ": shared output buffer has no CPU pointer");
        result.host_tensor = ov::Tensor(info.type, info.shape, ptr);
        return result;
    }

    result.device_tensor.expected_type = info.type;
    result.device_tensor.shape = info.shape;
    result.device_tensor.prefer_private = false;
    result.host_tensor = host_binding.host;
    return result;
}

}  // namespace gfx_plugin
}  // namespace ov
