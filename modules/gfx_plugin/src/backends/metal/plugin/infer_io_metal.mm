// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/plugin/infer_io_metal.hpp"

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_metal(const ov::Tensor& host,
                                MetalAllocatorCore* metal_core,
                                const char* error_prefix) {
    OPENVINO_ASSERT(host && host.data(), error_prefix, ": input host tensor is empty");
    const size_t bytes = host.get_byte_size();

    GpuTensor tensor{};
    tensor.shape = host.get_shape();
    tensor.expected_type = host.get_element_type();
    tensor.buf.type = host.get_element_type();
    tensor.buf.backend = GpuBackend::Metal;

    if (bytes == 0) {
        return tensor;
    }

    OPENVINO_ASSERT(metal_core, error_prefix, ": Metal allocator core is null");
    tensor.buf = metal_core->wrap_shared(host.data(), bytes, host.get_element_type());
    tensor.buf.backend = GpuBackend::Metal;
    tensor.buf.host_visible = true;
    tensor.prefer_private = false;
    return tensor;
}

OutputBindingResult bind_host_output_metal(const GpuTensor& dev,
                                          const OutputViewInfo& info,
                                          const ov::Tensor* host_override,
                                          MetalAllocatorCore* metal_core,
                                          MetalAllocator* metal_allocator,
                                          GpuCommandQueueHandle metal_queue,
                                          const char* error_prefix) {
    OutputBindingResult result{};
    result.device_tensor = dev;

    size_t bytes = 0;
    if (info.type != ov::element::dynamic) {
        bytes = info.type.size();
        for (auto d : info.shape) {
            bytes *= d;
        }
    }

    OPENVINO_ASSERT(metal_core && metal_allocator,
                    error_prefix, ": Metal allocator is not available");

    if (host_override && *host_override) {
        MetalBuffer shared = metal_core->wrap_shared(host_override->data(), bytes, info.type);
        if (bytes && dev.buf.buffer != shared.buffer) {
            gpu_copy_buffer(metal_queue, dev.buf, shared, bytes);
        }
        result.device_tensor = dev;
        result.device_tensor.buf = shared;
        result.device_tensor.expected_type = info.type;
        result.device_tensor.shape = info.shape;
        result.device_tensor.prefer_private = false;
        result.host_tensor = *host_override;
        return result;
    }

    if (dev.buf.host_visible && dev.buf.buffer) {
        void* ptr = metal_map_buffer(dev.buf);
        OPENVINO_ASSERT(ptr, error_prefix, ": shared output buffer has no CPU pointer");
        result.host_tensor = ov::Tensor(info.type, info.shape, ptr);
        result.device_tensor.expected_type = info.type;
        if (result.device_tensor.shape.empty()) {
            result.device_tensor.shape = info.shape;
        }
        result.device_tensor.prefer_private = false;
        return result;
    }

    if (bytes) {
        BufferDesc desc;
        desc.bytes = bytes;
        desc.type = info.type;
        desc.usage = BufferUsage::IO;
        desc.storage = MetalStorage::Shared;
        desc.cpu_read = true;
        desc.cpu_write = true;
        MetalBuffer shared = metal_allocator->allocate(desc, /*persistent=*/false);
        gpu_copy_buffer(metal_queue, dev.buf, shared, bytes);
        result.device_tensor = dev;
        result.device_tensor.buf = shared;
        result.device_tensor.expected_type = info.type;
        result.device_tensor.shape = info.shape;
        result.device_tensor.prefer_private = false;
        void* ptr = metal_map_buffer(shared);
        OPENVINO_ASSERT(ptr, error_prefix, ": shared output buffer has no CPU pointer");
        result.host_tensor = ov::Tensor(info.type, info.shape, ptr);
        return result;
    }

    result.device_tensor.expected_type = info.type;
    result.device_tensor.shape = info.shape;
    result.device_tensor.prefer_private = false;
    if (host_override && *host_override) {
        result.host_tensor = *host_override;
    } else {
        result.host_tensor = ov::Tensor(info.type, info.shape);
    }
    return result;
}

}  // namespace gfx_plugin
}  // namespace ov
