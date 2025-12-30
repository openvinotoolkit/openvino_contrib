// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/plugin/infer_io_vulkan.hpp"

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_vulkan(const ov::Tensor& host,
                                 GpuBufferPool* pool,
                                 BufferHandle* device_handle,
                                 BufferHandle* staging_handle,
                                 const char* error_prefix) {
    OPENVINO_ASSERT(host && host.data(), error_prefix, ": input host tensor is empty");
    const size_t bytes = host.get_byte_size();

    GpuTensor tensor{};
    tensor.shape = host.get_shape();
    tensor.expected_type = host.get_element_type();
    tensor.buf.type = host.get_element_type();
    tensor.buf.backend = GpuBackend::Vulkan;

    if (bytes == 0) {
        return tensor;
    }

    OPENVINO_ASSERT(pool && device_handle && staging_handle,
                    error_prefix, ": Vulkan input staging handles are missing");

    GpuBufferDesc staging_desc;
    staging_desc.bytes = bytes;
    staging_desc.type = host.get_element_type();
    staging_desc.usage = BufferUsage::Staging;
    staging_desc.cpu_read = false;
    staging_desc.cpu_write = true;
    staging_desc.prefer_device_local = false;
    GpuBuffer staging = pool->ensure(*staging_handle, staging_desc);
    gpu_copy_from_host(staging, host.data(), bytes);

    GpuBufferDesc device_desc;
    device_desc.bytes = bytes;
    device_desc.type = host.get_element_type();
    device_desc.usage = BufferUsage::IO;
    device_desc.cpu_read = false;
    device_desc.cpu_write = false;
    device_desc.prefer_device_local = true;
    GpuBuffer buf = pool->ensure(*device_handle, device_desc);
    if (bytes && staging.valid() && buf.valid()) {
        gpu_copy_buffer(nullptr, staging, buf, bytes);
    }

    tensor.buf = buf;
    tensor.buf.backend = GpuBackend::Vulkan;
    tensor.prefer_private = true;
    return tensor;
}

OutputBindingResult bind_host_output_vulkan(const GpuTensor& dev,
                                           const OutputViewInfo& info,
                                           const ov::Tensor* host_override,
                                           GpuBufferPool* pool,
                                           BufferHandle* staging_handle,
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

    OPENVINO_ASSERT(pool && staging_handle,
                    error_prefix, ": Vulkan output staging handle is missing");

    ov::Tensor host = host_override && *host_override ? *host_override : ov::Tensor(info.type, info.shape);
    if (bytes) {
        GpuBufferDesc staging_desc;
        staging_desc.bytes = bytes;
        staging_desc.type = info.type;
        staging_desc.usage = BufferUsage::Staging;
        staging_desc.cpu_read = true;
        staging_desc.cpu_write = false;
        staging_desc.prefer_device_local = false;
        GpuBuffer staging = pool->ensure(*staging_handle, staging_desc);
        if (dev.buf.buffer != staging.buffer) {
            gpu_copy_buffer(nullptr, dev.buf, staging, bytes);
        }
        gpu_copy_to_host(staging, host.data(), bytes);
    }
    result.host_tensor = host;
    result.device_tensor.expected_type = info.type;
    if (result.device_tensor.shape.empty()) {
        result.device_tensor.shape = info.shape;
    }
    return result;
}

}  // namespace gfx_plugin
}  // namespace ov
