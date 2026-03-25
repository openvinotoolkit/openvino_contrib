// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/plugin/infer_io_vulkan.hpp"

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_vulkan(const ov::Tensor& host,
                                 GpuBufferPool* pool,
                                 BufferHandle* device_handle,
                                 BufferHandle* staging_handle,
                                 const char* error_prefix) {
    HostInputBinding binding = prepare_host_input_binding(host, GpuBackend::Vulkan, error_prefix);
    GpuTensor tensor = binding.tensor;
    const size_t bytes = binding.bytes;
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
                                            const ov::Tensor* reusable_host,
                                            GpuBufferPool* pool,
                                            BufferHandle* staging_handle,
                                            const char* error_prefix) {
    OutputBindingResult result{};
    result.device_tensor = dev;

    HostOutputBinding host_binding = prepare_host_output_binding(info, host_override, reusable_host);
    size_t bytes = host_binding.bytes;

    OPENVINO_ASSERT(pool && staging_handle,
                    error_prefix, ": Vulkan output staging handle is missing");

    ov::Tensor host = host_binding.host;
    if (bytes) {
        if (dev.buf.host_visible && dev.buf.buffer) {
            if (pool && staging_handle) {
                pool->release(*staging_handle);
            }
            gpu_copy_to_host(dev.buf, host.data(), bytes);
            result.host_tensor = host;
            result.device_tensor.expected_type = info.type;
            if (result.device_tensor.shape.empty()) {
                result.device_tensor.shape = info.shape;
            }
            return result;
        }
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
