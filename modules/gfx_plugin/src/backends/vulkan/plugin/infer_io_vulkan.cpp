// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/plugin/infer_io_vulkan.hpp"

#include <chrono>

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_vulkan(const ov::Tensor& host,
                                 GpuBufferPool* pool,
                                 BufferHandle* device_handle,
                                 BufferHandle* staging_handle,
                                 GpuCommandBufferHandle command_buffer,
                                 GfxProfiler* profiler,
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
    const bool profiling = (profiler != nullptr);
    const auto transfer_start = profiling ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
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
        gpu_copy_buffer(command_buffer, staging, buf, bytes);
    }
    if (profiling) {
        const auto cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - transfer_start);
        profiler->record_transfer("input_h2d", bytes, true, cpu_us);
        profiler->increment_counter("h2d_transfer_count");
    }

    tensor.buf = buf;
    tensor.buf.backend = GpuBackend::Vulkan;
    tensor.prefer_private = true;
    return tensor;
}

VulkanOutputBindingResult prepare_host_output_vulkan(const GpuTensor& dev,
                                                     const OutputViewInfo& info,
                                                     const ov::Tensor* host_override,
                                                     const ov::Tensor* reusable_host,
                                                     GpuBufferPool* pool,
                                                     BufferHandle* staging_handle,
                                                     GpuCommandBufferHandle command_buffer,
                                                     GfxProfiler* /*profiler*/,
                                                     const char* error_prefix) {
    VulkanOutputBindingResult result{};
    result.binding.device_tensor = dev;

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
            result.readback_buffer = dev.buf;
            result.readback_bytes = bytes;
            result.binding.host_tensor = host;
            result.binding.device_tensor.expected_type = info.type;
            if (result.binding.device_tensor.shape.empty()) {
                result.binding.device_tensor.shape = info.shape;
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
            gpu_copy_buffer(command_buffer, dev.buf, staging, bytes);
        }
        result.readback_buffer = staging;
        result.readback_bytes = bytes;
    }
    result.binding.host_tensor = host;
    result.binding.device_tensor.expected_type = info.type;
    if (result.binding.device_tensor.shape.empty()) {
        result.binding.device_tensor.shape = info.shape;
    }
    return result;
}

void finalize_host_output_vulkan(VulkanOutputBindingResult& result,
                                 GfxProfiler* profiler,
                                 const char* error_prefix) {
    if (result.readback_bytes == 0) {
        return;
    }
    OPENVINO_ASSERT(result.binding.host_tensor && result.binding.host_tensor.data(),
                    error_prefix,
                    ": Vulkan output host tensor is empty");
    OPENVINO_ASSERT(result.readback_buffer.valid(),
                    error_prefix,
                    ": Vulkan output readback buffer is not initialized");

    const bool profiling = (profiler != nullptr);
    const auto transfer_start = profiling ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    gpu_copy_to_host(result.readback_buffer, result.binding.host_tensor.data(), result.readback_bytes);
    if (profiling) {
        const auto cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - transfer_start);
        profiler->record_transfer("output_d2h", result.readback_bytes, false, cpu_us);
        profiler->increment_counter("d2h_transfer_count");
    }
}

}  // namespace gfx_plugin
}  // namespace ov
