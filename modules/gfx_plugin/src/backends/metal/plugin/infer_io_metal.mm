// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/plugin/infer_io_metal.hpp"

#include <chrono>

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_metal(const ov::Tensor& host,
                                IGpuAllocator* allocator,
                                GfxProfiler* profiler,
                                const char* error_prefix) {
    HostInputBinding binding = prepare_host_input_binding(host, GpuBackend::Metal, error_prefix);
    GpuTensor tensor = binding.tensor;
    const size_t bytes = binding.bytes;
    if (bytes == 0) {
        return tensor;
    }

    OPENVINO_ASSERT(allocator, error_prefix, ": GPU allocator is null");
    const bool profiling = (profiler != nullptr);
    const auto transfer_start = profiling ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    tensor.buf = allocator->wrap_shared(const_cast<void*>(host.data()), bytes, host.get_element_type());
    tensor.buf.backend = allocator->backend();
    tensor.buf.host_visible = true;
    tensor.prefer_private = false;
    if (profiling) {
        const auto cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - transfer_start);
        profiler->record_transfer("input_h2d", bytes, true, cpu_us);
        profiler->increment_counter("h2d_transfer_count");
    }
    return tensor;
}

OutputBindingResult bind_host_output_metal(const GpuTensor& dev,
                                          const OutputViewInfo& info,
                                          const ov::Tensor* host_override,
                                          ov::Tensor* reusable_host,
                                          IGpuAllocator* allocator,
                                          GpuBufferPool* pool,
                                          BufferHandle* staging_handle,
                                          GpuCommandQueueHandle metal_queue,
                                          GfxProfiler* profiler,
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
        const bool profiling = (profiler != nullptr);
        const auto transfer_start = profiling ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
        GpuBuffer shared = allocator->wrap_shared(const_cast<void*>(host_override->data()), bytes, info.type);
        if (bytes && dev.buf.buffer != shared.buffer) {
            gpu_copy_buffer(metal_queue, dev.buf, shared, bytes);
        }
        if (profiling) {
            const auto cpu_us =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - transfer_start);
            profiler->record_transfer("output_d2h", bytes, false, cpu_us);
            profiler->increment_counter("d2h_transfer_count");
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
        const bool profiling = (profiler != nullptr);
        const auto transfer_start = profiling ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
        const size_t bytes = info.shape.empty() ? dev.buf.size : tensor_byte_size(info.shape, info.type);
        void* ptr = gpu_map_buffer(dev.buf);
        OPENVINO_ASSERT(ptr, error_prefix, ": shared output buffer has no CPU pointer");
        if (profiling) {
            const auto cpu_us =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - transfer_start);
            profiler->record_transfer("output_d2h", bytes, false, cpu_us);
            profiler->increment_counter("d2h_transfer_count");
        }
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
        const bool profiling = (profiler != nullptr);
        const auto transfer_start = profiling ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
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
        if (profiling) {
            const auto cpu_us =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - transfer_start);
            profiler->record_transfer("output_d2h", bytes, false, cpu_us);
            profiler->increment_counter("d2h_transfer_count");
        }
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
