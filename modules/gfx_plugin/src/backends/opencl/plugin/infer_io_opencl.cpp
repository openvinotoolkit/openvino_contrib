// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/plugin/infer_io_opencl.hpp"

#include <chrono>

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

GpuTensor bind_host_input_opencl(const ov::Tensor &host, GpuBufferPool *pool,
                                 BufferHandle *device_handle,
                                 GfxProfiler *profiler,
                                 const char *error_prefix) {
  HostInputBinding binding =
      prepare_host_input_binding(host, GpuBackend::OpenCL, error_prefix);
  GpuTensor tensor = binding.tensor;
  const size_t bytes = binding.bytes;
  if (bytes == 0) {
    return tensor;
  }

  OPENVINO_ASSERT(pool && device_handle, error_prefix,
                  ": OpenCL input device handle is missing");

  GpuBufferDesc device_desc;
  device_desc.bytes = bytes;
  device_desc.type = host.get_element_type();
  device_desc.usage = BufferUsage::IO;
  device_desc.cpu_read = false;
  device_desc.cpu_write = true;
  device_desc.prefer_device_local = false;
  GpuBuffer buf = pool->ensure(*device_handle, device_desc);

  const bool profiling = (profiler != nullptr);
  const auto transfer_start = profiling
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
  gpu_copy_from_host(buf, host.data(), bytes);
  if (profiling) {
    const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - transfer_start);
    profiler->record_transfer("input_h2d", bytes, true, cpu_us);
    profiler->increment_counter("h2d_transfer_count");
  }

  tensor.buf = buf;
  tensor.buf.backend = GpuBackend::OpenCL;
  tensor.prefer_private = false;
  return tensor;
}

OutputBindingResult
bind_host_output_opencl(const GpuTensor &dev, const OutputViewInfo &info,
                        const ov::Tensor *host_override,
                        ov::Tensor *reusable_host, GpuBufferPool *pool,
                        BufferHandle *staging_handle,
                        GpuCommandQueueHandle queue, GfxProfiler *profiler,
                        const char *error_prefix) {
  OutputBindingResult result{};
  result.device_tensor = dev;

  HostOutputBinding host_binding =
      prepare_host_output_binding(info, host_override, reusable_host);
  const size_t bytes = host_binding.bytes;
  ov::Tensor host = host_binding.host;
  if (bytes == 0) {
    result.host_tensor = host;
    return result;
  }

  OPENVINO_ASSERT(host && host.data(), error_prefix,
                  ": OpenCL output host tensor is empty");
  OPENVINO_ASSERT(dev.buf.valid(), error_prefix,
                  ": OpenCL output device buffer is not initialized");

  GpuBuffer readback = dev.buf;
  if (!dev.buf.host_visible) {
    OPENVINO_ASSERT(pool && staging_handle, error_prefix,
                    ": OpenCL output staging handle is missing");
    GpuBufferDesc staging_desc;
    staging_desc.bytes = bytes;
    staging_desc.type = info.type;
    staging_desc.usage = BufferUsage::Staging;
    staging_desc.cpu_read = true;
    staging_desc.cpu_write = false;
    staging_desc.prefer_device_local = false;
    readback = pool->ensure(*staging_handle, staging_desc);
    if (dev.buf.buffer != readback.buffer) {
      gpu_copy_buffer(queue, dev.buf, readback, bytes);
    }
  } else if (pool && staging_handle) {
    pool->release(*staging_handle);
  }

  const bool profiling = (profiler != nullptr);
  const auto transfer_start = profiling
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
  gpu_copy_to_host(readback, host.data(), bytes);
  if (profiling) {
    const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - transfer_start);
    profiler->record_transfer("output_d2h", bytes, false, cpu_us);
    profiler->increment_counter("d2h_transfer_count");
  }

  result.host_tensor = host;
  result.device_tensor.expected_type = info.type;
  if (result.device_tensor.shape.empty()) {
    result.device_tensor.shape = info.shape;
  }
  return result;
}

} // namespace gfx_plugin
} // namespace ov
