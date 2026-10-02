// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/memory_api.hpp"

#include "backends/opencl/runtime/opencl_api.hpp"
#include "openvino/core/except.hpp"

#include <unordered_map>
#include <vector>

namespace ov {
namespace gfx_plugin {

size_t opencl_allocation_bytes(size_t bytes, ov::element::Type type) {
  if (type != ov::element::boolean && type != ov::element::f16) {
    return bytes;
  }
  return ((bytes + 3u) / 4u) * 4u;
}

namespace {

cl_command_queue resolve_queue(GpuCommandQueueHandle execution_context) {
  if (execution_context) {
    return reinterpret_cast<cl_command_queue>(execution_context);
  }
  return OpenClRuntimeContext::instance()->queue();
}

std::unordered_map<GpuBufferHandle, std::vector<void *>> &
mapped_opencl_pointers() {
  thread_local std::unordered_map<GpuBufferHandle, std::vector<void *>>
      pointers;
  return pointers;
}

} // namespace

void *opencl_map_buffer(const GpuBuffer &buf) {
  if (!buf.buffer) {
    return nullptr;
  }
  auto ctx = OpenClRuntimeContext::instance();
  cl_int status = CL_SUCCESS;
  void *mapped = ctx->api().fn().clEnqueueMapBuffer(
      ctx->queue(), reinterpret_cast<cl_mem>(buf.buffer), CL_TRUE,
      CL_MAP_READ | CL_MAP_WRITE, 0, buf.size, 0, nullptr, nullptr, &status);
  opencl_check(status, "clEnqueueMapBuffer");
  mapped_opencl_pointers()[buf.buffer].push_back(mapped);
  return mapped;
}

void opencl_unmap_buffer(const GpuBuffer &buf) {
  if (!buf.buffer) {
    return;
  }
  auto ctx = OpenClRuntimeContext::instance();
  auto &pointers = mapped_opencl_pointers();
  auto it = pointers.find(buf.buffer);
  OPENVINO_ASSERT(
      it != pointers.end() && !it->second.empty(),
      "GFX OpenCL: unmap requested for buffer without a matching map");
  void *mapped = it->second.back();
  it->second.pop_back();
  if (it->second.empty()) {
    pointers.erase(it);
  }
  opencl_check(ctx->api().fn().clEnqueueUnmapMemObject(
                   ctx->queue(), reinterpret_cast<cl_mem>(buf.buffer), mapped,
                   0, nullptr, nullptr),
               "clEnqueueUnmapMemObject");
  ctx->finish();
}

void opencl_flush_buffer(const GpuBuffer &, size_t, size_t) {}

void opencl_invalidate_buffer(const GpuBuffer &, size_t, size_t) {}

void opencl_free_buffer(GpuBuffer &buf) {
  if (!buf.buffer) {
    return;
  }
  const auto &api = OpenClApi::instance();
  opencl_check(
      api.fn().clReleaseMemObject(reinterpret_cast<cl_mem>(buf.buffer)),
      "clReleaseMemObject");
  buf.buffer = nullptr;
  buf.size = 0;
  buf.allocation_uid = 0;
}

void opencl_copy_buffer(GpuCommandQueueHandle execution_context,
                        const GpuBuffer &src, const GpuBuffer &dst,
                        size_t bytes) {
  if (!src.buffer || !dst.buffer || bytes == 0) {
    return;
  }
  auto ctx = OpenClRuntimeContext::instance();
  opencl_check(
      ctx->api().fn().clEnqueueCopyBuffer(resolve_queue(execution_context),
                                          reinterpret_cast<cl_mem>(src.buffer),
                                          reinterpret_cast<cl_mem>(dst.buffer),
                                          0, 0, bytes, 0, nullptr, nullptr),
      "clEnqueueCopyBuffer");
  ctx->finish();
}

void opencl_copy_buffer_regions(GpuCommandQueueHandle execution_context,
                                const GpuBuffer &src, const GpuBuffer &dst,
                                const GpuBufferCopyRegion *regions,
                                size_t region_count) {
  if (!src.buffer || !dst.buffer || !regions || region_count == 0) {
    return;
  }
  auto ctx = OpenClRuntimeContext::instance();
  auto queue = resolve_queue(execution_context);
  for (size_t i = 0; i < region_count; ++i) {
    const auto &region = regions[i];
    if (region.bytes == 0) {
      continue;
    }
    opencl_check(ctx->api().fn().clEnqueueCopyBuffer(
                     queue, reinterpret_cast<cl_mem>(src.buffer),
                     reinterpret_cast<cl_mem>(dst.buffer), region.src_offset,
                     region.dst_offset, region.bytes, 0, nullptr, nullptr),
                 "clEnqueueCopyBuffer(region)");
  }
  ctx->finish();
}

} // namespace gfx_plugin
} // namespace ov
