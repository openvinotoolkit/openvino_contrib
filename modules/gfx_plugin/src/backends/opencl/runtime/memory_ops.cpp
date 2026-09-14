// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gpu_memory_ops.hpp"

#include "backends/opencl/runtime/memory_api.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

GpuMemoryOps make_opencl_ops() {
    GpuMemoryOps ops{};
    ops.map = [](const GpuBuffer& buf) -> void* { return opencl_map_buffer(buf); };
    ops.unmap = [](const GpuBuffer& buf) { opencl_unmap_buffer(buf); };
    ops.flush = [](const GpuBuffer& buf, size_t bytes, size_t offset) {
        opencl_flush_buffer(buf, bytes, offset);
    };
    ops.invalidate = [](const GpuBuffer& buf, size_t bytes, size_t offset) {
        opencl_invalidate_buffer(buf, bytes, offset);
    };
    ops.copy = [](GpuCommandQueueHandle execution_context,
                  const GpuBuffer& src,
                  const GpuBuffer& dst,
                  size_t bytes) { opencl_copy_buffer(execution_context, src, dst, bytes); };
    ops.copy_regions = [](GpuCommandQueueHandle execution_context,
                          const GpuBuffer& src,
                          const GpuBuffer& dst,
                          const GpuBufferCopyRegion* regions,
                          size_t region_count) {
        opencl_copy_buffer_regions(execution_context, src, dst, regions, region_count);
    };
    return ops;
}

}  // namespace

const GpuMemoryOps& opencl_memory_ops() {
    static const GpuMemoryOps ops = make_opencl_ops();
    return ops;
}

void ensure_opencl_memory_ops_registered() {
    static const bool registered = register_memory_ops(GpuBackend::OpenCL, &opencl_memory_ops);
    (void)registered;
}

namespace {
const bool kRegistered = (ensure_opencl_memory_ops_registered(), true);
}  // namespace

}  // namespace gfx_plugin
}  // namespace ov
