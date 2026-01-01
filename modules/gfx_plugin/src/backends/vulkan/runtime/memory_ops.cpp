// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gpu_memory_ops.hpp"

#include "backends/vulkan/runtime/memory_api.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
GpuMemoryOps make_vulkan_ops() {
    GpuMemoryOps ops{};
    ops.map = [](const GpuBuffer& buf) -> void* { return vulkan_map_buffer(buf); };
    ops.unmap = [](const GpuBuffer& buf) { vulkan_unmap_buffer(buf); };
    ops.flush = [](const GpuBuffer& buf, size_t bytes, size_t offset) {
        vulkan_flush_buffer(buf, bytes, offset);
    };
    ops.invalidate = [](const GpuBuffer& buf, size_t bytes, size_t offset) {
        vulkan_invalidate_buffer(buf, bytes, offset);
    };
    ops.copy = [](GpuCommandQueueHandle /*queue*/,
                  const GpuBuffer& src,
                  const GpuBuffer& dst,
                  size_t bytes) { vulkan_copy_buffer(src, dst, bytes); };
    return ops;
}
}  // namespace

const GpuMemoryOps& vulkan_memory_ops() {
    static const GpuMemoryOps ops = make_vulkan_ops();
    return ops;
}

}  // namespace gfx_plugin
}  // namespace ov
