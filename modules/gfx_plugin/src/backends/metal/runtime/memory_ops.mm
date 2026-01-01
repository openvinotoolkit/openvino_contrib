// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gpu_memory_ops.hpp"

#include "backends/metal/runtime/metal_memory.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
GpuMemoryOps make_metal_ops() {
    GpuMemoryOps ops{};
    ops.map = [](const GpuBuffer& buf) -> void* { return metal_map_buffer(buf); };
    ops.unmap = [](const GpuBuffer& buf) { metal_unmap_buffer(buf); };
    ops.copy = [](GpuCommandQueueHandle queue,
                  const GpuBuffer& src,
                  const GpuBuffer& dst,
                  size_t bytes) { metal_copy_buffer(queue, src, dst, bytes); };
    return ops;
}
}  // namespace

const GpuMemoryOps& metal_memory_ops() {
    static const GpuMemoryOps ops = make_metal_ops();
    return ops;
}

}  // namespace gfx_plugin
}  // namespace ov
