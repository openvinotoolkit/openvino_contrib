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

void ensure_metal_memory_ops_registered() {
    static const bool registered = register_memory_ops(GpuBackend::Metal, &metal_memory_ops);
    (void)registered;
}

namespace {
const bool kRegistered = (ensure_metal_memory_ops_registered(), true);
}  // namespace

}  // namespace gfx_plugin
}  // namespace ov
