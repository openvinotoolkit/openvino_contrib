// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gpu_memory_ops.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
const GpuMemoryOps& null_ops() {
    static const GpuMemoryOps ops{};
    return ops;
}
}  // namespace

const GpuMemoryOps& memory_ops_for_backend(GpuBackend backend) {
    switch (backend) {
        case GpuBackend::Metal:
            return metal_memory_ops();
        case GpuBackend::Vulkan:
            return vulkan_memory_ops();
        default:
            return null_ops();
    }
}

}  // namespace gfx_plugin
}  // namespace ov
