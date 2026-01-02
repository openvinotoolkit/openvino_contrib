// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gpu_memory_ops.hpp"

#include <array>

namespace ov {
namespace gfx_plugin {

namespace {
constexpr size_t kBackendCount = 2;

size_t backend_index(GpuBackend backend) {
    switch (backend) {
    case GpuBackend::Metal:
        return 0;
    case GpuBackend::Vulkan:
        return 1;
    default:
        break;
    }
    return 0;
}

std::array<GpuMemoryOpsFn, kBackendCount>& memory_registry() {
    static std::array<GpuMemoryOpsFn, kBackendCount> registry{};
    return registry;
}

const GpuMemoryOps& null_ops() {
    static const GpuMemoryOps ops{};
    return ops;
}
}  // namespace

bool register_memory_ops(GpuBackend backend, GpuMemoryOpsFn fn) {
    memory_registry()[backend_index(backend)] = fn;
    return true;
}

const GpuMemoryOps& memory_ops_for_backend(GpuBackend backend) {
    auto fn = memory_registry()[backend_index(backend)];
    if (fn) {
        return fn();
    }
    return null_ops();
}

}  // namespace gfx_plugin
}  // namespace ov
