// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gpu_memory_ops.hpp"

#include "openvino/core/except.hpp"

#include <algorithm>
#include <mutex>
#include <vector>

namespace ov {
namespace gfx_plugin {

namespace {
struct MemoryOpsRegistration {
    GpuBackend backend = GpuBackend::Unknown;
    GpuMemoryOpsFn factory = nullptr;
};

void validate_backend(GpuBackend backend) {
    if (backend == GpuBackend::Unknown) {
        OPENVINO_THROW("GFX: memory ops require a known backend");
    }
}

std::vector<MemoryOpsRegistration>& memory_registry() {
    static std::vector<MemoryOpsRegistration> registry;
    return registry;
}

std::mutex& memory_registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

const GpuMemoryOps& null_ops() {
    static const GpuMemoryOps ops{};
    return ops;
}
}  // namespace

bool register_memory_ops(GpuBackend backend, GpuMemoryOpsFn fn) {
    validate_backend(backend);
    std::lock_guard<std::mutex> lock(memory_registry_mutex());
    auto& registry = memory_registry();
    auto it = std::find_if(registry.begin(), registry.end(), [backend](const MemoryOpsRegistration& entry) {
        return entry.backend == backend;
    });
    if (it != registry.end()) {
        it->factory = fn;
    } else {
        registry.push_back({backend, fn});
    }
    return true;
}

const GpuMemoryOps& memory_ops_for_backend(GpuBackend backend) {
    validate_backend(backend);
    GpuMemoryOpsFn fn = nullptr;
    {
        std::lock_guard<std::mutex> lock(memory_registry_mutex());
        const auto& registry = memory_registry();
        auto it = std::find_if(registry.begin(), registry.end(), [backend](const MemoryOpsRegistration& entry) {
            return entry.backend == backend;
        });
        fn = it != registry.end() ? it->factory : nullptr;
    }
    if (fn) {
        return fn();
    }
    return null_ops();
}

}  // namespace gfx_plugin
}  // namespace ov
