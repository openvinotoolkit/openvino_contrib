// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/execution_dispatcher.hpp"

#include "openvino/core/except.hpp"

#include <algorithm>
#include <mutex>
#include <vector>

namespace ov {
namespace gfx_plugin {

namespace {
using StageFactoryFn = GpuStageFactory::StageFactoryFn;

struct StageFactoryRegistration {
    GpuBackend backend = GpuBackend::Unknown;
    StageFactoryFn factory = nullptr;
};

void validate_backend(GpuBackend backend) {
    if (backend == GpuBackend::Unknown) {
        OPENVINO_THROW("GFX: stage factory requires a known backend");
    }
}

std::vector<StageFactoryRegistration>& stage_registry() {
    static std::vector<StageFactoryRegistration> registry;
    return registry;
}

std::mutex& stage_registry_mutex() {
    static std::mutex mutex;
    return mutex;
}
}  // namespace

bool GpuStageFactory::register_factory(GpuBackend backend, StageFactoryFn fn) {
    validate_backend(backend);
    std::lock_guard<std::mutex> lock(stage_registry_mutex());
    auto& registry = stage_registry();
    auto it = std::find_if(registry.begin(), registry.end(), [backend](const StageFactoryRegistration& entry) {
        return entry.backend == backend;
    });
    if (it != registry.end()) {
        it->factory = fn;
    } else {
        registry.push_back({backend, fn});
    }
    return true;
}

GpuStageFactory::StageFactoryFn GpuStageFactory::factory_for_backend(GpuBackend backend) {
    validate_backend(backend);
    std::lock_guard<std::mutex> lock(stage_registry_mutex());
    const auto& registry = stage_registry();
    auto it = std::find_if(registry.begin(), registry.end(), [backend](const StageFactoryRegistration& entry) {
        return entry.backend == backend;
    });
    return it != registry.end() ? it->factory : nullptr;
}

std::unique_ptr<GpuStage> GpuStageFactory::create(const std::shared_ptr<const ov::Node>& node,
                                                  const RuntimeStageExecutableDescriptor* descriptor,
                                                  GpuBackend backend,
                                                  void* device,
                                                  void* queue) {
    OPENVINO_ASSERT(descriptor,
                    "GFX: stage materialization requires a compiler-owned "
                    "runtime executable descriptor for op ",
                    node ? node->get_type_name() : "<null>");
    auto fn = factory_for_backend(backend);
    if (fn) {
        return fn(node, descriptor, device, queue);
    }
    return nullptr;
}

}  // namespace gfx_plugin
}  // namespace ov
