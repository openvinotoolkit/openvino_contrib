// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/execution_dispatcher.hpp"

#include "openvino/core/except.hpp"

#include <array>

namespace ov {
namespace gfx_plugin {

namespace {
using StageFactoryFn = GpuStageFactory::StageFactoryFn;

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

std::array<StageFactoryFn, kBackendCount>& stage_registry() {
    static std::array<StageFactoryFn, kBackendCount> registry{};
    return registry;
}
}  // namespace

bool GpuStageFactory::register_factory(GpuBackend backend, StageFactoryFn fn) {
    stage_registry()[backend_index(backend)] = fn;
    return true;
}

GpuStageFactory::StageFactoryFn GpuStageFactory::factory_for_backend(GpuBackend backend) {
    return stage_registry()[backend_index(backend)];
}

std::unique_ptr<GpuStage> GpuStageFactory::create(const std::shared_ptr<const ov::Node>& node,
                                                  GpuBackend backend,
                                                  void* device,
                                                  void* queue) {
    auto fn = factory_for_backend(backend);
    if (fn) {
        return fn(node, device, queue);
    }
    return nullptr;
}

}  // namespace gfx_plugin
}  // namespace ov
