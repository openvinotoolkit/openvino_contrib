// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "openvino/core/node.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeStageExecutableDescriptor;

class GpuStageFactory {
public:
    using StageFactoryFn = std::unique_ptr<GpuStage> (*)(const std::shared_ptr<const ov::Node>&,
                                                         const RuntimeStageExecutableDescriptor*,
                                                         void* device,
                                                         void* queue);

    static bool register_factory(GpuBackend backend, StageFactoryFn fn);
    static StageFactoryFn factory_for_backend(GpuBackend backend);

    static std::unique_ptr<GpuStage> create(const std::shared_ptr<const ov::Node>& node,
                                            const RuntimeStageExecutableDescriptor* descriptor,
                                            GpuBackend backend = GpuBackend::Metal,
                                            void* device = nullptr,
                                            void* queue = nullptr);
};

// Compatibility alias matching the execution-dispatcher naming in docs.
class ExecutionDispatcher {
public:
    static std::unique_ptr<GpuStage> create(const std::shared_ptr<const ov::Node>& node,
                                            const RuntimeStageExecutableDescriptor* descriptor,
                                            GpuBackend backend = GpuBackend::Metal,
                                            void* device = nullptr,
                                            void* queue = nullptr) {
        return GpuStageFactory::create(node, descriptor, backend, device, queue);
    }
};

}  // namespace gfx_plugin
}  // namespace ov
