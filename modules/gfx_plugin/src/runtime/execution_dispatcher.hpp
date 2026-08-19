// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "runtime/gpu_stage.hpp"
#include "runtime/stage_materialization_context.hpp"

namespace ov {
namespace gfx_plugin {

class GpuStageFactory {
public:
    using StageFactoryFn = std::unique_ptr<GpuStage> (*)(const RuntimeStageMaterializationContext&,
                                                         void* device,
                                                         void* queue);

    static bool register_factory(GpuBackend backend, StageFactoryFn fn);
    static StageFactoryFn factory_for_backend(GpuBackend backend);

    static std::unique_ptr<GpuStage> create(const RuntimeStageMaterializationContext& context,
                                            GpuBackend backend,
                                            void* device = nullptr,
                                            void* queue = nullptr);
};

// Compatibility alias matching the execution-dispatcher naming in docs.
class ExecutionDispatcher {
public:
    static std::unique_ptr<GpuStage> create(const RuntimeStageMaterializationContext& context,
                                            GpuBackend backend,
                                            void* device = nullptr,
                                            void* queue = nullptr) {
        return GpuStageFactory::create(context, backend, device, queue);
    }
};

}  // namespace gfx_plugin
}  // namespace ov
