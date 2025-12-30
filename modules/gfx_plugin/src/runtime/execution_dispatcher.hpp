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

// Visibility helper for runtime factory symbols used by tests.
#if defined(__clang__) || defined(__GNUC__)
#    define GFX_STAGE_API __attribute__((visibility("default")))
#else
#    define GFX_STAGE_API
#endif

class GFX_STAGE_API GpuStageFactory {
public:
    static std::unique_ptr<GpuStage> create(const std::shared_ptr<const ov::Node>& node,
                                            GpuBackend backend = GpuBackend::Metal,
                                            void* device = nullptr,
                                            void* queue = nullptr);
};

// Compatibility alias matching the execution-dispatcher naming in docs.
class GFX_STAGE_API ExecutionDispatcher {
public:
    static std::unique_ptr<GpuStage> create(const std::shared_ptr<const ov::Node>& node,
                                            GpuBackend backend = GpuBackend::Metal,
                                            void* device = nullptr,
                                            void* queue = nullptr) {
        return GpuStageFactory::create(node, backend, device, queue);
    }
};

}  // namespace gfx_plugin
}  // namespace ov
