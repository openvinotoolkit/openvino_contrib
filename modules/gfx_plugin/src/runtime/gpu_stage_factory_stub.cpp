// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gpu_stage_factory.hpp"

#include "openvino/core/except.hpp"
#include "backends/vulkan/runtime/stage.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> GpuStageFactory::create(const std::shared_ptr<const ov::Node>& node,
                                                  GpuBackend backend,
                                                  void* /*device*/,
                                                  void* /*queue*/) {
    if (backend == GpuBackend::Vulkan) {
        return std::make_unique<VulkanStage>(node);
    }
    if (backend == GpuBackend::Metal) {
        OPENVINO_THROW("GFX Metal backend is not available in this build");
    }
    return nullptr;
}

}  // namespace gfx_plugin
}  // namespace ov
