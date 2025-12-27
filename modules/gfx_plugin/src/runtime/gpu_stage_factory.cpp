// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gpu_stage_factory.hpp"

#include "openvino/core/except.hpp"
#include "plugin/gfx_backend_config.hpp"

#if GFX_BACKEND_METAL_AVAILABLE
#include "backends/metal/runtime/op_factory.hpp"
#include "backends/metal/runtime/stage.hpp"
#endif

#if GFX_BACKEND_VULKAN_AVAILABLE
#include "backends/vulkan/runtime/stage.hpp"
#endif

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> GpuStageFactory::create(const std::shared_ptr<const ov::Node>& node,
                                                  GpuBackend backend,
                                                  void* device,
                                                  void* queue) {
    switch (backend) {
    case GpuBackend::Metal: {
#if GFX_BACKEND_METAL_AVAILABLE
        auto op = MetalOpFactory::create(node, device, queue);
        if (!op) {
            return nullptr;
        }
        return std::make_unique<MetalStage>(std::move(op));
#else
        OPENVINO_THROW("GFX Metal backend is not available in this build");
#endif
    }
    case GpuBackend::Vulkan: {
#if GFX_BACKEND_VULKAN_AVAILABLE
        return std::make_unique<VulkanStage>(node);
#else
        OPENVINO_THROW("GFX Vulkan backend is not available in this build");
#endif
    }
    default:
        break;
    }
    return nullptr;
}

}  // namespace gfx_plugin
}  // namespace ov
