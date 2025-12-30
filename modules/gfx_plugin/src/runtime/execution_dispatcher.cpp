// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/execution_dispatcher.hpp"

#include "runtime/gfx_stage_factory.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> GpuStageFactory::create(const std::shared_ptr<const ov::Node>& node,
                                                  GpuBackend backend,
                                                  void* device,
                                                  void* queue) {
    switch (backend) {
    case GpuBackend::Metal: {
        return create_metal_stage(node, device, queue);
    }
    case GpuBackend::Vulkan: {
        return create_vulkan_stage(node, device, queue);
    }
    default:
        break;
    }
    return nullptr;
}

}  // namespace gfx_plugin
}  // namespace ov
