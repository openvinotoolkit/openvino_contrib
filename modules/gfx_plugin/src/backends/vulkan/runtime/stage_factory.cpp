// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_stage_factory.hpp"

#include "backends/vulkan/runtime/vulkan_executor.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_vulkan_stage(const std::shared_ptr<const ov::Node>& node,
                                              void*,
                                              void*) {
    return std::make_unique<VulkanStage>(node);
}

}  // namespace gfx_plugin
}  // namespace ov
