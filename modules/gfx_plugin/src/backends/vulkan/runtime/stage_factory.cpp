// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/stage_factory.hpp"

#include "backends/vulkan/runtime/vulkan_executor.hpp"
#include "plugin/stateful_stage.hpp"
#include "runtime/execution_dispatcher.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_vulkan_stage(const std::shared_ptr<const ov::Node>& node,
                                              void*,
                                              void*) {
    if (auto stateful = create_stateful_stage(node)) {
        return stateful;
    }
    return std::make_unique<VulkanStage>(node);
}

void ensure_vulkan_stage_factory_registered() {
    static const bool registered = GpuStageFactory::register_factory(GpuBackend::Vulkan, &create_vulkan_stage);
    (void)registered;
}

namespace {
const bool kRegistered = (ensure_vulkan_stage_factory_registered(), true);
}  // namespace

}  // namespace gfx_plugin
}  // namespace ov
