// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/plugin/compiled_model_backend.hpp"

#include "backends/vulkan/runtime/vulkan_backend.hpp"
#include "backends/vulkan/runtime/profiling/profiler.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<VulkanBackendState> create_vulkan_backend_state() {
    auto& ctx = VulkanContext::instance();
    auto state = std::make_unique<VulkanBackendState>();
    state->device = reinterpret_cast<GpuDeviceHandle>(ctx.device());
    state->queue = reinterpret_cast<GpuCommandQueueHandle>(ctx.queue());
    state->const_manager = nullptr;
    return state;
}

std::unique_ptr<GfxProfiler> VulkanBackendState::create_profiler(const GfxProfilerConfig& cfg) const {
    (void)cfg;
    auto& ctx = VulkanContext::instance();
    return std::make_unique<VulkanProfiler>(ctx.device(),
                                            ctx.physical_device(),
                                            ctx.queue_family_index());
}

}  // namespace gfx_plugin
}  // namespace ov
