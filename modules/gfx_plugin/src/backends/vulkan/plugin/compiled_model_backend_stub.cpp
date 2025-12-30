// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/plugin/compiled_model_backend.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<VulkanBackendState> create_vulkan_backend_state() {
    return {};
}

void release_vulkan_backend_state(VulkanBackendState&) {}

GpuDeviceHandle get_vulkan_device_handle(const VulkanBackendState* state) {
    return state ? state->device : nullptr;
}

GpuCommandQueueHandle get_vulkan_command_queue(const VulkanBackendState* state) {
    return state ? state->queue : nullptr;
}

GpuBufferManager* get_vulkan_const_buffer_manager(VulkanBackendState* state) {
    return state ? state->const_manager : nullptr;
}

const GpuBufferManager* get_vulkan_const_buffer_manager(const VulkanBackendState* state) {
    return state ? state->const_manager : nullptr;
}

bool vulkan_has_const_manager(const VulkanBackendState* /*state*/) {
    return true;
}

}  // namespace gfx_plugin
}  // namespace ov
