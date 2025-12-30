// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "plugin/gfx_backend_config.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "backends/vulkan/plugin/compiled_model_state.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<VulkanBackendState> create_vulkan_backend_state();
void release_vulkan_backend_state(VulkanBackendState& state);
GpuDeviceHandle get_vulkan_device_handle(const VulkanBackendState* state);
GpuCommandQueueHandle get_vulkan_command_queue(const VulkanBackendState* state);
GpuBufferManager* get_vulkan_const_buffer_manager(VulkanBackendState* state);
const GpuBufferManager* get_vulkan_const_buffer_manager(const VulkanBackendState* state);
bool vulkan_has_const_manager(const VulkanBackendState* state);

}  // namespace gfx_plugin
}  // namespace ov
