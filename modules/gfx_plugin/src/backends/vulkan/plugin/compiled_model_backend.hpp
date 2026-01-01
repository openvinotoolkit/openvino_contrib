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

}  // namespace gfx_plugin
}  // namespace ov
