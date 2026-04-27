// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/gfx_plugin/properties.hpp"

namespace ov {
namespace gfx_plugin {

// Vulkan-specific keys.
constexpr const char* kVkBufferProperty = "VK_BUFFER";
constexpr const char* kVulkanBufferProperty = "VULKAN_BUFFER";
constexpr const char* kVkMemoryProperty = "VK_MEMORY";
constexpr const char* kVulkanMemoryProperty = "VULKAN_MEMORY";
constexpr const char* kHostVisibleProperty = "HOST_VISIBLE";

}  // namespace gfx_plugin
}  // namespace ov
