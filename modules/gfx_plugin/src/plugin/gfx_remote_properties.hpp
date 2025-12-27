// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace ov {
namespace gfx_plugin {

// Common remote tensor/property keys.
constexpr const char* kGfxBufferProperty = "GFX_BUFFER";
constexpr const char* kGfxMemoryProperty = "GFX_MEMORY";
constexpr const char* kGfxHostVisibleProperty = "GFX_HOST_VISIBLE";
constexpr const char* kGfxStorageModeProperty = "GFX_STORAGE_MODE";

// Metal-specific keys.
constexpr const char* kMtlBufferProperty = "MTL_BUFFER";
constexpr const char* kStorageModeProperty = "STORAGE_MODE";

// Vulkan-specific keys.
constexpr const char* kVkBufferProperty = "VK_BUFFER";
constexpr const char* kVulkanBufferProperty = "VULKAN_BUFFER";
constexpr const char* kVkMemoryProperty = "VK_MEMORY";
constexpr const char* kVulkanMemoryProperty = "VULKAN_MEMORY";
constexpr const char* kHostVisibleProperty = "HOST_VISIBLE";

}  // namespace gfx_plugin
}  // namespace ov
