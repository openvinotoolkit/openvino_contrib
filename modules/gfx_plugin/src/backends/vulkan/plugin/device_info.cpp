// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/gfx_device_info.hpp"

#include "backends/vulkan/runtime/vulkan_backend.hpp"

namespace ov {
namespace gfx_plugin {

void fill_vulkan_device_info(GfxDeviceInfo& info, const ov::AnyMap& /*properties*/) {
    const auto& ctx = VulkanContext::instance();
    const auto& name = ctx.device_name();
    if (!name.empty()) {
        info.device_name = name;
        info.full_name = "GFX (" + name + ")";
        info.available_devices = {name};
    } else {
        info.device_name = "GFX (Vulkan)";
        info.full_name = "GFX (Vulkan)";
    }
}

}  // namespace gfx_plugin
}  // namespace ov
