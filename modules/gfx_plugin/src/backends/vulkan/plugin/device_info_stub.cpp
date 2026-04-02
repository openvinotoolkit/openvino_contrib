// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/gfx_device_info.hpp"

namespace ov {
namespace gfx_plugin {

void fill_vulkan_device_info(GfxDeviceInfo& info, const ov::AnyMap& /*properties*/) {
    info.device_name = "GFX (Vulkan)";
    info.full_name = "GFX (Vulkan)";
    info.available_devices = {"0"};
}

}  // namespace gfx_plugin
}  // namespace ov
