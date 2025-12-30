// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/gfx_device_info.hpp"

#include "backends/metal/runtime/metal_memory.hpp"

namespace ov {
namespace gfx_plugin {

void fill_metal_device_info(GfxDeviceInfo& info, const ov::AnyMap& /*properties*/) {
    auto names = metal_get_device_names();
    if (!names.empty()) {
        info.available_devices = names;
        info.device_name = names.front();
        info.full_name = "GFX (" + info.device_name + ")";
    } else {
        info.device_name = "GFX";
        info.full_name = "GFX";
    }
}

}  // namespace gfx_plugin
}  // namespace ov
