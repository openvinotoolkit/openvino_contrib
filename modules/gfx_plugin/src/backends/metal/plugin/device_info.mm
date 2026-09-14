// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/gfx_device_info.hpp"

#include "backends/metal/runtime/metal_memory.hpp"
#include "plugin/gfx_property_utils.hpp"

namespace ov {
namespace gfx_plugin {

void fill_metal_device_info(GfxDeviceInfo& info, const ov::AnyMap& properties) {
    auto names = metal_get_device_names();
    const int requested_id = parse_device_id(properties);
    for (size_t idx = 0; idx < names.size(); ++idx) {
        info.available_devices.push_back(std::to_string(idx));
    }
    if (!names.empty()) {
        const size_t selected_idx =
            requested_id >= 0 && static_cast<size_t>(requested_id) < names.size() ? static_cast<size_t>(requested_id)
                                                                                   : 0u;
        info.device_name = names[selected_idx];
        info.full_name = "GFX (" + info.device_name + ")";
    } else {
        info.device_name = "GFX";
        info.full_name = "GFX";
    }
}

}  // namespace gfx_plugin
}  // namespace ov
