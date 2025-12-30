// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/gfx_device_info.hpp"

namespace ov {
namespace gfx_plugin {

void fill_metal_device_info(GfxDeviceInfo& info, const ov::AnyMap& /*properties*/) {
    info.device_name = "GFX (Metal)";
    info.full_name = "GFX (Metal)";
}

}  // namespace gfx_plugin
}  // namespace ov
