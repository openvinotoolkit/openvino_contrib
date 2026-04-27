// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <vector>

#include "openvino/runtime/properties.hpp"
#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxDeviceInfo {
    GpuBackend backend = GpuBackend::Metal;
    std::string backend_name;
    std::string device_name;
    std::string full_name;
    ov::device::Type device_type = ov::device::Type::INTEGRATED;
    std::vector<std::string> capabilities;
    std::vector<std::string> available_devices;
    std::string device_id;
};

GfxDeviceInfo query_device_info_from_properties(const ov::AnyMap& properties,
                                                bool log_fallback,
                                                const char* log_tag);

GfxDeviceInfo query_device_info(GpuBackend backend, const ov::AnyMap& properties);

}  // namespace gfx_plugin
}  // namespace ov
