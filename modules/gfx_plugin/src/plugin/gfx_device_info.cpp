// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/gfx_device_info.hpp"

#include <string>
#include <vector>

#include "plugin/gfx_property_utils.hpp"
#include "runtime/gfx_backend_caps.hpp"
#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {
void fill_metal_device_info(GfxDeviceInfo& info, const ov::AnyMap& properties);
void fill_vulkan_device_info(GfxDeviceInfo& info, const ov::AnyMap& properties);
}  // namespace gfx_plugin
}  // namespace ov

namespace ov {
namespace gfx_plugin {
namespace {

void finalize_device_info(GfxDeviceInfo& info) {
    if (info.backend_name.empty()) {
        info.backend_name = backend_to_string(info.backend);
    }
    if (info.device_name.empty()) {
        info.device_name = "GFX";
    }
    if (info.full_name.empty()) {
        if (info.device_name == "GFX") {
            info.full_name = "GFX";
        } else {
            info.full_name = "GFX (" + info.device_name + ")";
        }
    }
    if (info.available_devices.empty()) {
        info.available_devices = {info.device_name};
    }
    if (info.capabilities.empty()) {
        info.capabilities = query_backend_caps(info.backend, ov::AnyMap{}).device_capabilities();
    }
}

}  // namespace

GfxDeviceInfo query_device_info(GpuBackend backend, const ov::AnyMap& properties) {
    GfxDeviceInfo info;
    info.backend = backend;
    info.backend_name = backend_to_string(backend);
    info.device_type = ov::device::Type::INTEGRATED;
    const int device_id = parse_device_id(properties);
    info.device_id = device_id >= 0 ? std::to_string(device_id) : std::string{"0"};
    info.capabilities = query_backend_caps(backend, properties).device_capabilities();

    switch (backend) {
    case GpuBackend::Metal: {
        fill_metal_device_info(info, properties);
        break;
    }
    case GpuBackend::Vulkan: {
        fill_vulkan_device_info(info, properties);
        break;
    }
    default:
        info.device_name = "GFX";
        info.full_name = "GFX";
        break;
    }

    finalize_device_info(info);
    return info;
}

GfxDeviceInfo query_device_info_from_properties(const ov::AnyMap& properties,
                                                bool log_fallback,
                                                const char* log_tag) {
    const auto backend = resolve_backend_kind_from_properties(properties, log_fallback, log_tag);
    return query_device_info(backend, properties);
}

}  // namespace gfx_plugin
}  // namespace ov
