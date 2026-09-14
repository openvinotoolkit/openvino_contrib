// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace ov {
namespace gfx_plugin {

enum class GpuDeviceFamily {
    Generic,
    Apple,
    QualcommAdreno,
    BroadcomV3D,
};

inline const char* gpu_device_family_name(GpuDeviceFamily family) {
    switch (family) {
    case GpuDeviceFamily::Apple:
        return "apple";
    case GpuDeviceFamily::QualcommAdreno:
        return "adreno";
    case GpuDeviceFamily::BroadcomV3D:
        return "broadcom_v3d";
    case GpuDeviceFamily::Generic:
    default:
        return "generic";
    }
}

}  // namespace gfx_plugin
}  // namespace ov
