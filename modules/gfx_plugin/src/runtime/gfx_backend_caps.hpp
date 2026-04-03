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

struct GfxBackendCaps {
    GpuBackend backend = GpuBackend::Metal;
    bool supports_fp32 = true;
    bool supports_fp16 = true;
    bool supports_int8 = false;
    bool supports_export_import = true;

    std::vector<std::string> device_capabilities() const;
};

GfxBackendCaps query_backend_caps(GpuBackend backend, const ov::AnyMap& properties);

}  // namespace gfx_plugin
}  // namespace ov
