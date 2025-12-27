// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_backend_caps.hpp"

namespace ov {
namespace gfx_plugin {

std::vector<std::string> GfxBackendCaps::device_capabilities() const {
    std::vector<std::string> caps;
    if (supports_fp32) {
        caps.push_back(ov::device::capability::FP32);
    }
    if (supports_fp16) {
        caps.push_back(ov::device::capability::FP16);
    }
    if (supports_int8) {
        caps.push_back(ov::device::capability::INT8);
    }
    if (supports_export_import) {
        caps.push_back(ov::device::capability::EXPORT_IMPORT);
    }
    return caps;
}

GfxBackendCaps query_backend_caps(GpuBackend backend, const ov::AnyMap& /*properties*/) {
    GfxBackendCaps caps;
    caps.backend = backend;
    // For now we advertise FP32/FP16 and export/import across backends.
    // Backend-specific refinement can be added here without changing callers.
    return caps;
}

}  // namespace gfx_plugin
}  // namespace ov
