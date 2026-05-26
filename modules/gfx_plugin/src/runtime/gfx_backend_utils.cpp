// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {

const char* backend_to_string(GpuBackend backend) {
    switch (backend) {
    case GpuBackend::Metal:
        return kBackendMetal;
    case GpuBackend::OpenCL:
        return kBackendOpenCL;
    default:
        return "unknown";
    }
}

GpuBackend parse_backend_kind(const std::string& value) {
    const auto backend = ov::util::to_lower(value);
    if (backend == kBackendMetal) {
        return GpuBackend::Metal;
    }
    if (backend == kBackendOpenCL) {
        return GpuBackend::OpenCL;
    }
    OPENVINO_THROW("Unsupported GFX_BACKEND value: ", value, ". Expected 'metal' or 'opencl'.");
}

GpuBackend default_backend_kind() {
    return parse_backend_kind(kGfxDefaultBackend);
}

bool backend_supported(GpuBackend backend) {
    switch (backend) {
    case GpuBackend::Metal:
        return kGfxBackendMetalAvailable;
    case GpuBackend::OpenCL:
        return kGfxBackendOpenCLAvailable;
    default:
        return false;
    }
}

}  // namespace gfx_plugin
}  // namespace ov
