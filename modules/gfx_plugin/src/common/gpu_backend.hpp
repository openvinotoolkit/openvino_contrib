// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace ov {
namespace gfx_plugin {

inline constexpr const char* kBackendMetal = "metal";
inline constexpr const char* kBackendOpenCL = "opencl";

enum class GpuBackend { Unknown = 255, Metal = 0, OpenCL = 1 };

inline const char* backend_to_string(GpuBackend backend) {
    switch (backend) {
    case GpuBackend::Metal:
        return kBackendMetal;
    case GpuBackend::OpenCL:
        return kBackendOpenCL;
    case GpuBackend::Unknown:
    default:
        return "unknown";
    }
}

inline bool backend_known(GpuBackend backend) {
    return backend == GpuBackend::Metal || backend == GpuBackend::OpenCL;
}

}  // namespace gfx_plugin
}  // namespace ov
