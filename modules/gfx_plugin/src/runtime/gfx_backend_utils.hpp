// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "openvino/core/except.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

inline constexpr const char* kBackendMetal = "metal";
inline constexpr const char* kBackendVulkan = "vulkan";

inline const char* backend_to_string(GpuBackend backend) {
    switch (backend) {
    case GpuBackend::Metal:
        return kBackendMetal;
    case GpuBackend::Vulkan:
        return kBackendVulkan;
    default:
        return "unknown";
    }
}

inline GpuBackend parse_backend_kind(const std::string& value) {
    const auto backend = ov::util::to_lower(value);
    if (backend == kBackendMetal) {
        return GpuBackend::Metal;
    }
    if (backend == kBackendVulkan) {
        return GpuBackend::Vulkan;
    }
    OPENVINO_THROW("Unsupported GFX_BACKEND value: ", value, ". Expected 'metal' or 'vulkan'.");
}

inline GpuBackend default_backend_kind() {
    return parse_backend_kind(kGfxDefaultBackend);
}

inline bool backend_supported(GpuBackend backend) {
    switch (backend) {
    case GpuBackend::Metal:
        return kGfxBackendMetalAvailable;
    case GpuBackend::Vulkan:
        return kGfxBackendVulkanAvailable;
    default:
        return false;
    }
}

inline GpuBackend fallback_backend_kind() {
    if (backend_supported(GpuBackend::Metal)) {
        return GpuBackend::Metal;
    }
    if (backend_supported(GpuBackend::Vulkan)) {
        return GpuBackend::Vulkan;
    }
    return default_backend_kind();
}

}  // namespace gfx_plugin
}  // namespace ov
