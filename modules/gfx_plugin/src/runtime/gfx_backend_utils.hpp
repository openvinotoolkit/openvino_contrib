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

const char* backend_to_string(GpuBackend backend);
GpuBackend parse_backend_kind(const std::string& value);
GpuBackend default_backend_kind();
bool backend_supported(GpuBackend backend);
GpuBackend fallback_backend_kind();

}  // namespace gfx_plugin
}  // namespace ov
