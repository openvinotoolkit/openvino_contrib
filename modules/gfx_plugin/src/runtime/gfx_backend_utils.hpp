// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "common/gpu_backend.hpp"

namespace ov {
namespace gfx_plugin {

GpuBackend parse_backend_kind(const std::string& value);
GpuBackend default_backend_kind();
bool backend_supported(GpuBackend backend);

}  // namespace gfx_plugin
}  // namespace ov
