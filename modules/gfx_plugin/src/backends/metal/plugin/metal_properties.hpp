// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/gfx_plugin/properties.hpp"

namespace ov {
namespace gfx_plugin {

// Metal-specific keys.
constexpr const char* kMtlBufferProperty = "MTL_BUFFER";
constexpr const char* kStorageModeProperty = "STORAGE_MODE";

}  // namespace gfx_plugin
}  // namespace ov
