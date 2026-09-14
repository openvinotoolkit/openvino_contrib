// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace gfx_plugin {

std::vector<ov::PropertyName> gfx_plugin_supported_properties();
std::vector<ov::PropertyName> gfx_internal_supported_properties();
std::vector<ov::PropertyName> gfx_caching_properties();

std::vector<ov::PropertyName> gfx_compiled_model_default_ro_properties();
std::vector<ov::PropertyName> gfx_compiled_model_supported_properties();

}  // namespace gfx_plugin
}  // namespace ov
