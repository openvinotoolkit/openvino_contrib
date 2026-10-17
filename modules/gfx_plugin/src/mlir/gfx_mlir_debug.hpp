// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdlib>
#include <string>

#include "openvino/util/common_util.hpp"

namespace ov {
namespace gfx_plugin {

inline bool gfx_mlir_debug_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("GFX_MLIR_DEBUG");
        if (!env || !*env) {
            return false;
        }
        std::string value = ov::util::to_lower(env);
        return value == "1" || value == "true" || value == "yes" || value == "on";
    }();
    return enabled;
}

}  // namespace gfx_plugin
}  // namespace ov
