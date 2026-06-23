// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace utils {

inline const char* gfx_plugin_runtime_path() {
    const char* env_path = std::getenv("GFX_PLUGIN_PATH");
    if (env_path && *env_path) {
        return env_path;
    }
    env_path = std::getenv("OV_GFX_PLUGIN_PATH");
    if (env_path && *env_path) {
        return env_path;
    }
    return nullptr;
}

inline bool register_gfx_plugin_runtime_path(ov::Core& core, std::string* error = nullptr) {
    if (error) {
        error->clear();
    }
    const char* path = gfx_plugin_runtime_path();
    if (!path) {
        return true;
    }
    try {
        core.register_plugin(path, "GFX");
        return true;
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find("already registered") != std::string::npos) {
            return true;
        }
        if (error) {
            *error = std::string("GFX plugin unavailable: ") + e.what();
        }
        return false;
    }
}

inline bool ensure_gfx_plugin_available(ov::Core& core, std::string* error = nullptr) {
    if (!register_gfx_plugin_runtime_path(core, error)) {
        return false;
    }
    try {
        const auto backend = core.get_property("GFX", "GFX_BACKEND").as<std::string>();
        if (!backend.empty()) {
            return true;
        }
        if (error) {
            *error = "GFX backend not available";
        }
        return false;
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("GFX backend property unavailable: ") + e.what();
        }
        return false;
    }
}

inline void ensure_gfx_plugin_available_or_throw(ov::Core& core) {
    std::string error;
    if (!ensure_gfx_plugin_available(core, &error)) {
        throw std::runtime_error(error.empty() ? "GFX plugin unavailable" : error);
    }
}

}  // namespace utils
}  // namespace test
}  // namespace ov
