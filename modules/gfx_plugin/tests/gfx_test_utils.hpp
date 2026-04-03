// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace utils {

inline void try_register_gfx_plugin(ov::Core& core) {
    const char* env_path = std::getenv("GFX_PLUGIN_PATH");
    if (!env_path || !*env_path) {
        return;
    }
    target_device = "GFX";
    target_plugin_name = env_path;
    try {
        core.register_plugin(env_path, "GFX");
    } catch (...) {
        // Ignore registration errors (e.g. already registered).
    }
}

inline void configure_gfx_plugin_cache_from_env() {
    static bool configured = false;
    const char* env_path = std::getenv("GFX_PLUGIN_PATH");
    if (!env_path || !*env_path) {
        return;
    }
    target_device = "GFX";
    target_plugin_name = env_path;
    if (!configured) {
        PluginCache::get().reset();
        configured = true;
    }
}

inline std::string gfx_backend_name() {
    try {
        ov::Core core;
        try_register_gfx_plugin(core);
        auto any = core.get_property("GFX", "GFX_BACKEND");
        return any.as<std::string>();
    } catch (...) {
        return {};
    }
}

inline bool gfx_backend_is_metal() {
    return gfx_backend_name() == "metal";
}

inline bool gfx_backend_is_vulkan() {
    return gfx_backend_name() == "vulkan";
}

inline bool device_available(const std::string& name) {
    try {
        ov::Core core;
        try_register_gfx_plugin(core);
        const auto devices = core.get_available_devices();
        return std::find(devices.begin(), devices.end(), name) != devices.end();
    } catch (...) {
        return false;
    }
}

inline std::string require_gfx_backend() {
    const auto backend = gfx_backend_name();
    EXPECT_FALSE(backend.empty()) << "GFX backend not available";
    return backend;
}

inline void require_template_device() {
    EXPECT_TRUE(device_available("TEMPLATE")) << "TEMPLATE reference device not available";
}

}  // namespace utils
}  // namespace test
}  // namespace ov
