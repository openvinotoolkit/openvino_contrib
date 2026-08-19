// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "gfx_plugin_runtime_path.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace utils {

inline void try_register_gfx_plugin(ov::Core& core) {
    const char* env_path = gfx_plugin_runtime_path();
    if (!env_path || !*env_path) {
        return;
    }
    target_device = "GFX";
    target_plugin_name = env_path;
    (void)register_gfx_plugin_runtime_path(core);
}

inline void configure_gfx_plugin_cache_from_env() {
    static bool configured = false;
    const char* env_path = gfx_plugin_runtime_path();
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

inline bool gfx_backend_is_opencl() {
    return gfx_backend_name() == "opencl";
}

inline bool device_available(const std::string& name) {
    try {
        ov::Core core;
        try_register_gfx_plugin(core);
        (void)core.get_property(name, ov::supported_properties);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool core_has_device(const ov::Core& core, const std::string& name) {
    try {
        (void)core.get_property(name, ov::supported_properties);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool ensure_template_plugin(ov::Core& core) {
    if (core_has_device(core, "TEMPLATE")) {
        return true;
    }
    ov::test::utils::register_template_plugin(core);
    return core_has_device(core, "TEMPLATE");
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
