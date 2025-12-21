// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/metal_logger.hpp"

#include <cstdlib>
#include <iostream>

namespace ov {
namespace gfx_plugin {

namespace {
MetalLogLevel env_level() {
    if (const char* trace = std::getenv("OV_GFX_TRACE")) {
        if (std::string(trace) != "0")
            return MetalLogLevel::Trace;
    }
    if (const char* dbg = std::getenv("OV_GFX_DEBUG")) {
        if (std::string(dbg) != "0")
            return MetalLogLevel::Debug;
    }
    return MetalLogLevel::Error;
}
}  // namespace

const MetalLogConfig& metal_log_config() {
    static const MetalLogConfig cfg = []() {
        MetalLogConfig c;
        c.level = env_level();
        const char* test_dbg = std::getenv("OV_GFX_TEST_DEBUG");
        c.test_debug = test_dbg && std::string(test_dbg) != "0";
        return c;
    }();
    return cfg;
}

void metal_log(MetalLogLevel level, const char* category, const std::string& msg) {
    const auto& cfg = metal_log_config();
    if (level > cfg.level && !(cfg.test_debug && level <= MetalLogLevel::Debug))
        return;
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cerr << "[GFX][" << category << "][" << metal_log_level_str(level) << "] "
              << msg << std::endl;
}

bool metal_log_debug_enabled() {
    const auto& cfg = metal_log_config();
    return cfg.level >= MetalLogLevel::Debug || cfg.test_debug;
}

bool metal_log_trace_enabled() {
    return metal_log_config().level >= MetalLogLevel::Trace;
}

MetalLogStream::MetalLogStream(MetalLogLevel level, const char* category)
    : m_level(level), m_category(category) {}

MetalLogStream::~MetalLogStream() {
    metal_log(m_level, m_category, m_stream.str());
}

MetalLogStream make_metal_log_stream(MetalLogLevel level, const char* category) {
    return MetalLogStream(level, category);
}

}  // namespace gfx_plugin
}  // namespace ov
