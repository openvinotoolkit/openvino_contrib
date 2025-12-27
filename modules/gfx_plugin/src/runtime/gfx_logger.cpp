// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_logger.hpp"

#include <cstdlib>
#include <iostream>

namespace ov {
namespace gfx_plugin {

namespace {
GfxLogLevel env_level() {
    if (const char* trace = std::getenv("OV_GFX_TRACE")) {
        if (std::string(trace) != "0")
            return GfxLogLevel::Trace;
    }
    if (const char* dbg = std::getenv("OV_GFX_DEBUG")) {
        if (std::string(dbg) != "0")
            return GfxLogLevel::Debug;
    }
    return GfxLogLevel::Error;
}
}  // namespace

namespace {
GfxLogConfig make_config_from_env() {
    GfxLogConfig c;
    c.level = env_level();
    const char* test_dbg = std::getenv("OV_GFX_TEST_DEBUG");
    c.test_debug = test_dbg && std::string(test_dbg) != "0";
    return c;
}
}  // namespace

const GfxLogConfig& gfx_log_config() {
    static GfxLogConfig cfg;
    cfg = make_config_from_env();
    return cfg;
}

void gfx_log(GfxLogLevel level, const char* category, const std::string& msg) {
    const auto& cfg = gfx_log_config();
    if (level > cfg.level && !(cfg.test_debug && level <= GfxLogLevel::Debug))
        return;
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cerr << "[GFX][" << category << "][" << gfx_log_level_str(level) << "] "
              << msg << std::endl;
}

bool gfx_log_debug_enabled() {
    const auto& cfg = gfx_log_config();
    return cfg.level >= GfxLogLevel::Debug || cfg.test_debug;
}

bool gfx_log_trace_enabled() {
    return gfx_log_config().level >= GfxLogLevel::Trace;
}

GfxLogStream::GfxLogStream(GfxLogLevel level, const char* category)
    : m_level(level), m_category(category) {}

GfxLogStream::~GfxLogStream() {
    gfx_log(m_level, m_category, m_stream.str());
}

GfxLogStream make_gfx_log_stream(GfxLogLevel level, const char* category) {
    return GfxLogStream(level, category);
}

}  // namespace gfx_plugin
}  // namespace ov
