// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {

using MetalLogLevel = GfxLogLevel;
using MetalLogConfig = GfxLogConfig;
using MetalLogStream = GfxLogStream;

inline const MetalLogConfig& metal_log_config() {
    return gfx_log_config();
}

inline void metal_log(MetalLogLevel level, const char* category, const std::string& msg) {
    gfx_log(level, category, msg);
}

inline bool metal_log_debug_enabled() {
    return gfx_log_debug_enabled();
}

inline bool metal_log_trace_enabled() {
    return gfx_log_trace_enabled();
}

inline MetalLogStream make_metal_log_stream(MetalLogLevel level, const char* category) {
    return make_gfx_log_stream(level, category);
}

inline const char* metal_log_level_str(MetalLogLevel lvl) {
    return gfx_log_level_str(lvl);
}

}  // namespace gfx_plugin
}  // namespace ov
