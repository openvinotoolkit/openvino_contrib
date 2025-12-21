// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>
#include <string>
#include <sstream>

namespace ov {
namespace gfx_plugin {

enum class MetalLogLevel {
    Error = 0,
    Warn,
    Info,
    Debug,
    Trace
};

struct MetalLogConfig {
    MetalLogLevel level;
    bool test_debug;
};

// Returns cached configuration derived from environment variables.
const MetalLogConfig& metal_log_config();

void metal_log(MetalLogLevel level, const char* category, const std::string& msg);

bool metal_log_debug_enabled();
bool metal_log_trace_enabled();

class MetalLogStream {
public:
    MetalLogStream(MetalLogLevel level, const char* category);
    ~MetalLogStream();

    template <typename T>
    MetalLogStream& operator<<(const T& v) {
        m_stream << v;
        return *this;
    }

private:
    MetalLogLevel m_level;
    const char* m_category;
    std::ostringstream m_stream;
};

MetalLogStream make_metal_log_stream(MetalLogLevel level, const char* category);

#define GFX_LOGSTREAM_ERROR(cat) ::ov::gfx_plugin::make_metal_log_stream(::ov::gfx_plugin::MetalLogLevel::Error, cat)
#define GFX_LOGSTREAM_WARN(cat)  ::ov::gfx_plugin::make_metal_log_stream(::ov::gfx_plugin::MetalLogLevel::Warn,  cat)
#define GFX_LOGSTREAM_INFO(cat)  ::ov::gfx_plugin::make_metal_log_stream(::ov::gfx_plugin::MetalLogLevel::Info,  cat)
#define GFX_LOGSTREAM_DEBUG(cat) ::ov::gfx_plugin::make_metal_log_stream(::ov::gfx_plugin::MetalLogLevel::Debug, cat)
#define GFX_LOGSTREAM_TRACE(cat) ::ov::gfx_plugin::make_metal_log_stream(::ov::gfx_plugin::MetalLogLevel::Trace, cat)

inline const char* metal_log_level_str(MetalLogLevel lvl) {
    switch (lvl) {
        case MetalLogLevel::Error: return "ERR";
        case MetalLogLevel::Warn:  return "WRN";
        case MetalLogLevel::Info:  return "INF";
        case MetalLogLevel::Debug: return "DBG";
        case MetalLogLevel::Trace: return "TRC";
    }
    return "UNK";
}

#define GFX_LOG_STREAM(level, cat, stream_expr)                     \
    do {                                                              \
        std::ostringstream _metal_log_ss;                             \
        _metal_log_ss << stream_expr;                                 \
        ::ov::gfx_plugin::metal_log(level, cat, _metal_log_ss.str()); \
    } while (0)

#define GFX_LOG_ERROR(cat, msg) GFX_LOG_STREAM(::ov::gfx_plugin::MetalLogLevel::Error, cat, msg)
#define GFX_LOG_WARN(cat, msg)  GFX_LOG_STREAM(::ov::gfx_plugin::MetalLogLevel::Warn,  cat, msg)
#define GFX_LOG_INFO(cat, msg)  GFX_LOG_STREAM(::ov::gfx_plugin::MetalLogLevel::Info,  cat, msg)
#define GFX_LOG_DEBUG(cat, msg) GFX_LOG_STREAM(::ov::gfx_plugin::MetalLogLevel::Debug, cat, msg)
#define GFX_LOG_TRACE(cat, msg) GFX_LOG_STREAM(::ov::gfx_plugin::MetalLogLevel::Trace, cat, msg)

}  // namespace gfx_plugin
}  // namespace ov
