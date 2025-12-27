// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <mutex>
#include <sstream>
#include <string>

namespace ov {
namespace gfx_plugin {

enum class GfxLogLevel {
    Error = 0,
    Warn,
    Info,
    Debug,
    Trace
};

struct GfxLogConfig {
    GfxLogLevel level;
    bool test_debug;
};

// Returns cached configuration derived from environment variables.
const GfxLogConfig& gfx_log_config();

void gfx_log(GfxLogLevel level, const char* category, const std::string& msg);

bool gfx_log_debug_enabled();
bool gfx_log_trace_enabled();

class GfxLogStream {
public:
    GfxLogStream(GfxLogLevel level, const char* category);
    ~GfxLogStream();

    template <typename T>
    GfxLogStream& operator<<(const T& v) {
        m_stream << v;
        return *this;
    }

private:
    GfxLogLevel m_level;
    const char* m_category;
    std::ostringstream m_stream;
};

GfxLogStream make_gfx_log_stream(GfxLogLevel level, const char* category);

#define GFX_LOGSTREAM_ERROR(cat) ::ov::gfx_plugin::make_gfx_log_stream(::ov::gfx_plugin::GfxLogLevel::Error, cat)
#define GFX_LOGSTREAM_WARN(cat)  ::ov::gfx_plugin::make_gfx_log_stream(::ov::gfx_plugin::GfxLogLevel::Warn,  cat)
#define GFX_LOGSTREAM_INFO(cat)  ::ov::gfx_plugin::make_gfx_log_stream(::ov::gfx_plugin::GfxLogLevel::Info,  cat)
#define GFX_LOGSTREAM_DEBUG(cat) ::ov::gfx_plugin::make_gfx_log_stream(::ov::gfx_plugin::GfxLogLevel::Debug, cat)
#define GFX_LOGSTREAM_TRACE(cat) ::ov::gfx_plugin::make_gfx_log_stream(::ov::gfx_plugin::GfxLogLevel::Trace, cat)

inline const char* gfx_log_level_str(GfxLogLevel lvl) {
    switch (lvl) {
        case GfxLogLevel::Error: return "ERR";
        case GfxLogLevel::Warn:  return "WRN";
        case GfxLogLevel::Info:  return "INF";
        case GfxLogLevel::Debug: return "DBG";
        case GfxLogLevel::Trace: return "TRC";
    }
    return "UNK";
}

#define GFX_LOG_STREAM(level, cat, stream_expr)                  \
    do {                                                          \
        std::ostringstream _gfx_log_ss;                           \
        _gfx_log_ss << stream_expr;                               \
        ::ov::gfx_plugin::gfx_log(level, cat, _gfx_log_ss.str()); \
    } while (0)

#define GFX_LOG_ERROR(cat, msg) GFX_LOG_STREAM(::ov::gfx_plugin::GfxLogLevel::Error, cat, msg)
#define GFX_LOG_WARN(cat, msg)  GFX_LOG_STREAM(::ov::gfx_plugin::GfxLogLevel::Warn,  cat, msg)
#define GFX_LOG_INFO(cat, msg)  GFX_LOG_STREAM(::ov::gfx_plugin::GfxLogLevel::Info,  cat, msg)
#define GFX_LOG_DEBUG(cat, msg) GFX_LOG_STREAM(::ov::gfx_plugin::GfxLogLevel::Debug, cat, msg)
#define GFX_LOG_TRACE(cat, msg) GFX_LOG_STREAM(::ov::gfx_plugin::GfxLogLevel::Trace, cat, msg)

}  // namespace gfx_plugin
}  // namespace ov
