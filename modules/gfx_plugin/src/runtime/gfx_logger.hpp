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

inline GfxLogStream gfx_log_stream(GfxLogLevel level, const char* category) {
    return make_gfx_log_stream(level, category);
}

inline GfxLogStream gfx_log_error(const char* category) {
    return make_gfx_log_stream(GfxLogLevel::Error, category);
}

inline GfxLogStream gfx_log_warn(const char* category) {
    return make_gfx_log_stream(GfxLogLevel::Warn, category);
}

inline GfxLogStream gfx_log_info(const char* category) {
    return make_gfx_log_stream(GfxLogLevel::Info, category);
}

inline GfxLogStream gfx_log_debug(const char* category) {
    return make_gfx_log_stream(GfxLogLevel::Debug, category);
}

inline GfxLogStream gfx_log_trace(const char* category) {
    return make_gfx_log_stream(GfxLogLevel::Trace, category);
}

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

}  // namespace gfx_plugin
}  // namespace ov
