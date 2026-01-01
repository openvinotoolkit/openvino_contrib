// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/gfx_plugin/profiling.hpp"

namespace ov {
namespace gfx_plugin {

inline ProfilingLevel parse_profiling_level(const ov::Any& value) {
    if (value.is<int>()) {
        const int v = value.as<int>();
        if (v <= 0)
            return ProfilingLevel::Off;
        if (v == 1)
            return ProfilingLevel::Standard;
        return ProfilingLevel::Detailed;
    }
    if (value.is<unsigned int>()) {
        const unsigned int v = value.as<unsigned int>();
        if (v == 0)
            return ProfilingLevel::Off;
        if (v == 1)
            return ProfilingLevel::Standard;
        return ProfilingLevel::Detailed;
    }
    if (value.is<bool>()) {
        return value.as<bool>() ? ProfilingLevel::Standard : ProfilingLevel::Off;
    }
    if (value.is<std::string>()) {
        auto s = value.as<std::string>();
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "0" || s == "off" || s == "false") {
            return ProfilingLevel::Off;
        }
        if (s == "1" || s == "standard" || s == "on" || s == "true") {
            return ProfilingLevel::Standard;
        }
        if (s == "2" || s == "detailed" || s == "detail") {
            return ProfilingLevel::Detailed;
        }
    }
    OPENVINO_THROW("Unsupported profiling level type/value");
}

inline std::string escape_json_string(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += ' ';
            } else {
                out += c;
            }
            break;
        }
    }
    return out;
}

inline const char* profiling_level_to_string(ProfilingLevel level) {
    switch (level) {
    case ProfilingLevel::Standard:
        return "standard";
    case ProfilingLevel::Detailed:
        return "detailed";
    case ProfilingLevel::Off:
    default:
        return "off";
    }
}

inline const char* profiling_status_to_string(ov::ProfilingInfo::Status st) {
    switch (st) {
    case ov::ProfilingInfo::Status::EXECUTED:
        return "EXECUTED";
    case ov::ProfilingInfo::Status::NOT_RUN:
        return "NOT_RUN";
    case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
        return "OPTIMIZED_OUT";
    default:
        return "UNKNOWN";
    }
}

inline GfxProfilerConfig make_profiler_config(ProfilingLevel level) {
    GfxProfilerConfig cfg;
    cfg.level = level;
    const bool detailed = (level == ProfilingLevel::Detailed);
    cfg.include_segments = detailed;
    cfg.include_allocations = detailed;
    cfg.include_transfers = detailed;
    return cfg;
}

inline std::string build_profiling_report_json(std::string_view backend,
                                               ProfilingLevel level,
                                               const std::vector<ov::ProfilingInfo>& infos,
                                               std::string_view extended_json = {}) {
    std::ostringstream report;
    report << "{\"backend\":\"" << escape_json_string(backend)
           << "\",\"level\":\"" << profiling_level_to_string(level)
           << "\",\"nodes\":[";
    for (size_t i = 0; i < infos.size(); ++i) {
        const auto& info = infos[i];
        if (i)
            report << ",";
        report << "{\"name\":\"" << escape_json_string(info.node_name)
               << "\",\"type\":\"" << escape_json_string(info.node_type)
               << "\",\"exec\":\"" << escape_json_string(info.exec_type)
               << "\",\"real_us\":" << info.real_time.count()
               << ",\"cpu_us\":" << info.cpu_time.count()
               << ",\"status\":\"" << profiling_status_to_string(info.status) << "\"}";
    }
    report << "]";
    if (!extended_json.empty()) {
        report << ",\"extended\":" << extended_json;
    }
    report << "}";
    return report.str();
}

}  // namespace gfx_plugin
}  // namespace ov
