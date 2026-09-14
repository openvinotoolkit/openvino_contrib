// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
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

inline std::string_view extract_json_array_field(std::string_view json, std::string_view field_name) {
    const std::string needle = "\"" + std::string(field_name) + "\":[";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string_view::npos) {
        return {};
    }
    const size_t array_start = key_pos + needle.size() - 1;
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = array_start; i < json.size(); ++i) {
        const char c = json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '[') {
            depth += 1;
            continue;
        }
        if (c == ']') {
            depth -= 1;
            if (depth == 0) {
                return json.substr(array_start, i - array_start + 1);
            }
        }
    }
    return {};
}

inline std::vector<std::string_view> split_top_level_json_array_objects(std::string_view array_json) {
    std::vector<std::string_view> objects;
    if (array_json.size() < 2 || array_json.front() != '[' || array_json.back() != ']') {
        return objects;
    }
    bool in_string = false;
    bool escaped = false;
    size_t object_start = std::string_view::npos;
    size_t depth = 0;
    for (size_t i = 1; i + 1 < array_json.size(); ++i) {
        const char c = array_json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            if (depth == 0) {
                object_start = i;
            }
            depth += 1;
            continue;
        }
        if (c == '}') {
            if (depth == 0) {
                continue;
            }
            depth -= 1;
            if (depth == 0 && object_start != std::string_view::npos) {
                objects.push_back(array_json.substr(object_start, i - object_start + 1));
                object_start = std::string_view::npos;
            }
        }
    }
    return objects;
}

inline uint64_t extract_json_uint_field(std::string_view object_json, std::string_view field_name, uint64_t fallback = 0) {
    const std::string needle = "\"" + std::string(field_name) + "\":";
    const size_t pos = object_json.find(needle);
    if (pos == std::string_view::npos) {
        return fallback;
    }
    size_t begin = pos + needle.size();
    while (begin < object_json.size() && std::isspace(static_cast<unsigned char>(object_json[begin]))) {
        ++begin;
    }
    size_t end = begin;
    while (end < object_json.size() && std::isdigit(static_cast<unsigned char>(object_json[end]))) {
        ++end;
    }
    if (end == begin) {
        return fallback;
    }
    return static_cast<uint64_t>(std::strtoull(std::string(object_json.substr(begin, end - begin)).c_str(), nullptr, 10));
}

inline std::string shift_trace_event_ts(std::string_view event_json, uint64_t offset_us) {
    if (offset_us == 0) {
        return std::string(event_json);
    }
    const std::string needle = "\"ts\":";
    const size_t pos = event_json.find(needle);
    if (pos == std::string_view::npos) {
        return std::string(event_json);
    }
    size_t begin = pos + needle.size();
    while (begin < event_json.size() && std::isspace(static_cast<unsigned char>(event_json[begin]))) {
        ++begin;
    }
    size_t end = begin;
    while (end < event_json.size() && std::isdigit(static_cast<unsigned char>(event_json[end]))) {
        ++end;
    }
    if (end == begin) {
        return std::string(event_json);
    }
    const uint64_t ts = static_cast<uint64_t>(std::strtoull(std::string(event_json.substr(begin, end - begin)).c_str(), nullptr, 10));
    std::ostringstream out;
    out << event_json.substr(0, begin) << (ts + offset_us) << event_json.substr(end);
    return out.str();
}

inline std::string build_trace_export_json(std::string_view compile_json, std::string_view extended_json) {
    const auto compile_events = extract_json_array_field(compile_json, "traceEvents");
    const auto extended_events = extract_json_array_field(extended_json, "traceEvents");
    if (compile_events.empty() && extended_events.empty()) {
        return {};
    }

    const auto compile_objects = split_top_level_json_array_objects(compile_events);
    const auto extended_objects = split_top_level_json_array_objects(extended_events);
    uint64_t extended_offset_us = 0;
    for (const auto& event : compile_objects) {
        const uint64_t ts = extract_json_uint_field(event, "ts");
        const uint64_t dur = extract_json_uint_field(event, "dur");
        extended_offset_us = std::max(extended_offset_us, ts + dur);
    }

    std::ostringstream trace;
    trace << "{\"traceEvents\":[";
    bool first = true;
    const auto append_events = [&](const std::vector<std::string_view>& events, uint64_t ts_offset) {
        for (const auto& event : events) {
            if (!first) {
                trace << ',';
            }
            first = false;
            trace << shift_trace_event_ts(event, ts_offset);
        }
    };
    append_events(compile_objects, 0);
    append_events(extended_objects, extended_offset_us);
    trace << "],\"displayTimeUnit\":\"ms\"}";
    return trace.str();
}

inline void maybe_write_trace_export_file(std::string_view compile_json, std::string_view extended_json) {
    const char* path = std::getenv("OV_GFX_PROFILE_TRACE_FILE");
    if (!path || path[0] == '\0') {
        return;
    }
    const auto trace_json = build_trace_export_json(compile_json, extended_json);
    if (trace_json.empty()) {
        return;
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return;
    }
    out << trace_json;
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
                                               std::string_view extended_json = {},
                                               std::string_view compile_json = {}) {
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
    if (!compile_json.empty()) {
        report << ",\"compile\":" << compile_json;
    }
    if (!extended_json.empty()) {
        report << ",\"extended\":" << extended_json;
    }
    report << "}";
    const auto json = report.str();
    maybe_write_trace_export_file(compile_json, extended_json);
    return json;
}

}  // namespace gfx_plugin
}  // namespace ov
