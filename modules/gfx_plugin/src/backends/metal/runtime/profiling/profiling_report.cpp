// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/profiling/profiling_report.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

namespace {
std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

const char* level_to_string(ProfilingLevel level) {
    switch (level) {
    case ProfilingLevel::Off:
        return "off";
    case ProfilingLevel::Standard:
        return "standard";
    case ProfilingLevel::Detailed:
        return "detailed";
    default:
        return "unknown";
    }
}
}  // namespace

std::string MetalProfilingReport::to_json() const {
    std::ostringstream oss;
    oss << '{';
    oss << "\"level\":\"" << level_to_string(level) << "\",";
    oss << "\"counters_supported\":" << (counters_supported ? "true" : "false") << ',';
    oss << "\"counters_used\":" << (counters_used ? "true" : "false") << ',';
    oss << "\"total_gpu_us\":" << total_gpu_us << ',';
    oss << "\"total_cpu_us\":" << total_cpu_us << ',';
    oss << "\"total_wall_us\":" << total_wall_us << ',';
    oss << "\"total_h2d_bytes\":" << total_h2d_bytes << ',';
    oss << "\"total_d2h_bytes\":" << total_d2h_bytes << ',';

    oss << "\"memory_stats\":{";
    oss << "\"bytes_allocated_total\":" << memory_stats.bytes_allocated_total << ',';
    oss << "\"bytes_in_freelist\":" << memory_stats.bytes_in_freelist << ',';
    oss << "\"bytes_live_transient\":" << memory_stats.bytes_live_transient << ',';
    oss << "\"bytes_live_handles\":" << memory_stats.bytes_live_handles << ',';
    oss << "\"bytes_persistent\":" << memory_stats.bytes_persistent << ',';
    oss << "\"peak_live\":" << memory_stats.peak_live << ',';
    oss << "\"num_alloc_calls\":" << memory_stats.num_alloc_calls << ',';
    oss << "\"num_reuse_hits\":" << memory_stats.num_reuse_hits << ',';
    oss << "\"h2d_bytes\":" << memory_stats.h2d_bytes << ',';
    oss << "\"d2h_bytes\":" << memory_stats.d2h_bytes << ',';
    oss << "\"h2d_ops\":" << memory_stats.h2d_ops << ',';
    oss << "\"d2h_ops\":" << memory_stats.d2h_ops;
    oss << "},";

    oss << "\"nodes\":[";
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        if (i)
            oss << ',';
        oss << '{';
        oss << "\"node_id\":" << n.node_id << ',';
        oss << "\"node_name\":\"" << escape_json(n.node_name) << "\",";
        oss << "\"node_type\":\"" << escape_json(n.node_type) << "\",";
        oss << "\"exec_type\":\"" << escape_json(n.exec_type) << "\",";
        oss << "\"gpu_us\":" << n.gpu_us << ',';
        oss << "\"cpu_us\":" << n.cpu_us << ',';
        oss << "\"dispatches\":" << n.dispatches;
        oss << '}';
    }
    oss << "],";

    oss << "\"transfers\":[";
    for (size_t i = 0; i < transfers.size(); ++i) {
        const auto& t = transfers[i];
        if (i)
            oss << ',';
        oss << '{';
        oss << "\"tag\":\"" << escape_json(t.tag) << "\",";
        oss << "\"bytes\":" << t.bytes << ',';
        oss << "\"cpu_us\":" << t.cpu_us << ',';
        oss << "\"gpu_us\":" << t.gpu_us << ',';
        oss << "\"h2d\":" << (t.h2d ? "true" : "false");
        oss << '}';
    }
    oss << "],";

    oss << "\"allocations\":[";
    for (size_t i = 0; i < allocations.size(); ++i) {
        const auto& a = allocations[i];
        if (i)
            oss << ',';
        oss << '{';
        oss << "\"tag\":\"" << escape_json(a.tag) << "\",";
        oss << "\"bytes\":" << a.bytes << ',';
        oss << "\"cpu_us\":" << a.cpu_us << ',';
        oss << "\"reused\":" << (a.reused ? "true" : "false");
        oss << '}';
    }
    oss << "],";

    oss << "\"alloc_summary\":[";
    for (size_t i = 0; i < alloc_summary.size(); ++i) {
        const auto& a = alloc_summary[i];
        if (i)
            oss << ',';
        oss << '{';
        oss << "\"tag\":\"" << escape_json(a.tag) << "\",";
        oss << "\"bytes\":" << a.bytes << ',';
        oss << "\"alloc_count\":" << a.alloc_count << ',';
        oss << "\"reuse_count\":" << a.reuse_count << ',';
        oss << "\"cpu_us\":" << a.cpu_us;
        oss << '}';
    }
    oss << "],";

    oss << "\"segments\":[";
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& s = segments[i];
        if (i)
            oss << ',';
        oss << '{';
        oss << "\"tag\":\"" << escape_json(s.tag) << "\",";
        oss << "\"gpu_us\":" << s.gpu_us << ',';
        oss << "\"cpu_us\":" << s.cpu_us << ',';
        oss << "\"dispatches\":" << s.dispatches;
        oss << '}';
    }
    oss << ']';

    oss << '}';
    return oss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
