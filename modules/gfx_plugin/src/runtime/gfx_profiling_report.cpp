// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_profiling_report.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <sstream>
#include <thread>

#if defined(__APPLE__)
#    include <os/signpost.h>
#endif

namespace ov {
namespace gfx_plugin {

namespace {

std::string escape_json(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
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

const char* trace_sink_from_env(ProfilingLevel level) {
    if (level != ProfilingLevel::Detailed) {
        return "";
    }
    const char* value = std::getenv("OV_GFX_PROFILE_TRACE");
    if (!value || value[0] == '\0') {
        return "";
    }
    std::string mode{value};
    std::transform(mode.begin(),
                   mode.end(),
                   mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode == "perfetto" || mode == "trace_event") {
        return "perfetto";
    }
#if defined(__APPLE__)
    if (mode == "signpost" || mode == "os_signpost") {
        return "signpost";
    }
#endif
    return "";
}

uint64_t monotonic_wall_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint64_t current_thread_id_hash() {
    return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

#if defined(__APPLE__)
os_log_t gfx_signpost_log() {
    static os_log_t log = os_log_create("org.openvino.gfx", "profiling");
    return log;
}

void maybe_emit_signpost(std::string_view backend, const GfxProfilingSegmentEntry& segment) {
    os_log_t log = gfx_signpost_log();
    if (!os_signpost_enabled(log)) {
        return;
    }
    os_signpost_event_emit(log,
                           OS_SIGNPOST_ID_EXCLUSIVE,
                           "gfx.segment",
                           "backend=%{public}s phase=%{public}s name=%{public}s cpu_us=%llu gpu_us=%llu inflight=%lld queue=%llu cmd=%llu",
                           std::string(backend).c_str(),
                           segment.phase.c_str(),
                           segment.name.c_str(),
                           static_cast<unsigned long long>(segment.cpu_us),
                           static_cast<unsigned long long>(segment.gpu_us),
                           static_cast<long long>(segment.inflight_slot),
                           static_cast<unsigned long long>(segment.queue_id),
                           static_cast<unsigned long long>(segment.cmd_buffer_id));
}
#endif

struct PhaseSummary {
    std::string phase;
    uint64_t count = 0;
    uint64_t cpu_us = 0;
    uint64_t gpu_us = 0;
    uint64_t dispatches = 0;
    uint64_t bytes_in = 0;
    uint64_t bytes_out = 0;
    uint64_t macs_est = 0;
    uint64_t flops_est = 0;
};

struct RooflineEstimate {
    uint64_t bytes_moved = 0;
    uint64_t flops_est = 0;
    uint64_t macs_est = 0;
    double arithmetic_intensity = 0.0;
    double gpu_tflops = 0.0;
    double cpu_tflops = 0.0;
    double gpu_gbps = 0.0;
    double cpu_gbps = 0.0;
    const char* dominant_regime = "unknown";
    const char* confidence = "low";
};

struct TransferSummary {
    const char* direction = "";
    uint64_t count = 0;
    uint64_t bytes = 0;
    uint64_t cpu_us = 0;
    uint64_t gpu_us = 0;
};

struct Insight {
    const char* category = "";
    const char* severity = "";
    std::string message;
};

struct TraceEventJson {
    uint64_t ts = 0;
    uint64_t order = 0;
    std::string json;
};

std::vector<PhaseSummary> build_phase_summaries(const GfxProfilingReport& report) {
    std::vector<PhaseSummary> summaries;
    for (const auto& seg : report.segments) {
        auto it = std::find_if(summaries.begin(), summaries.end(), [&](const PhaseSummary& s) { return s.phase == seg.phase; });
        if (it == summaries.end()) {
            summaries.push_back(PhaseSummary{std::string{seg.phase}});
            it = std::prev(summaries.end());
        }
        it->count += 1;
        it->cpu_us += seg.cpu_us;
        it->gpu_us += seg.gpu_us;
        it->dispatches += seg.dispatches;
        it->bytes_in += seg.bytes_in;
        it->bytes_out += seg.bytes_out;
        it->macs_est += seg.macs_est;
        it->flops_est += seg.flops_est;
    }
    return summaries;
}

std::array<TransferSummary, 2> build_transfer_summaries(const GfxProfilingReport& report) {
    std::array<TransferSummary, 2> summaries{{{"h2d"}, {"d2h"}}};
    for (const auto& transfer : report.transfers) {
        auto& summary = transfer.h2d ? summaries[0] : summaries[1];
        summary.count += 1;
        summary.bytes += transfer.bytes;
        summary.cpu_us += transfer.cpu_us;
        summary.gpu_us += transfer.gpu_us;
    }
    return summaries;
}

uint64_t counter_value(const GfxProfilingReport& report, std::string_view name) {
    auto it = std::find_if(report.counters.begin(),
                           report.counters.end(),
                           [&](const GfxProfilingCounterEntry& entry) { return entry.name == name; });
    return it == report.counters.end() ? 0 : it->value;
}

uint64_t phase_cpu_us(const std::vector<PhaseSummary>& summaries, std::string_view phase) {
    auto it = std::find_if(summaries.begin(), summaries.end(), [&](const PhaseSummary& s) { return s.phase == phase; });
    return it == summaries.end() ? 0 : it->cpu_us;
}

uint64_t phase_count(const std::vector<PhaseSummary>& summaries, std::string_view phase) {
    auto it = std::find_if(summaries.begin(), summaries.end(), [&](const PhaseSummary& s) { return s.phase == phase; });
    return it == summaries.end() ? 0 : it->count;
}

double safe_div(double num, double denom) {
    return denom > 0.0 ? (num / denom) : 0.0;
}

RooflineEstimate estimate_roofline(uint64_t flops_est,
                                   uint64_t macs_est,
                                   uint64_t bytes_in,
                                   uint64_t bytes_out,
                                   uint64_t cpu_us,
                                   uint64_t gpu_us) {
    RooflineEstimate estimate;
    estimate.flops_est = flops_est;
    estimate.macs_est = macs_est;
    estimate.bytes_moved = bytes_in + bytes_out;
    estimate.arithmetic_intensity =
        estimate.bytes_moved > 0 ? safe_div(static_cast<double>(estimate.flops_est), static_cast<double>(estimate.bytes_moved)) : 0.0;
    estimate.gpu_tflops =
        gpu_us > 0 ? safe_div(static_cast<double>(estimate.flops_est), static_cast<double>(gpu_us) * 1.0e6) : 0.0;
    estimate.cpu_tflops =
        cpu_us > 0 ? safe_div(static_cast<double>(estimate.flops_est), static_cast<double>(cpu_us) * 1.0e6) : 0.0;
    estimate.gpu_gbps =
        gpu_us > 0 ? safe_div(static_cast<double>(estimate.bytes_moved), static_cast<double>(gpu_us) * 1.0e3) : 0.0;
    estimate.cpu_gbps =
        cpu_us > 0 ? safe_div(static_cast<double>(estimate.bytes_moved), static_cast<double>(cpu_us) * 1.0e3) : 0.0;

    if (estimate.flops_est == 0 || estimate.bytes_moved == 0) {
        estimate.dominant_regime = "unknown";
        estimate.confidence = "low";
        return estimate;
    }

    if (estimate.arithmetic_intensity < 4.0) {
        estimate.dominant_regime = "memory";
        estimate.confidence = "high";
    } else if (estimate.arithmetic_intensity >= 16.0) {
        estimate.dominant_regime = "compute";
        estimate.confidence = "high";
    } else {
        estimate.dominant_regime = "mixed";
        estimate.confidence = "medium";
    }
    return estimate;
}

std::string format_decimal(double value, int precision = 4) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(precision);
    oss << value;
    return oss.str();
}

std::vector<Insight> build_insights(const GfxProfilingReport& report,
                                    const std::vector<PhaseSummary>& phase_summaries,
                                    const std::array<TransferSummary, 2>& transfer_summaries) {
    std::vector<Insight> insights;

    if (!report.counters_used || report.total_gpu_us == 0) {
        insights.push_back({"gpu_timing",
                            "info",
                            "GPU per-op timestamps are unavailable; GPU gaps must be inferred from wall/CPU timings."});
    }

    const uint64_t submit_count =
        std::max<uint64_t>(counter_value(report, "submit_count"), counter_value(report, "vkQueueSubmit_count"));
    if (submit_count > 3) {
        insights.push_back(
            {"submission", "warning", "More than three submit operations were observed in a single infer."});
    }

    const uint64_t wait_cpu = phase_cpu_us(phase_summaries, "wait");
    if (report.total_wall_us > 0 && wait_cpu * 100 >= report.total_wall_us * 20) {
        insights.push_back(
            {"sync", "warning", "Host wait time exceeds 20% of end-to-end wall time; infer is sync-heavy."});
    }

    const uint64_t barrier_count =
        counter_value(report, "cross_submit_barrier_count") + counter_value(report, "barrier_count");
    if (barrier_count > 0) {
        insights.push_back(
            {"barrier", "info", "Explicit barrier activity was recorded; check whether cross-submit barriers are minimal."});
    }

    const uint64_t transfer_cpu = transfer_summaries[0].cpu_us + transfer_summaries[1].cpu_us;
    if (report.total_wall_us > 0 && transfer_cpu * 100 >= report.total_wall_us * 10 &&
        (report.total_h2d_bytes + report.total_d2h_bytes) > 0) {
        insights.push_back({"transfer",
                            "info",
                            "Upload/download CPU overhead is visible in the infer wall time; inspect staging and readback paths."});
    }

    if (report.total_gpu_us > 0 && report.total_cpu_us > report.total_gpu_us * 2) {
        insights.push_back(
            {"cpu_overhead", "info", "Accumulated CPU phase time is more than 2x larger than measured GPU time."});
    }

    const auto total_flops_est = std::accumulate(
        phase_summaries.begin(), phase_summaries.end(), uint64_t{0}, [](uint64_t value, const PhaseSummary& summary) {
            return value + summary.flops_est;
        });
    const auto total_macs_est = std::accumulate(
        phase_summaries.begin(), phase_summaries.end(), uint64_t{0}, [](uint64_t value, const PhaseSummary& summary) {
            return value + summary.macs_est;
        });
    const auto total_bytes_in = std::accumulate(
        phase_summaries.begin(), phase_summaries.end(), uint64_t{0}, [](uint64_t value, const PhaseSummary& summary) {
            return value + summary.bytes_in;
        });
    const auto total_bytes_out = std::accumulate(
        phase_summaries.begin(), phase_summaries.end(), uint64_t{0}, [](uint64_t value, const PhaseSummary& summary) {
            return value + summary.bytes_out;
        });
    const auto infer_roofline = estimate_roofline(total_flops_est,
                                                  total_macs_est,
                                                  total_bytes_in,
                                                  total_bytes_out,
                                                  report.total_cpu_us,
                                                  report.total_gpu_us);
    if (infer_roofline.flops_est > 0 && infer_roofline.bytes_moved > 0) {
        std::ostringstream msg;
        msg << "Arithmetic intensity is "
            << format_decimal(infer_roofline.arithmetic_intensity, 2)
            << " flop/byte; inferred regime is "
            << infer_roofline.dominant_regime
            << " (confidence=" << infer_roofline.confidence << ").";
        insights.push_back({"roofline",
                            std::string_view(infer_roofline.dominant_regime) == "memory" ? "info" : "info",
                            msg.str()});
    }

    if (counter_value(report, "vulkan_owns_command_buffer") > 0 ||
        counter_value(report, "vulkan_internal_submit_wait") > 0) {
        insights.push_back({"hazard",
                            "warning",
                            "Fallback kernel-owned command buffer submissions were observed; this usually indicates micro-submit overhead."});
    }

    if (phase_count(phase_summaries, "compile") > 0) {
        insights.push_back({"compile", "info", "Compilation/setup work was recorded inside the profiled infer path."});
    }

    const uint64_t pipeline_creations = counter_value(report, "pipeline_creation_count");
    if (pipeline_creations > 0) {
        insights.push_back({"pipeline_creation",
                            pipeline_creations > 1 ? "warning" : "info",
                            "Pipeline creation was observed during infer; check prewarm and pipeline cache effectiveness."});
    }

    const uint64_t descriptor_updates = counter_value(report, "descriptor_update_count");
    const uint64_t descriptor_writes = counter_value(report, "descriptor_write_count");
    if (descriptor_updates > 0 || descriptor_writes > 0) {
        std::ostringstream msg;
        msg << "Descriptor/binding updates were observed during infer";
        if (descriptor_updates > 0) {
            msg << " (updates=" << descriptor_updates;
            if (descriptor_writes > 0) {
                msg << ", writes=" << descriptor_writes;
            }
            msg << ")";
        } else {
            msg << " (writes=" << descriptor_writes << ")";
        }
        msg << "; cache reuse may still be improvable.";
        insights.push_back({"descriptor_update",
                            descriptor_updates > 3 ? "warning" : "info",
                            msg.str()});
    }

    const uint64_t binding_prepares = counter_value(report, "binding_prepare_count");
    if (binding_prepares > 0) {
        insights.push_back({"binding_prepare",
                            binding_prepares > 3 ? "warning" : "info",
                            "Backend binding preparation ran inside infer; repeated binding-table churn may be increasing CPU overhead."});
    }

    return insights;
}

template <typename T, typename ScoreFn>
std::vector<const T*> top_entries(const std::vector<T>& entries, size_t limit, ScoreFn score_fn) {
    std::vector<const T*> sorted;
    sorted.reserve(entries.size());
    for (const auto& entry : entries) {
        sorted.push_back(&entry);
    }
    std::stable_sort(sorted.begin(), sorted.end(), [&](const T* lhs, const T* rhs) {
        const auto lhs_score = score_fn(*lhs);
        const auto rhs_score = score_fn(*rhs);
        return lhs_score > rhs_score;
    });
    if (sorted.size() > limit) {
        sorted.resize(limit);
    }
    return sorted;
}

std::string build_trace_events_json(const GfxProfilingReport& report) {
    if (report.trace_sink != "perfetto") {
        return {};
    }
    std::vector<TraceEventJson> events;
    events.reserve(report.segments.size() + report.transfers.size() + report.allocations.size() + 1);
    uint64_t order = 0;
    for (const auto& segment : report.segments) {
        const uint64_t start_us = segment.wall_ts_us >= segment.cpu_us ? (segment.wall_ts_us - segment.cpu_us)
                                                                       : segment.wall_ts_us;
        std::ostringstream event;
        event << '{';
        event << "\"name\":\"" << escape_json(segment.name) << "\",";
        event << "\"cat\":\"" << escape_json(segment.phase) << "\",";
        event << "\"ph\":\"" << (segment.cpu_us != 0 ? "X" : "i") << "\",";
        event << "\"ts\":" << start_us << ',';
        if (segment.cpu_us != 0) {
            event << "\"dur\":" << segment.cpu_us << ',';
        }
        event << "\"pid\":1,";
        event << "\"tid\":" << segment.thread_id << ',';
        event << "\"args\":{";
        event << "\"backend\":\"" << escape_json(report.backend) << "\",";
        event << "\"gpu_us\":" << segment.gpu_us << ',';
        event << "\"dispatches\":" << segment.dispatches << ',';
        event << "\"bytes_in\":" << segment.bytes_in << ',';
        event << "\"bytes_out\":" << segment.bytes_out << ',';
        event << "\"inflight_slot\":" << segment.inflight_slot << ',';
        event << "\"queue_id\":" << segment.queue_id << ',';
        event << "\"cmd_buffer_id\":" << segment.cmd_buffer_id;
        event << "}}";
        events.push_back({start_us, order++, event.str()});
    }

    for (const auto& transfer : report.transfers) {
        const uint64_t start_us = transfer.wall_ts_us >= transfer.cpu_us ? (transfer.wall_ts_us - transfer.cpu_us)
                                                                         : transfer.wall_ts_us;
        std::ostringstream event;
        event << '{';
        event << "\"name\":\"" << escape_json(transfer.tag.empty() ? (transfer.h2d ? "transfer_h2d" : "transfer_d2h") : transfer.tag)
              << "\",";
        event << "\"cat\":\"transfer\",";
        event << "\"ph\":\"" << (transfer.cpu_us != 0 ? "X" : "i") << "\",";
        event << "\"ts\":" << start_us << ',';
        if (transfer.cpu_us != 0) {
            event << "\"dur\":" << transfer.cpu_us << ',';
        }
        event << "\"pid\":1,";
        event << "\"tid\":" << transfer.thread_id << ',';
        event << "\"args\":{";
        event << "\"backend\":\"" << escape_json(report.backend) << "\",";
        event << "\"direction\":\"" << (transfer.h2d ? "h2d" : "d2h") << "\",";
        event << "\"bytes\":" << transfer.bytes << ',';
        event << "\"gpu_us\":" << transfer.gpu_us;
        event << "}}";
        events.push_back({start_us, order++, event.str()});
    }

    for (const auto& alloc : report.allocations) {
        const uint64_t start_us = alloc.wall_ts_us >= alloc.cpu_us ? (alloc.wall_ts_us - alloc.cpu_us) : alloc.wall_ts_us;
        std::ostringstream event;
        event << '{';
        event << "\"name\":\"" << escape_json(alloc.tag.empty() ? "allocation" : alloc.tag) << "\",";
        event << "\"cat\":\"allocation\",";
        event << "\"ph\":\"" << (alloc.cpu_us != 0 ? "X" : "i") << "\",";
        event << "\"ts\":" << start_us << ',';
        if (alloc.cpu_us != 0) {
            event << "\"dur\":" << alloc.cpu_us << ',';
        }
        event << "\"pid\":1,";
        event << "\"tid\":" << alloc.thread_id << ',';
        event << "\"args\":{";
        event << "\"backend\":\"" << escape_json(report.backend) << "\",";
        event << "\"bytes\":" << alloc.bytes << ',';
        event << "\"reused\":" << (alloc.reused ? "true" : "false");
        event << "}}";
        events.push_back({start_us, order++, event.str()});
    }

    const uint64_t counter_ts = report.total_wall_us;
    if (!report.counters.empty()) {
        std::ostringstream event;
        event << '{';
        event << "\"name\":\"counters\",";
        event << "\"cat\":\"counter\",";
        event << "\"ph\":\"C\",";
        event << "\"ts\":" << counter_ts << ',';
        event << "\"pid\":1,";
        event << "\"tid\":0,";
        event << "\"args\":{";
        for (size_t i = 0; i < report.counters.size(); ++i) {
            if (i) {
                event << ',';
            }
            event << '"' << escape_json(report.counters[i].name) << "\":" << report.counters[i].value;
        }
        event << "}}";
        events.push_back({counter_ts, order++, event.str()});
    }

    std::stable_sort(events.begin(), events.end(), [](const TraceEventJson& lhs, const TraceEventJson& rhs) {
        if (lhs.ts != rhs.ts) {
            return lhs.ts < rhs.ts;
        }
        return lhs.order < rhs.order;
    });

    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < events.size(); ++i) {
        if (i) {
            oss << ',';
        }
        oss << events[i].json;
    }
    oss << ']';
    return oss.str();
}

}  // namespace

std::string GfxProfilingReport::to_json() const {
    const auto phase_summaries = build_phase_summaries(*this);
    const auto transfer_summaries = build_transfer_summaries(*this);
    const auto insights = build_insights(*this, phase_summaries, transfer_summaries);
    const auto total_flops_est = std::accumulate(
        phase_summaries.begin(), phase_summaries.end(), uint64_t{0}, [](uint64_t value, const PhaseSummary& summary) {
            return value + summary.flops_est;
        });
    const auto total_macs_est = std::accumulate(
        phase_summaries.begin(), phase_summaries.end(), uint64_t{0}, [](uint64_t value, const PhaseSummary& summary) {
            return value + summary.macs_est;
        });
    const auto total_bytes_in = std::accumulate(
        phase_summaries.begin(), phase_summaries.end(), uint64_t{0}, [](uint64_t value, const PhaseSummary& summary) {
            return value + summary.bytes_in;
        });
    const auto total_bytes_out = std::accumulate(
        phase_summaries.begin(), phase_summaries.end(), uint64_t{0}, [](uint64_t value, const PhaseSummary& summary) {
            return value + summary.bytes_out;
        });
    const auto infer_roofline =
        estimate_roofline(total_flops_est, total_macs_est, total_bytes_in, total_bytes_out, total_cpu_us, total_gpu_us);

    std::ostringstream oss;
    oss << '{';
    oss << "\"schema_version\":" << schema_version << ',';
    oss << "\"backend\":\"" << escape_json(backend) << "\",";
    if (!trace_sink.empty()) {
        oss << "\"trace_sink\":\"" << escape_json(trace_sink) << "\",";
    }
    oss << "\"level\":\"" << level_to_string(level) << "\",";
    oss << "\"counters_supported\":" << (counters_supported ? "true" : "false") << ',';
    oss << "\"counters_used\":" << (counters_used ? "true" : "false") << ',';
    oss << "\"total_gpu_us\":" << total_gpu_us << ',';
    oss << "\"total_cpu_us\":" << total_cpu_us << ',';
    oss << "\"total_wall_us\":" << total_wall_us << ',';
    oss << "\"total_h2d_bytes\":" << total_h2d_bytes << ',';
    oss << "\"total_d2h_bytes\":" << total_d2h_bytes << ',';

    oss << "\"nodes\":[";
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        if (i) {
            oss << ',';
        }
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
        if (i) {
            oss << ',';
        }
        oss << '{';
        oss << "\"tag\":\"" << escape_json(t.tag) << "\",";
        oss << "\"bytes\":" << t.bytes << ',';
        oss << "\"cpu_us\":" << t.cpu_us << ',';
        oss << "\"gpu_us\":" << t.gpu_us << ',';
        oss << "\"wall_ts_us\":" << t.wall_ts_us << ',';
        oss << "\"thread_id\":" << t.thread_id << ',';
        oss << "\"h2d\":" << (t.h2d ? "true" : "false");
        oss << '}';
    }
    oss << "],";

    oss << "\"allocations\":[";
    for (size_t i = 0; i < allocations.size(); ++i) {
        const auto& a = allocations[i];
        if (i) {
            oss << ',';
        }
        oss << '{';
        oss << "\"tag\":\"" << escape_json(a.tag) << "\",";
        oss << "\"bytes\":" << a.bytes << ',';
        oss << "\"cpu_us\":" << a.cpu_us << ',';
        oss << "\"wall_ts_us\":" << a.wall_ts_us << ',';
        oss << "\"thread_id\":" << a.thread_id << ',';
        oss << "\"reused\":" << (a.reused ? "true" : "false");
        oss << '}';
    }
    oss << "],";

    oss << "\"segments\":[";
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& s = segments[i];
        if (i) {
            oss << ',';
        }
        oss << '{';
        oss << "\"phase\":\"" << escape_json(s.phase) << "\",";
        oss << "\"name\":\"" << escape_json(s.name) << "\",";
        oss << "\"gpu_us\":" << s.gpu_us << ',';
        oss << "\"cpu_us\":" << s.cpu_us << ',';
        oss << "\"wall_ts_us\":" << s.wall_ts_us << ',';
        oss << "\"dispatches\":" << s.dispatches << ',';
        oss << "\"bytes_in\":" << s.bytes_in << ',';
        oss << "\"bytes_out\":" << s.bytes_out << ',';
        oss << "\"macs_est\":" << s.macs_est << ',';
        oss << "\"flops_est\":" << s.flops_est << ',';
        oss << "\"thread_id\":" << s.thread_id << ',';
        oss << "\"queue_id\":" << s.queue_id << ',';
        oss << "\"cmd_buffer_id\":" << s.cmd_buffer_id << ',';
        oss << "\"inflight_slot\":" << s.inflight_slot;
        oss << '}';
    }
    oss << "],";

    oss << "\"counters\":[";
    for (size_t i = 0; i < counters.size(); ++i) {
        const auto& c = counters[i];
        if (i) {
            oss << ',';
        }
        oss << '{';
        oss << "\"name\":\"" << escape_json(c.name) << "\",";
        oss << "\"value\":" << c.value;
        oss << '}';
    }
    oss << "],";

    const auto trace_events_json = build_trace_events_json(*this);
    if (!trace_events_json.empty()) {
        oss << "\"traceEvents\":" << trace_events_json << ',';
    }

    oss << "\"summary\":{";
    oss << "\"counter_map\":{";
    for (size_t i = 0; i < counters.size(); ++i) {
        if (i) {
            oss << ',';
        }
        oss << '"' << escape_json(counters[i].name) << "\":" << counters[i].value;
    }
    oss << "},";

    oss << "\"phase_totals\":[";
    for (size_t i = 0; i < phase_summaries.size(); ++i) {
        const auto& summary = phase_summaries[i];
        if (i) {
            oss << ',';
        }
        oss << '{';
        oss << "\"phase\":\"" << escape_json(summary.phase) << "\",";
        oss << "\"count\":" << summary.count << ',';
        oss << "\"cpu_us\":" << summary.cpu_us << ',';
        oss << "\"gpu_us\":" << summary.gpu_us << ',';
        oss << "\"dispatches\":" << summary.dispatches << ',';
        oss << "\"bytes_in\":" << summary.bytes_in << ',';
        oss << "\"bytes_out\":" << summary.bytes_out << ',';
        oss << "\"macs_est\":" << summary.macs_est << ',';
        oss << "\"flops_est\":" << summary.flops_est << ',';
        const auto roofline = estimate_roofline(summary.flops_est,
                                                summary.macs_est,
                                                summary.bytes_in,
                                                summary.bytes_out,
                                                summary.cpu_us,
                                                summary.gpu_us);
        oss << "\"bytes_moved\":" << roofline.bytes_moved << ',';
        oss << "\"arithmetic_intensity\":" << format_decimal(roofline.arithmetic_intensity, 6) << ',';
        oss << "\"gpu_tflops\":" << format_decimal(roofline.gpu_tflops, 6) << ',';
        oss << "\"cpu_tflops\":" << format_decimal(roofline.cpu_tflops, 6) << ',';
        oss << "\"gpu_gbps\":" << format_decimal(roofline.gpu_gbps, 6) << ',';
        oss << "\"cpu_gbps\":" << format_decimal(roofline.cpu_gbps, 6) << ',';
        oss << "\"dominant_regime\":\"" << roofline.dominant_regime << "\",";
        oss << "\"confidence\":\"" << roofline.confidence << "\"";
        oss << '}';
    }
    oss << "],";

    oss << "\"roofline\":{";
    oss << "\"bytes_moved\":" << infer_roofline.bytes_moved << ',';
    oss << "\"flops_est\":" << infer_roofline.flops_est << ',';
    oss << "\"macs_est\":" << infer_roofline.macs_est << ',';
    oss << "\"arithmetic_intensity\":" << format_decimal(infer_roofline.arithmetic_intensity, 6) << ',';
    oss << "\"gpu_tflops\":" << format_decimal(infer_roofline.gpu_tflops, 6) << ',';
    oss << "\"cpu_tflops\":" << format_decimal(infer_roofline.cpu_tflops, 6) << ',';
    oss << "\"gpu_gbps\":" << format_decimal(infer_roofline.gpu_gbps, 6) << ',';
    oss << "\"cpu_gbps\":" << format_decimal(infer_roofline.cpu_gbps, 6) << ',';
    oss << "\"dominant_regime\":\"" << infer_roofline.dominant_regime << "\",";
    oss << "\"confidence\":\"" << infer_roofline.confidence << "\"";
    oss << "},";

    oss << "\"transfer_totals\":[";
    for (size_t i = 0; i < transfer_summaries.size(); ++i) {
        const auto& summary = transfer_summaries[i];
        if (i) {
            oss << ',';
        }
        oss << '{';
        oss << "\"direction\":\"" << summary.direction << "\",";
        oss << "\"count\":" << summary.count << ',';
        oss << "\"bytes\":" << summary.bytes << ',';
        oss << "\"cpu_us\":" << summary.cpu_us << ',';
        oss << "\"gpu_us\":" << summary.gpu_us;
        oss << '}';
    }
    oss << "],";

    const auto hot_nodes = top_entries(nodes, 5, [](const GfxProfilingNodeEntry& entry) {
        return std::max(entry.gpu_us, entry.cpu_us);
    });
    oss << "\"hot_nodes\":[";
    for (size_t i = 0; i < hot_nodes.size(); ++i) {
        const auto& node = *hot_nodes[i];
        if (i) {
            oss << ',';
        }
        oss << '{';
        oss << "\"node_id\":" << node.node_id << ',';
        oss << "\"name\":\"" << escape_json(node.node_name) << "\",";
        oss << "\"type\":\"" << escape_json(node.node_type) << "\",";
        oss << "\"gpu_us\":" << node.gpu_us << ',';
        oss << "\"cpu_us\":" << node.cpu_us << ',';
        oss << "\"dispatches\":" << node.dispatches;
        oss << '}';
    }
    oss << "],";

    const auto hot_segments = top_entries(segments, 5, [](const GfxProfilingSegmentEntry& entry) {
        return std::max(entry.gpu_us, entry.cpu_us);
    });
    oss << "\"hot_segments\":[";
    for (size_t i = 0; i < hot_segments.size(); ++i) {
        const auto& segment = *hot_segments[i];
        if (i) {
            oss << ',';
        }
        oss << '{';
        oss << "\"phase\":\"" << escape_json(segment.phase) << "\",";
        oss << "\"name\":\"" << escape_json(segment.name) << "\",";
        oss << "\"gpu_us\":" << segment.gpu_us << ',';
        oss << "\"cpu_us\":" << segment.cpu_us << ',';
        oss << "\"dispatches\":" << segment.dispatches << ',';
        oss << "\"bytes_in\":" << segment.bytes_in << ',';
        oss << "\"bytes_out\":" << segment.bytes_out << ',';
        oss << "\"inflight_slot\":" << segment.inflight_slot << ',';
        oss << "\"cmd_buffer_id\":" << segment.cmd_buffer_id;
        oss << '}';
    }
    oss << "],";

    oss << "\"diagnostics\":[";
    for (size_t i = 0; i < insights.size(); ++i) {
        if (i) {
            oss << ',';
        }
        oss << '{';
        oss << "\"category\":\"" << insights[i].category << "\",";
        oss << "\"severity\":\"" << insights[i].severity << "\",";
        oss << "\"message\":\"" << escape_json(insights[i].message) << "\"";
        oss << '}';
    }
    oss << "]";
    oss << "}";
    oss << '}';
    return oss.str();
}

void GfxProfilingTrace::reset(ProfilingLevel level) {
    m_report = {};
    m_report.schema_version = 2;
    m_report.level = level;
    m_report.trace_sink = trace_sink_from_env(level);
    m_origin_wall_us = monotonic_wall_us();
}

void GfxProfilingTrace::set_backend(std::string_view backend) {
    m_report.backend = std::string{backend};
}

void GfxProfilingTrace::set_counter_capability(bool supported, bool used) {
    m_report.counters_supported = supported;
    m_report.counters_used = used;
}

void GfxProfilingTrace::set_total_gpu_us(uint64_t value) {
    m_report.total_gpu_us = value;
}

void GfxProfilingTrace::set_total_cpu_us(uint64_t value) {
    m_report.total_cpu_us = value;
}

void GfxProfilingTrace::set_total_wall_us(uint64_t value) {
    m_report.total_wall_us = value;
}

void GfxProfilingTrace::set_counter(std::string_view name, uint64_t value) {
    auto it = std::find_if(m_report.counters.begin(),
                           m_report.counters.end(),
                           [&](const GfxProfilingCounterEntry& entry) { return entry.name == name; });
    if (it == m_report.counters.end()) {
        m_report.counters.push_back(GfxProfilingCounterEntry{std::string{name}, value});
    } else {
        it->value = value;
    }
}

void GfxProfilingTrace::increment_counter(std::string_view name, uint64_t delta) {
    auto it = std::find_if(m_report.counters.begin(),
                           m_report.counters.end(),
                           [&](const GfxProfilingCounterEntry& entry) { return entry.name == name; });
    if (it == m_report.counters.end()) {
        m_report.counters.push_back(GfxProfilingCounterEntry{std::string{name}, delta});
    } else {
        it->value += delta;
    }
}

void GfxProfilingTrace::add_node(const GfxProfilingNodeEntry& entry) {
    m_report.nodes.push_back(entry);
}

void GfxProfilingTrace::add_transfer(const char* tag, uint64_t bytes, bool h2d, uint64_t cpu_us, uint64_t gpu_us) {
    GfxProfilingTransferEntry entry;
    entry.tag = tag ? tag : "";
    entry.bytes = bytes;
    entry.cpu_us = cpu_us;
    entry.gpu_us = gpu_us;
    entry.wall_ts_us = monotonic_wall_us() - m_origin_wall_us;
    entry.thread_id = current_thread_id_hash();
    entry.h2d = h2d;
    m_report.transfers.push_back(std::move(entry));
    if (h2d) {
        m_report.total_h2d_bytes += bytes;
    } else {
        m_report.total_d2h_bytes += bytes;
    }
}

void GfxProfilingTrace::add_allocation(const char* tag, uint64_t bytes, bool reused, uint64_t cpu_us) {
    GfxProfilingAllocEntry entry;
    entry.tag = tag ? tag : "";
    entry.bytes = bytes;
    entry.reused = reused;
    entry.cpu_us = cpu_us;
    entry.wall_ts_us = monotonic_wall_us() - m_origin_wall_us;
    entry.thread_id = current_thread_id_hash();
    m_report.allocations.push_back(std::move(entry));
}

void GfxProfilingTrace::add_segment(std::string_view phase,
                                    std::string_view name,
                                    uint64_t cpu_us,
                                    uint64_t gpu_us,
                                    uint32_t dispatches,
                                    uint64_t bytes_in,
                                    uint64_t bytes_out,
                                    uint64_t macs_est,
                                    uint64_t flops_est,
                                    int64_t inflight_slot,
                                    uint64_t queue_id,
                                    uint64_t cmd_buffer_id) {
    GfxProfilingSegmentEntry entry;
    entry.phase = std::string{phase};
    entry.name = std::string{name};
    entry.cpu_us = cpu_us;
    entry.gpu_us = gpu_us;
    entry.wall_ts_us = monotonic_wall_us() - m_origin_wall_us;
    entry.dispatches = dispatches;
    entry.bytes_in = bytes_in;
    entry.bytes_out = bytes_out;
    entry.macs_est = macs_est;
    entry.flops_est = flops_est;
    entry.thread_id = current_thread_id_hash();
    entry.queue_id = queue_id;
    entry.cmd_buffer_id = cmd_buffer_id;
    entry.inflight_slot = inflight_slot;
    m_report.segments.push_back(std::move(entry));
#if defined(__APPLE__)
    if (m_report.trace_sink == "signpost") {
        maybe_emit_signpost(m_report.backend, m_report.segments.back());
    }
#endif
}

}  // namespace gfx_plugin
}  // namespace ov
