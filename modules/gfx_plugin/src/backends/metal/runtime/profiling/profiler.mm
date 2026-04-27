// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/profiling/profiler.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#include "openvino/core/except.hpp"

#include <unordered_map>

namespace ov {
namespace gfx_plugin {

namespace {
uint64_t to_us(std::chrono::steady_clock::duration d) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
}
}  // namespace

MetalProfiler::MetalProfiler(GfxProfilerConfig cfg, MetalDeviceCaps caps, MetalDeviceHandle device)
    : m_cfg(cfg), m_timestamps(device, caps.supports_counter_sampling) {
    m_counters_supported = caps.supports_counter_sampling;
    m_use_timestamps = (cfg.level == ProfilingLevel::Detailed) && m_counters_supported;
    m_counters_used = m_use_timestamps;
}

void MetalProfiler::set_config(const GfxProfilerConfig& cfg) {
    m_cfg = cfg;
    m_use_timestamps = (m_cfg.level == ProfilingLevel::Detailed) && m_counters_supported;
    m_counters_used = m_use_timestamps;
}

void MetalProfiler::begin_infer(size_t expected_samples) {
    m_nodes.clear();
    m_total_gpu_us = 0;
    m_total_cpu_us = 0;
    m_total_wall_us = 0;
    m_wall_start = std::chrono::steady_clock::now();
    m_counters_used = false;
    m_trace.reset(m_cfg.level);
    m_trace.set_backend("metal");
    m_trace.set_counter_capability(m_counters_supported, false);

    if (m_use_timestamps) {
        m_timestamps.begin_frame(expected_samples);
        m_counters_used = m_timestamps.supported();
        if (!m_counters_used) {
            m_use_timestamps = false;
        }
    }
    m_trace.set_counter_capability(m_counters_supported, m_counters_used);
}

void MetalProfiler::end_infer(GpuCommandBufferHandle command_buffer) {
    record_command_buffer_gpu_time(command_buffer);

    if (m_use_timestamps) {
        m_timestamps.resolve();
        compute_gpu_times();
    }

    m_total_cpu_us = 0;
    for (const auto& node : m_nodes) {
        m_total_cpu_us += node.cpu_us;
    }

    const auto wall = std::chrono::steady_clock::now() - m_wall_start;
    m_total_wall_us = to_us(wall);

    if (m_total_gpu_us == 0) {
        uint64_t sum_gpu = 0;
        for (const auto& node : m_nodes) {
            sum_gpu += node.gpu_us;
        }
        m_total_gpu_us = sum_gpu;
    }

    m_trace.set_total_gpu_us(m_total_gpu_us);
    m_trace.set_total_cpu_us(m_total_cpu_us);
    m_trace.set_total_wall_us(m_total_wall_us);
    m_trace.set_counter_capability(m_counters_supported, m_counters_used);
}

void MetalProfiler::record_segment(std::string_view phase,
                                   std::string_view name,
                                   std::chrono::microseconds cpu_us,
                                   uint64_t gpu_us,
                                   uint32_t dispatches,
                                   uint64_t bytes_in,
                                   uint64_t bytes_out,
                                   uint64_t macs_est,
                                   uint64_t flops_est,
                                   int64_t inflight_slot,
                                   uint64_t queue_id,
                                   uint64_t cmd_buffer_id) {
    m_trace.add_segment(phase,
                        name,
                        static_cast<uint64_t>(cpu_us.count()),
                        gpu_us,
                        dispatches,
                        bytes_in,
                        bytes_out,
                        macs_est,
                        flops_est,
                        inflight_slot,
                        queue_id,
                        cmd_buffer_id);
}

void MetalProfiler::record_transfer(const char* tag,
                                    uint64_t bytes,
                                    bool h2d,
                                    std::chrono::microseconds cpu_us,
                                    uint64_t gpu_us) {
    m_trace.add_transfer(tag, bytes, h2d, static_cast<uint64_t>(cpu_us.count()), gpu_us);
}

void MetalProfiler::increment_counter(std::string_view name, uint64_t delta) {
    m_trace.increment_counter(name, delta);
}

void MetalProfiler::set_counter(std::string_view name, uint64_t value) {
    m_trace.set_counter(name, value);
}

void MetalProfiler::begin_node(uint32_t node_id,
                               const char* node_name,
                               const char* node_type,
                               const char* exec_type) {
    if (node_id >= m_nodes.size()) {
        m_nodes.resize(node_id + 1);
    }
    auto& rec = m_nodes[node_id];
    rec.node_id = node_id;
    if (rec.node_name.empty() && node_name) {
        rec.node_name = node_name;
    }
    if (rec.node_type.empty() && node_type) {
        rec.node_type = node_type;
    }
    if (rec.exec_type.empty() && exec_type) {
        rec.exec_type = exec_type;
    }
}

void MetalProfiler::end_node(uint32_t node_id,
                             std::chrono::microseconds cpu_us,
                             MetalGpuTimestamps::SampleIndex sample_begin,
                             MetalGpuTimestamps::SampleIndex sample_end) {
    if (node_id >= m_nodes.size()) {
        return;
    }
    auto& rec = m_nodes[node_id];
    rec.cpu_us += static_cast<uint64_t>(cpu_us.count());
    rec.dispatches += 1;
    if (m_use_timestamps && sample_begin >= 0 && sample_end >= 0) {
        rec.samples.push_back(SamplePair{sample_begin, sample_end});
    }
}

MetalGpuTimestamps::SampleIndex MetalProfiler::gpu_sample_begin(MetalCommandEncoderHandle encoder) {
    if (!m_use_timestamps || !m_timestamps.supported()) {
        return -1;
    }
    return m_timestamps.sample_begin(encoder);
}

MetalGpuTimestamps::SampleIndex MetalProfiler::gpu_sample_end(MetalCommandEncoderHandle encoder) {
    if (!m_use_timestamps || !m_timestamps.supported()) {
        return -1;
    }
    return m_timestamps.sample_end(encoder);
}

void MetalProfiler::set_memory_stats(const MetalMemoryStats& stats) {
    m_memory_stats = stats;
    m_trace.set_counter("bytes_allocated_total", stats.bytes_allocated_total);
    m_trace.set_counter("bytes_in_freelist", stats.bytes_in_freelist);
    m_trace.set_counter("bytes_live_transient", stats.bytes_live_transient);
    m_trace.set_counter("bytes_live_handles", stats.bytes_live_handles);
    m_trace.set_counter("bytes_persistent", stats.bytes_persistent);
    m_trace.set_counter("peak_live", stats.peak_live);
    m_trace.set_counter("num_alloc_calls", stats.num_alloc_calls);
    m_trace.set_counter("num_reuse_hits", stats.num_reuse_hits);
    m_trace.set_counter("h2d_bytes_counter", stats.h2d_bytes);
    m_trace.set_counter("d2h_bytes_counter", stats.d2h_bytes);
    m_trace.set_counter("h2d_ops", stats.h2d_ops);
    m_trace.set_counter("d2h_ops", stats.d2h_ops);
}

void MetalProfiler::record_alloc(const char* tag, size_t bytes, bool reused, std::chrono::microseconds cpu_us) {
    if (!m_cfg.include_allocations) {
        return;
    }
    m_trace.add_allocation(tag, static_cast<uint64_t>(bytes), reused, static_cast<uint64_t>(cpu_us.count()));
}

void MetalProfiler::compute_gpu_times() {
    const double factor = m_timestamps.gpu_ticks_to_ns_factor();
    if (factor <= 0.0) {
        return;
    }
    for (auto& rec : m_nodes) {
        uint64_t gpu_us = 0;
        for (const auto& pair : rec.samples) {
            const uint64_t begin = m_timestamps.get_timestamp(pair.begin);
            const uint64_t end = m_timestamps.get_timestamp(pair.end);
            if (end <= begin) {
                continue;
            }
            const double ns = static_cast<double>(end - begin) * factor;
            if (ns <= 0.0) {
                continue;
            }
            gpu_us += static_cast<uint64_t>(ns / 1000.0);
        }
        rec.gpu_us = gpu_us;
    }
}

void MetalProfiler::record_command_buffer_gpu_time(MetalCommandBufferHandle command_buffer) {
#ifdef __OBJC__
    if (!command_buffer) {
        return;
    }
    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(command_buffer);
    if (!cb) {
        return;
    }
    const double start_s = cb.GPUStartTime;
    const double end_s = cb.GPUEndTime;
    if (end_s > start_s && start_s > 0.0) {
        m_total_gpu_us = static_cast<uint64_t>((end_s - start_s) * 1e6);
    }
#else
    (void)command_buffer;
#endif
}

std::vector<ov::ProfilingInfo> MetalProfiler::export_ov() const {
    std::vector<ov::ProfilingInfo> out;
    out.reserve(m_nodes.size());
    for (const auto& rec : m_nodes) {
        if (rec.node_name.empty()) {
            continue;
        }
        ov::ProfilingInfo info;
        info.node_name = rec.node_name;
        info.node_type = rec.node_type;
        info.exec_type = rec.exec_type.empty() ? std::string{"GFX"} : rec.exec_type;
        info.status = ov::ProfilingInfo::Status::EXECUTED;
        const uint64_t measured_us = rec.gpu_us ? rec.gpu_us : rec.cpu_us;
        info.real_time = std::chrono::microseconds{static_cast<int64_t>(measured_us)};
        info.cpu_time = std::chrono::microseconds{static_cast<int64_t>(rec.cpu_us)};
        out.push_back(std::move(info));
    }
    return out;
}

GfxProfilingReport MetalProfiler::export_extended() const {
    auto report = m_trace.report();
    report.total_h2d_bytes = std::max<uint64_t>(report.total_h2d_bytes, m_memory_stats.h2d_bytes);
    report.total_d2h_bytes = std::max<uint64_t>(report.total_d2h_bytes, m_memory_stats.d2h_bytes);
    report.nodes.reserve(report.nodes.size() + m_nodes.size());
    for (const auto& rec : m_nodes) {
        if (rec.node_name.empty())
            continue;
        GfxProfilingNodeEntry entry;
        entry.node_id = rec.node_id;
        entry.node_name = rec.node_name;
        entry.node_type = rec.node_type;
        entry.exec_type = rec.exec_type.empty() ? std::string{"GFX"} : rec.exec_type;
        entry.gpu_us = rec.gpu_us;
        entry.cpu_us = rec.cpu_us;
        entry.dispatches = rec.dispatches;
        report.nodes.push_back(std::move(entry));
    }

    return report;
}

std::string MetalProfiler::export_extended_json() const {
    return export_extended().to_json();
}

GfxProfilingReport MetalProfiler::export_extended_report() const {
    return export_extended();
}

}  // namespace gfx_plugin
}  // namespace ov
