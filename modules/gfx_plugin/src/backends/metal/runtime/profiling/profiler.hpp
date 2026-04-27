// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "backends/metal/runtime/memory/buffer.hpp"
#include "backends/metal/runtime/memory/device_caps.hpp"
#include "backends/metal/runtime/memory/memory_stats.hpp"
#include "backends/metal/runtime/profiling/gpu_timestamps.hpp"
#include "runtime/gfx_profiler.hpp"

namespace ov {
namespace gfx_plugin {

// Visibility helper: export profiler symbols for tests and downstream components.
#if defined(__clang__) || defined(__GNUC__)
#    define GFX_PROFILER_API __attribute__((visibility("default")))
#else
#    define GFX_PROFILER_API
#endif

class GFX_PROFILER_API MetalProfiler : public GfxProfiler {
public:
    MetalProfiler(GfxProfilerConfig cfg, MetalDeviceCaps caps, MetalDeviceHandle device);

    void set_config(const GfxProfilerConfig& cfg) override;
    const GfxProfilerConfig& config() const override { return m_cfg; }

    void begin_infer(size_t expected_samples) override;
    void end_infer(GpuCommandBufferHandle command_buffer) override;
    void record_segment(std::string_view phase,
                        std::string_view name,
                        std::chrono::microseconds cpu_us,
                        uint64_t gpu_us = 0,
                        uint32_t dispatches = 0,
                        uint64_t bytes_in = 0,
                        uint64_t bytes_out = 0,
                        uint64_t macs_est = 0,
                        uint64_t flops_est = 0,
                        int64_t inflight_slot = -1,
                        uint64_t queue_id = 0,
                        uint64_t cmd_buffer_id = 0) override;
    void record_transfer(const char* tag,
                         uint64_t bytes,
                         bool h2d,
                         std::chrono::microseconds cpu_us,
                         uint64_t gpu_us = 0) override;
    void increment_counter(std::string_view name, uint64_t delta = 1) override;
    void set_counter(std::string_view name, uint64_t value) override;

    void begin_node(uint32_t node_id, const char* node_name, const char* node_type, const char* exec_type);
    void end_node(uint32_t node_id,
                  std::chrono::microseconds cpu_us,
                  MetalGpuTimestamps::SampleIndex sample_begin,
                  MetalGpuTimestamps::SampleIndex sample_end);

    MetalGpuTimestamps::SampleIndex gpu_sample_begin(MetalCommandEncoderHandle encoder);
    MetalGpuTimestamps::SampleIndex gpu_sample_end(MetalCommandEncoderHandle encoder);

    void set_memory_stats(const MetalMemoryStats& stats);
    void record_alloc(const char* tag, size_t bytes, bool reused, std::chrono::microseconds cpu_us);

    std::vector<ov::ProfilingInfo> export_ov() const override;
    GfxProfilingReport export_extended() const;
    std::string export_extended_json() const override;
    GfxProfilingReport export_extended_report() const override;
    void* native_handle() override { return this; }

private:
    struct SamplePair {
        MetalGpuTimestamps::SampleIndex begin = -1;
        MetalGpuTimestamps::SampleIndex end = -1;
    };

    struct NodeRecord {
        uint32_t node_id = 0;
        std::string node_name;
        std::string node_type;
        std::string exec_type;
        uint64_t gpu_us = 0;
        uint64_t cpu_us = 0;
        uint32_t dispatches = 0;
        std::vector<SamplePair> samples;
    };

    void compute_gpu_times();
    void record_command_buffer_gpu_time(MetalCommandBufferHandle command_buffer);

    GfxProfilerConfig m_cfg;
    MetalGpuTimestamps m_timestamps;

    bool m_use_timestamps = false;

    std::chrono::steady_clock::time_point m_wall_start;
    uint64_t m_total_wall_us = 0;
    uint64_t m_total_gpu_us = 0;
    uint64_t m_total_cpu_us = 0;

    std::vector<NodeRecord> m_nodes;
    MetalMemoryStats m_memory_stats{};
    GfxProfilingTrace m_trace;

    bool m_counters_supported = false;
    bool m_counters_used = false;
};

}  // namespace gfx_plugin
}  // namespace ov
