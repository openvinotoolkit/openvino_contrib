// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "openvino/gfx_plugin/profiling.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxProfilingNodeEntry {
    uint32_t node_id = 0;
    std::string node_name;
    std::string node_type;
    std::string exec_type;
    uint64_t gpu_us = 0;
    uint64_t cpu_us = 0;
    uint32_t dispatches = 0;
};

struct GfxProfilingTransferEntry {
    std::string tag;
    uint64_t bytes = 0;
    uint64_t cpu_us = 0;
    uint64_t gpu_us = 0;
    uint64_t wall_ts_us = 0;
    uint64_t thread_id = 0;
    bool h2d = true;
};

struct GfxProfilingAllocEntry {
    std::string tag;
    uint64_t bytes = 0;
    uint64_t cpu_us = 0;
    uint64_t wall_ts_us = 0;
    uint64_t thread_id = 0;
    bool reused = false;
};

struct GfxProfilingSegmentEntry {
    std::string phase;
    std::string name;
    uint64_t gpu_us = 0;
    uint64_t cpu_us = 0;
    uint64_t wall_ts_us = 0;
    uint32_t dispatches = 0;
    uint64_t bytes_in = 0;
    uint64_t bytes_out = 0;
    uint64_t macs_est = 0;
    uint64_t flops_est = 0;
    uint64_t thread_id = 0;
    uint64_t queue_id = 0;
    uint64_t cmd_buffer_id = 0;
    int64_t inflight_slot = -1;
};

struct GfxProfilingCounterEntry {
    std::string name;
    uint64_t value = 0;
};

struct GfxProfilingReport {
    uint32_t schema_version = 2;
    std::string backend;
    std::string trace_sink;
    ProfilingLevel level = ProfilingLevel::Off;
    bool counters_supported = false;
    bool counters_used = false;

    uint64_t total_gpu_us = 0;
    uint64_t total_cpu_us = 0;
    uint64_t total_wall_us = 0;

    uint64_t total_h2d_bytes = 0;
    uint64_t total_d2h_bytes = 0;

    std::vector<GfxProfilingNodeEntry> nodes;
    std::vector<GfxProfilingTransferEntry> transfers;
    std::vector<GfxProfilingAllocEntry> allocations;
    std::vector<GfxProfilingSegmentEntry> segments;
    std::vector<GfxProfilingCounterEntry> counters;

    std::string to_json() const;
};

class GfxProfilingTrace {
public:
    void reset(ProfilingLevel level);
    void set_backend(std::string_view backend);

    void set_counter_capability(bool supported, bool used);
    void set_total_gpu_us(uint64_t value);
    void set_total_cpu_us(uint64_t value);
    void set_total_wall_us(uint64_t value);
    void set_counter(std::string_view name, uint64_t value);
    void increment_counter(std::string_view name, uint64_t delta = 1);

    void add_node(const GfxProfilingNodeEntry& entry);
    void add_transfer(const char* tag, uint64_t bytes, bool h2d, uint64_t cpu_us, uint64_t gpu_us = 0);
    void add_allocation(const char* tag, uint64_t bytes, bool reused, uint64_t cpu_us);
    void add_segment(std::string_view phase,
                     std::string_view name,
                     uint64_t cpu_us,
                     uint64_t gpu_us = 0,
                     uint32_t dispatches = 0,
                     uint64_t bytes_in = 0,
                     uint64_t bytes_out = 0,
                     uint64_t macs_est = 0,
                     uint64_t flops_est = 0,
                     int64_t inflight_slot = -1,
                     uint64_t queue_id = 0,
                     uint64_t cmd_buffer_id = 0);

    const GfxProfilingReport& report() const {
        return m_report;
    }

    std::string to_json() const {
        return m_report.to_json();
    }

private:
    GfxProfilingReport m_report;
    uint64_t m_origin_wall_us = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
