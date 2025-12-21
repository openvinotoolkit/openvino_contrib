// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/memory/metal_memory_stats.hpp"
#include "runtime/profiling/metal_profiler_config.hpp"

namespace ov {
namespace gfx_plugin {

struct MetalProfilingNodeEntry {
    uint32_t node_id = 0;
    std::string node_name;
    std::string node_type;
    std::string exec_type;
    uint64_t gpu_us = 0;
    uint64_t cpu_us = 0;
    uint32_t dispatches = 0;
};

struct MetalProfilingTransferEntry {
    std::string tag;
    uint64_t bytes = 0;
    uint64_t cpu_us = 0;
    uint64_t gpu_us = 0;
    bool h2d = true;
};

struct MetalProfilingAllocEntry {
    std::string tag;
    uint64_t bytes = 0;
    uint64_t cpu_us = 0;
    bool reused = false;
};

struct MetalProfilingAllocSummaryEntry {
    std::string tag;
    uint64_t bytes = 0;
    uint64_t alloc_count = 0;
    uint64_t reuse_count = 0;
    uint64_t cpu_us = 0;
};

struct MetalProfilingSegmentEntry {
    std::string tag;
    uint64_t gpu_us = 0;
    uint64_t cpu_us = 0;
    uint32_t dispatches = 0;
};

struct MetalProfilingReport {
    ProfilingLevel level = ProfilingLevel::Off;
    bool counters_supported = false;
    bool counters_used = false;

    uint64_t total_gpu_us = 0;
    uint64_t total_cpu_us = 0;
    uint64_t total_wall_us = 0;

    uint64_t total_h2d_bytes = 0;
    uint64_t total_d2h_bytes = 0;

    MetalMemoryStats memory_stats{};

    std::vector<MetalProfilingNodeEntry> nodes;
    std::vector<MetalProfilingTransferEntry> transfers;
    std::vector<MetalProfilingAllocEntry> allocations;
    std::vector<MetalProfilingAllocSummaryEntry> alloc_summary;
    std::vector<MetalProfilingSegmentEntry> segments;

    std::string to_json() const;
};

}  // namespace gfx_plugin
}  // namespace ov
