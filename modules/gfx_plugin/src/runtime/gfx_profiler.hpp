// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <chrono>
#include <string>
#include <string_view>
#include <vector>

#include "openvino/runtime/profiling_info.hpp"
#include "openvino/gfx_plugin/profiling.hpp"
#include "runtime/gfx_profiling_report.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfiler {
public:
    virtual ~GfxProfiler() = default;

    virtual void set_config(const GfxProfilerConfig& cfg) = 0;
    virtual const GfxProfilerConfig& config() const = 0;

    virtual void begin_infer(size_t expected_samples) = 0;
    virtual void end_infer(GpuCommandBufferHandle command_buffer) = 0;

    virtual void record_segment(std::string_view /*phase*/,
                                std::string_view /*name*/,
                                std::chrono::microseconds /*cpu_us*/,
                                uint64_t /*gpu_us*/ = 0,
                                uint32_t /*dispatches*/ = 0,
                                uint64_t /*bytes_in*/ = 0,
                                uint64_t /*bytes_out*/ = 0,
                                uint64_t /*macs_est*/ = 0,
                                uint64_t /*flops_est*/ = 0,
                                int64_t /*inflight_slot*/ = -1,
                                uint64_t /*queue_id*/ = 0,
                                uint64_t /*cmd_buffer_id*/ = 0) {}
    virtual void record_transfer(const char* /*tag*/,
                                 uint64_t /*bytes*/,
                                 bool /*h2d*/,
                                 std::chrono::microseconds /*cpu_us*/,
                                 uint64_t /*gpu_us*/ = 0) {}
    virtual void increment_counter(std::string_view /*name*/, uint64_t /*delta*/ = 1) {}
    virtual void set_counter(std::string_view /*name*/, uint64_t /*value*/) {}

    virtual std::vector<ov::ProfilingInfo> export_ov() const = 0;
    virtual std::string export_extended_json() const { return {}; }
    virtual GfxProfilingReport export_extended_report() const { return {}; }

    virtual void* native_handle() = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
