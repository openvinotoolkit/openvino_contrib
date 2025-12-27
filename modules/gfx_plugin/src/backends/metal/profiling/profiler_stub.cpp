// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/profiling/profiler.hpp"

namespace ov {
namespace gfx_plugin {

MetalProfiler::MetalProfiler(GfxProfilerConfig cfg, MetalDeviceCaps /*caps*/, MetalDeviceHandle /*device*/)
    : m_cfg(cfg), m_timestamps(nullptr, false) {}

void MetalProfiler::set_config(GfxProfilerConfig cfg) {
    m_cfg = cfg;
}

void MetalProfiler::begin_infer(size_t /*expected_samples*/) {}

void MetalProfiler::end_infer(MetalCommandBufferHandle /*command_buffer*/) {}

void MetalProfiler::begin_node(uint32_t /*node_id*/,
                               const char* /*node_name*/,
                               const char* /*node_type*/,
                               const char* /*exec_type*/) {}

void MetalProfiler::end_node(uint32_t /*node_id*/,
                             std::chrono::microseconds /*cpu_us*/,
                             MetalGpuTimestamps::SampleIndex /*sample_begin*/,
                             MetalGpuTimestamps::SampleIndex /*sample_end*/) {}

MetalGpuTimestamps::SampleIndex MetalProfiler::gpu_sample_begin(MetalCommandEncoderHandle /*encoder*/) {
    return -1;
}

MetalGpuTimestamps::SampleIndex MetalProfiler::gpu_sample_end(MetalCommandEncoderHandle /*encoder*/) {
    return -1;
}

void MetalProfiler::set_memory_stats(const MetalMemoryStats& /*stats*/) {}

void MetalProfiler::record_alloc(const char* /*tag*/,
                                 size_t /*bytes*/,
                                 bool /*reused*/,
                                 std::chrono::microseconds /*cpu_us*/) {}

std::vector<ov::ProfilingInfo> MetalProfiler::export_ov() const {
    return {};
}

MetalProfilingReport MetalProfiler::export_extended() const {
    return {};
}

}  // namespace gfx_plugin
}  // namespace ov
