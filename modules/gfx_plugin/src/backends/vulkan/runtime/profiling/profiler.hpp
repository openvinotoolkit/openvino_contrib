// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <chrono>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "openvino/runtime/profiling_info.hpp"
#include "runtime/gfx_profiler.hpp"

namespace ov {
namespace gfx_plugin {

class VulkanProfiler : public GfxProfiler {
public:
    struct SamplePair {
        uint32_t begin = UINT32_MAX;
        uint32_t end = UINT32_MAX;
    };

    VulkanProfiler(VkDevice device,
                   VkPhysicalDevice physical_device,
                   uint32_t queue_family_index);

    void set_config(const GfxProfilerConfig& cfg) override;
    const GfxProfilerConfig& config() const override { return m_cfg; }
    bool enabled() const { return m_enabled; }
    bool timestamps_supported() const { return m_supported; }

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

    void begin_node(uint32_t node_id,
                    const char* node_name,
                    const char* node_type,
                    const char* exec_type);
    void end_node(uint32_t node_id, SamplePair samples);

    SamplePair reserve_samples();
    void write_timestamp(VkCommandBuffer cmd, uint32_t query_index) const;

    std::vector<ov::ProfilingInfo> export_ov() const override;
    std::string export_extended_json() const override;
    GfxProfilingReport export_extended_report() const override;
    void* native_handle() override { return this; }

private:
    struct NodeRec {
        uint32_t node_id = 0;
        std::string node_name;
        std::string node_type;
        std::string exec_type;
        uint64_t gpu_us = 0;
        uint32_t dispatches = 0;
        std::vector<SamplePair> pending_samples;
    };

    void ensure_query_pool(size_t sample_pairs);
    uint64_t read_timestamp(uint32_t query_index) const;

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
    uint32_t m_queue_family_index = 0;
    VkQueryPool m_query_pool = VK_NULL_HANDLE;
    uint32_t m_query_count = 0;
    uint32_t m_next_query = 0;
    float m_timestamp_period = 0.0f;  // ns per tick
    bool m_supported = false;
    bool m_enabled = false;
    GfxProfilerConfig m_cfg{};
    std::chrono::steady_clock::time_point m_wall_start{};
    GfxProfilingTrace m_trace;

    std::vector<NodeRec> m_nodes;
};

}  // namespace gfx_plugin
}  // namespace ov
