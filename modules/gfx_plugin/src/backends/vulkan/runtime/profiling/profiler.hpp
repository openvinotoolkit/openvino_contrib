// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "openvino/runtime/profiling_info.hpp"
#include "openvino/gfx_plugin/profiling.hpp"

namespace ov {
namespace gfx_plugin {

class VulkanProfiler {
public:
    struct SamplePair {
        uint32_t begin = UINT32_MAX;
        uint32_t end = UINT32_MAX;
    };

    VulkanProfiler(VkDevice device,
                   VkPhysicalDevice physical_device,
                   uint32_t queue_family_index);

    void set_config(const GfxProfilerConfig& cfg);
    bool enabled() const { return m_enabled && m_supported; }

    void begin_infer(size_t expected_samples);
    void end_infer();

    void begin_node(uint32_t node_id,
                    const char* node_name,
                    const char* node_type,
                    const char* exec_type);
    void end_node(uint32_t node_id, SamplePair samples);

    SamplePair reserve_samples();
    void write_timestamp(VkCommandBuffer cmd, uint32_t query_index) const;

    std::vector<ov::ProfilingInfo> export_ov() const;

private:
    struct NodeRec {
        uint32_t node_id = 0;
        std::string node_name;
        std::string node_type;
        std::string exec_type;
        uint64_t gpu_us = 0;
        uint32_t dispatches = 0;
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

    std::vector<NodeRec> m_nodes;
};

}  // namespace gfx_plugin
}  // namespace ov
