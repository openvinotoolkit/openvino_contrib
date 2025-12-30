// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/profiling/profiler.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
uint64_t to_us(double ns) {
    if (ns <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(ns / 1000.0);
}
}  // namespace

VulkanProfiler::VulkanProfiler(VkDevice device,
                               VkPhysicalDevice physical_device,
                               uint32_t queue_family_index)
    : m_device(device),
      m_physical_device(physical_device),
      m_queue_family_index(queue_family_index) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physical_device, &props);
    m_timestamp_period = props.limits.timestampPeriod;
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &qcount, nullptr);
    VkQueueFamilyProperties qprops{};
    if (qcount > 0 && m_queue_family_index < qcount) {
        std::vector<VkQueueFamilyProperties> qprops_list(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &qcount, qprops_list.data());
        qprops = qprops_list[m_queue_family_index];
    }
    m_supported = (m_timestamp_period > 0.0f) &&
                  (qprops.timestampValidBits > 0) &&
                  (props.limits.timestampComputeAndGraphics != 0);
}

void VulkanProfiler::set_config(const GfxProfilerConfig& cfg) {
    m_cfg = cfg;
    m_enabled = (m_cfg.level != ProfilingLevel::Off);
}

void VulkanProfiler::begin_infer(size_t expected_samples) {
    m_nodes.clear();
    m_next_query = 0;
    if (!enabled()) {
        return;
    }
    ensure_query_pool(expected_samples);
}

void VulkanProfiler::end_infer() {
    // No-op: timestamps are resolved per node once command buffer completes.
}

void VulkanProfiler::begin_node(uint32_t node_id,
                                const char* node_name,
                                const char* node_type,
                                const char* exec_type) {
    if (!enabled()) {
        return;
    }
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

VulkanProfiler::SamplePair VulkanProfiler::reserve_samples() {
    SamplePair pair{};
    if (!enabled()) {
        return pair;
    }
    if (m_next_query + 1 >= m_query_count) {
        return pair;
    }
    pair.begin = m_next_query++;
    pair.end = m_next_query++;
    return pair;
}

void VulkanProfiler::write_timestamp(VkCommandBuffer cmd, uint32_t query_index) const {
    if (!enabled() || !cmd || !m_query_pool || query_index == UINT32_MAX) {
        return;
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, m_query_pool, query_index);
}

void VulkanProfiler::end_node(uint32_t node_id, SamplePair samples) {
    if (!enabled()) {
        return;
    }
    if (node_id >= m_nodes.size()) {
        return;
    }
    if (samples.begin == UINT32_MAX || samples.end == UINT32_MAX) {
        return;
    }
    const uint64_t begin = read_timestamp(samples.begin);
    const uint64_t end = read_timestamp(samples.end);
    if (end > begin) {
        const double ns = static_cast<double>(end - begin) * static_cast<double>(m_timestamp_period);
        m_nodes[node_id].gpu_us += to_us(ns);
        m_nodes[node_id].dispatches += 1;
    }
}

std::vector<ov::ProfilingInfo> VulkanProfiler::export_ov() const {
    std::vector<ov::ProfilingInfo> out;
    if (!enabled()) {
        return out;
    }
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
        const auto us = std::chrono::microseconds{static_cast<int64_t>(rec.gpu_us)};
        info.real_time = us;
        info.cpu_time = us;
        out.push_back(std::move(info));
    }
    return out;
}

void VulkanProfiler::ensure_query_pool(size_t sample_pairs) {
    if (!enabled()) {
        return;
    }
    const uint32_t desired = sample_pairs == 0 ? 0u : static_cast<uint32_t>(sample_pairs * 2);
    if (desired == 0) {
        return;
    }
    if (m_query_pool) {
        vkDestroyQueryPool(m_device, m_query_pool, nullptr);
        m_query_pool = VK_NULL_HANDLE;
        m_query_count = 0;
    }
    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = desired;
    VkResult res = vkCreateQueryPool(m_device, &info, nullptr, &m_query_pool);
    if (res != VK_SUCCESS) {
        OPENVINO_THROW("GFX Vulkan: vkCreateQueryPool failed: ", static_cast<int>(res));
    }
    m_query_count = desired;
}

uint64_t VulkanProfiler::read_timestamp(uint32_t query_index) const {
    if (!enabled() || !m_query_pool) {
        return 0;
    }
    uint64_t value = 0;
    VkResult res = vkGetQueryPoolResults(m_device,
                                         m_query_pool,
                                         query_index,
                                         1,
                                         sizeof(uint64_t),
                                         &value,
                                         sizeof(uint64_t),
                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (res != VK_SUCCESS) {
        return 0;
    }
    return value;
}

}  // namespace gfx_plugin
}  // namespace ov
