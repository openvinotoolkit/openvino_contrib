// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "runtime/gpu_buffer_manager.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxPartitioningDeviceInfo {
    GpuBackend backend = GpuBackend::Metal;
    std::string device_key;
    uint32_t preferred_simd_width = 1;
    uint32_t subgroup_size = 1;
    uint32_t max_total_threads_per_group = 1;
    std::array<uint32_t, 3> max_threads_per_group{{1, 1, 1}};
};

struct GfxWorkgroupShape {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;

    uint64_t total_threads() const {
        return static_cast<uint64_t>(x) * static_cast<uint64_t>(y) * static_cast<uint64_t>(z);
    }
};

class GfxPartitioningModel {
public:
    explicit GfxPartitioningModel(GfxPartitioningDeviceInfo info) : m_info(std::move(info)) {}

    const GfxPartitioningDeviceInfo& device_info() const {
        return m_info;
    }

    uint32_t wave_width() const;
    uint32_t max_total_threads() const;
    uint32_t max_threads_for_axis(size_t axis) const;
    uint32_t linear_thread_budget(uint32_t hard_cap = 0) const;
    uint32_t align_thread_count(uint32_t requested_threads, uint32_t hard_cap = 0) const;
    uint32_t select_1d_thread_count(uint64_t work_items,
                                    uint32_t wide_work_waves = 2,
                                    uint64_t medium_work_threshold = 1024,
                                    uint64_t wide_work_threshold = 4096,
                                    uint32_t hard_cap = 0) const;
    std::vector<GfxWorkgroupShape> enumerate_2d_thread_shapes(uint64_t work_items_h,
                                                              uint64_t work_items_w,
                                                              uint32_t wide_work_waves = 2,
                                                              size_t max_results = 8,
                                                              uint32_t hard_cap = 0) const;
    bool supports(const GfxWorkgroupShape& shape) const;
    std::vector<GfxWorkgroupShape> filter_supported(const std::vector<GfxWorkgroupShape>& candidates) const;

private:
    GfxPartitioningDeviceInfo m_info;
};

GfxPartitioningDeviceInfo query_partitioning_device_info(const GpuBufferManager* buffer_manager);
GfxPartitioningDeviceInfo make_default_partitioning_device_info(GpuBackend backend);
uint64_t bucketize_partition_work(uint64_t value);

}  // namespace gfx_plugin
}  // namespace ov
