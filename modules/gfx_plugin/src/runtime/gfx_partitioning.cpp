// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_partitioning.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace ov {
namespace gfx_plugin {

namespace {

uint64_t ceil_div_u64(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::vector<uint32_t> enumerate_axis_sizes(uint32_t axis_limit, uint64_t work_items) {
    std::vector<uint32_t> values;
    axis_limit = std::max<uint32_t>(1u, axis_limit);
    uint32_t value = 1;
    while (value < axis_limit) {
        values.push_back(value);
        if (value > axis_limit / 2) {
            break;
        }
        value <<= 1;
    }
    values.push_back(axis_limit);
    const uint32_t work_limited =
        static_cast<uint32_t>(std::max<uint64_t>(1u, std::min<uint64_t>(axis_limit, std::max<uint64_t>(1u, work_items))));
    values.push_back(work_limited);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

struct RankedWorkgroupShape {
    GfxWorkgroupShape shape{};
    uint64_t thread_gap = std::numeric_limits<uint64_t>::max();
    uint64_t aspect_gap = std::numeric_limits<uint64_t>::max();
    uint64_t tail_waste = std::numeric_limits<uint64_t>::max();
};

bool is_better_shape(const RankedWorkgroupShape& lhs, const RankedWorkgroupShape& rhs) {
    if (lhs.aspect_gap != rhs.aspect_gap) {
        return lhs.aspect_gap < rhs.aspect_gap;
    }
    if (lhs.thread_gap != rhs.thread_gap) {
        return lhs.thread_gap < rhs.thread_gap;
    }
    if (lhs.tail_waste != rhs.tail_waste) {
        return lhs.tail_waste < rhs.tail_waste;
    }
    if (lhs.shape.total_threads() != rhs.shape.total_threads()) {
        return lhs.shape.total_threads() > rhs.shape.total_threads();
    }
    if (lhs.shape.x != rhs.shape.x) {
        return lhs.shape.x > rhs.shape.x;
    }
    return lhs.shape.y > rhs.shape.y;
}

GfxPartitioningDeviceInfo make_info_from_device_info(const GpuExecutionDeviceInfo& info) {
    GfxPartitioningDeviceInfo partition_info{};
    partition_info.backend = info.backend;
    partition_info.device_key = info.device_key;
    partition_info.preferred_simd_width = std::max<uint32_t>(info.preferred_simd_width, 1u);
    partition_info.subgroup_size = std::max<uint32_t>(info.subgroup_size, 1u);
    partition_info.max_total_threads_per_group = std::max<uint32_t>(info.max_total_threads_per_group, 1u);
    partition_info.max_threads_per_group = {std::max<uint32_t>(info.max_threads_per_group[0], 1u),
                                            std::max<uint32_t>(info.max_threads_per_group[1], 1u),
                                            std::max<uint32_t>(info.max_threads_per_group[2], 1u)};
    return partition_info;
}

}  // namespace

uint32_t GfxPartitioningModel::wave_width() const {
    return std::max<uint32_t>(1u, std::max(m_info.subgroup_size, m_info.preferred_simd_width));
}

uint32_t GfxPartitioningModel::max_total_threads() const {
    return std::max<uint32_t>(1u, m_info.max_total_threads_per_group);
}

uint32_t GfxPartitioningModel::max_threads_for_axis(size_t axis) const {
    if (axis >= m_info.max_threads_per_group.size()) {
        return 1u;
    }
    return std::max<uint32_t>(1u, m_info.max_threads_per_group[axis]);
}

uint32_t GfxPartitioningModel::linear_thread_budget(uint32_t hard_cap) const {
    uint32_t budget = std::min(max_threads_for_axis(0), max_total_threads());
    if (hard_cap > 0) {
        budget = std::min(budget, hard_cap);
    }
    return std::max<uint32_t>(1u, budget);
}

uint32_t GfxPartitioningModel::align_thread_count(uint32_t requested_threads, uint32_t hard_cap) const {
    const uint32_t budget = linear_thread_budget(hard_cap);
    uint32_t threads = std::max<uint32_t>(1u, std::min(requested_threads, budget));
    const uint32_t wave = wave_width();
    if (wave <= 1 || threads < wave) {
        return threads;
    }
    threads = (threads / wave) * wave;
    return std::max<uint32_t>(wave, std::min(threads, budget));
}

uint32_t GfxPartitioningModel::select_1d_thread_count(uint64_t work_items,
                                                      uint32_t wide_work_waves,
                                                      uint64_t medium_work_threshold,
                                                      uint64_t wide_work_threshold,
                                                      uint32_t hard_cap) const {
    const uint32_t budget = linear_thread_budget(hard_cap);
    const uint32_t wave = wave_width();
    if (budget <= 1) {
        return 1;
    }

    uint32_t threads = std::min<uint32_t>(budget, wave);
    if (wave <= 1) {
        if (work_items >= wide_work_threshold) {
            threads = budget;
        } else if (work_items >= medium_work_threshold) {
            threads = std::min<uint32_t>(budget, std::max<uint32_t>(32u, budget / 2));
        } else {
            threads = std::min<uint32_t>(budget, 32u);
        }
        return std::max<uint32_t>(1u, threads);
    }

    const uint32_t supported_waves = std::max<uint32_t>(1u, budget / wave);
    const uint32_t target_waves = std::max<uint32_t>(1u, std::min(wide_work_waves, supported_waves));
    if (work_items >= wide_work_threshold && target_waves > 1) {
        threads = wave * target_waves;
    } else if (work_items >= medium_work_threshold && budget >= wave) {
        threads = wave;
    }
    return align_thread_count(threads, hard_cap);
}

std::vector<GfxWorkgroupShape> GfxPartitioningModel::enumerate_2d_thread_shapes(uint64_t work_items_h,
                                                                                 uint64_t work_items_w,
                                                                                 uint32_t wide_work_waves,
                                                                                 size_t max_results,
                                                                                 uint32_t hard_cap) const {
    std::vector<RankedWorkgroupShape> ranked;
    const uint32_t wave = wave_width();
    const uint32_t total_budget = linear_thread_budget(hard_cap);
    const uint32_t supported_waves = wave > 0 ? std::max<uint32_t>(1u, total_budget / std::max<uint32_t>(wave, 1u)) : 1u;
    const uint32_t target_waves = wave > 1 ? std::max<uint32_t>(1u, std::min(wide_work_waves, supported_waves)) : 1u;
    const uint64_t target_threads = std::min<uint64_t>(total_budget,
                                                       wave > 1 ? static_cast<uint64_t>(wave) * target_waves : total_budget);
    const uint64_t min_threads = std::min<uint64_t>(total_budget,
                                                    wave > 1 ? std::max<uint32_t>(8u, wave / 2) : 1u);
    const auto x_values = enumerate_axis_sizes(max_threads_for_axis(0), work_items_w);
    const auto y_values = enumerate_axis_sizes(max_threads_for_axis(1), work_items_h);
    const uint64_t effective_h = std::max<uint64_t>(1u, work_items_h);
    const uint64_t effective_w = std::max<uint64_t>(1u, work_items_w);

    ranked.reserve(x_values.size() * y_values.size());
    for (uint32_t x : x_values) {
        for (uint32_t y : y_values) {
            GfxWorkgroupShape shape{x, y, 1};
            if (!supports(shape) || shape.total_threads() > total_budget || shape.total_threads() < min_threads) {
                continue;
            }

            const uint64_t covered_h = ceil_div_u64(effective_h, y) * static_cast<uint64_t>(y);
            const uint64_t covered_w = ceil_div_u64(effective_w, x) * static_cast<uint64_t>(x);
            RankedWorkgroupShape candidate;
            candidate.shape = shape;
            candidate.thread_gap = target_threads > shape.total_threads() ? target_threads - shape.total_threads()
                                                                          : shape.total_threads() - target_threads;
            candidate.aspect_gap = effective_w * static_cast<uint64_t>(y) > effective_h * static_cast<uint64_t>(x)
                                       ? effective_w * static_cast<uint64_t>(y) - effective_h * static_cast<uint64_t>(x)
                                       : effective_h * static_cast<uint64_t>(x) - effective_w * static_cast<uint64_t>(y);
            candidate.tail_waste = covered_h * covered_w - effective_h * effective_w;
            ranked.push_back(candidate);
        }
    }

    std::sort(ranked.begin(), ranked.end(), is_better_shape);
    std::vector<GfxWorkgroupShape> result;
    result.reserve(std::min(max_results, ranked.size()));
    for (const auto& candidate : ranked) {
        if (result.size() >= max_results) {
            break;
        }
        result.push_back(candidate.shape);
    }
    return result;
}

bool GfxPartitioningModel::supports(const GfxWorkgroupShape& shape) const {
    return shape.x > 0 && shape.y > 0 && shape.z > 0 && shape.total_threads() <= max_total_threads() &&
           shape.x <= max_threads_for_axis(0) && shape.y <= max_threads_for_axis(1) &&
           shape.z <= max_threads_for_axis(2);
}

std::vector<GfxWorkgroupShape> GfxPartitioningModel::filter_supported(
    const std::vector<GfxWorkgroupShape>& candidates) const {
    std::vector<GfxWorkgroupShape> supported;
    supported.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (supports(candidate)) {
            supported.push_back(candidate);
        }
    }
    return supported;
}

GfxPartitioningDeviceInfo query_partitioning_device_info(const GpuBufferManager* buffer_manager) {
    if (buffer_manager) {
        if (const auto info = buffer_manager->query_execution_device_info()) {
            return make_info_from_device_info(*info);
        }
    }
    return make_default_partitioning_device_info(GpuBackend::Metal);
}

GfxPartitioningDeviceInfo make_default_partitioning_device_info(GpuBackend backend) {
    GfxPartitioningDeviceInfo info{};
    info.backend = backend;
    info.preferred_simd_width = 32;
    info.subgroup_size = 32;
    if (backend == GpuBackend::Vulkan) {
        info.max_total_threads_per_group = 128;
        info.max_threads_per_group = {128, 128, 64};
        info.device_key = "vulkan:default";
    } else {
        info.max_total_threads_per_group = 256;
        info.max_threads_per_group = {256, 256, 64};
        info.device_key = "metal:default";
    }
    return info;
}

uint64_t bucketize_partition_work(uint64_t value) {
    uint64_t bucket = 1;
    while (bucket < std::max<uint64_t>(1, value)) {
        bucket <<= 1;
    }
    return bucket;
}

}  // namespace gfx_plugin
}  // namespace ov
