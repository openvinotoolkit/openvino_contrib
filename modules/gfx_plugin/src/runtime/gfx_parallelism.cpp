// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_parallelism.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace ov {
namespace gfx_plugin {

namespace {

struct MatMulCacheKey {
    GpuBackend backend = GpuBackend::Metal;
    std::string device_key;
    ov::Shape output_shape;

    bool operator==(const MatMulCacheKey& other) const {
        return backend == other.backend && device_key == other.device_key && output_shape == other.output_shape;
    }
};

struct MatMulCacheKeyHash {
    size_t operator()(const MatMulCacheKey& key) const {
        size_t h = static_cast<size_t>(key.backend);
        h ^= std::hash<std::string>{}(key.device_key) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (const auto dim : key.output_shape) {
            h ^= std::hash<size_t>{}(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

class MatMulTuningCache {
public:
    static MatMulTuningCache& instance() {
        static MatMulTuningCache cache;
        return cache;
    }

    std::optional<MatMulParallelismPlan> find(const MatMulCacheKey& key) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_entries.find(key);
        if (it == m_entries.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    void store(const MatMulCacheKey& key, const MatMulParallelismPlan& plan) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries[key] = plan;
    }

private:
    mutable std::mutex m_mutex;
    std::unordered_map<MatMulCacheKey, MatMulParallelismPlan, MatMulCacheKeyHash> m_entries;
};

struct ChunkCacheKey {
    GpuBackend backend = GpuBackend::Metal;
    std::string device_key;
    std::string op_kind;
    uint64_t total_bucket = 0;
    uint64_t work_bucket = 0;

    bool operator==(const ChunkCacheKey& other) const {
        return backend == other.backend && device_key == other.device_key && op_kind == other.op_kind &&
               total_bucket == other.total_bucket && work_bucket == other.work_bucket;
    }
};

struct ChunkCacheKeyHash {
    size_t operator()(const ChunkCacheKey& key) const {
        size_t h = static_cast<size_t>(key.backend);
        h ^= std::hash<std::string>{}(key.device_key) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<std::string>{}(key.op_kind) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.total_bucket) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.work_bucket) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class ChunkTuningCache {
public:
    static ChunkTuningCache& instance() {
        static ChunkTuningCache cache;
        return cache;
    }

    std::optional<ChunkDispatchPlan> find(const ChunkCacheKey& key) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_entries.find(key);
        if (it == m_entries.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    void store(const ChunkCacheKey& key, const ChunkDispatchPlan& plan) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries[key] = plan;
    }

private:
    mutable std::mutex m_mutex;
    std::unordered_map<ChunkCacheKey, ChunkDispatchPlan, ChunkCacheKeyHash> m_entries;
};

struct Conv2DDirectCacheKey {
    GpuBackend backend = GpuBackend::Metal;
    std::string device_key;
    ov::Shape output_shape;
    uint64_t input_channels = 0;
    uint64_t output_channels = 0;
    uint64_t kernel_work = 0;
    bool stride2 = false;

    bool operator==(const Conv2DDirectCacheKey& other) const {
        return backend == other.backend && device_key == other.device_key && output_shape == other.output_shape &&
               input_channels == other.input_channels && output_channels == other.output_channels &&
               kernel_work == other.kernel_work && stride2 == other.stride2;
    }
};

struct Conv2DDirectCacheKeyHash {
    size_t operator()(const Conv2DDirectCacheKey& key) const {
        size_t h = static_cast<size_t>(key.backend);
        h ^= std::hash<std::string>{}(key.device_key) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (const auto dim : key.output_shape) {
            h ^= std::hash<size_t>{}(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        h ^= std::hash<uint64_t>{}(key.input_channels) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.output_channels) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.kernel_work) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(key.stride2) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class Conv2DDirectTuningCache {
public:
    static Conv2DDirectTuningCache& instance() {
        static Conv2DDirectTuningCache cache;
        return cache;
    }

    std::optional<Conv2DDirectPlan> find(const Conv2DDirectCacheKey& key) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_entries.find(key);
        if (it == m_entries.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    void store(const Conv2DDirectCacheKey& key, const Conv2DDirectPlan& plan) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries[key] = plan;
    }

private:
    mutable std::mutex m_mutex;
    std::unordered_map<Conv2DDirectCacheKey, Conv2DDirectPlan, Conv2DDirectCacheKeyHash> m_entries;
};

uint64_t shape_prefix_product(const ov::Shape& shape) {
    if (shape.size() <= 2) {
        return 1;
    }
    uint64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= std::max<uint64_t>(1, static_cast<uint64_t>(shape[i]));
    }
    return batch;
}

bool supports_candidate(const GfxParallelismCaps& caps, uint32_t h, uint32_t w) {
    const uint64_t total = static_cast<uint64_t>(h) * static_cast<uint64_t>(w);
    return h > 0 && w > 0 && total <= caps.max_total_threads_per_group && h <= caps.max_threads_per_group[1] &&
           w <= caps.max_threads_per_group[0];
}

std::string make_device_key_fallback(const GfxParallelismCaps& caps) {
    std::ostringstream os;
    os << static_cast<int>(caps.backend) << ':' << caps.preferred_simd_width << ':' << caps.subgroup_size << ':'
       << caps.max_total_threads_per_group << ':' << caps.max_threads_per_group[0] << ':' << caps.max_threads_per_group[1]
       << ':' << caps.max_threads_per_group[2];
    return os.str();
}

MatMulCacheKey make_cache_key(const GfxParallelismCaps& caps, const ov::Shape& output_shape) {
    MatMulCacheKey key;
    key.backend = caps.backend;
    key.device_key = caps.device_key.empty() ? make_device_key_fallback(caps) : caps.device_key;
    key.output_shape = output_shape;
    return key;
}

uint64_t bucketize(uint64_t value) {
    uint64_t bucket = 1;
    while (bucket < std::max<uint64_t>(1, value)) {
        bucket <<= 1;
    }
    return bucket;
}

ChunkCacheKey make_chunk_cache_key(const GfxParallelismCaps& caps,
                                   const std::string& op_kind,
                                   uint64_t total_elems,
                                   uint64_t work_per_elem) {
    ChunkCacheKey key;
    key.backend = caps.backend;
    key.device_key = caps.device_key.empty() ? make_device_key_fallback(caps) : caps.device_key;
    key.op_kind = op_kind;
    key.total_bucket = bucketize(total_elems);
    key.work_bucket = bucketize(work_per_elem);
    return key;
}

Conv2DDirectCacheKey make_conv2d_direct_cache_key(const GfxParallelismCaps& caps,
                                                  const ov::Shape& output_shape,
                                                  uint64_t input_channels,
                                                  uint64_t output_channels,
                                                  uint64_t kernel_work,
                                                  bool stride2) {
    Conv2DDirectCacheKey key;
    key.backend = caps.backend;
    key.device_key = caps.device_key.empty() ? make_device_key_fallback(caps) : caps.device_key;
    key.output_shape = output_shape;
    key.input_channels = input_channels;
    key.output_channels = output_channels;
    key.kernel_work = kernel_work;
    key.stride2 = stride2;
    return key;
}

GfxParallelismCaps make_caps_from_device_info(const GpuExecutionDeviceInfo& info) {
    GfxParallelismCaps caps{};
    caps.backend = info.backend;
    caps.device_key = info.device_key;
    caps.preferred_simd_width = std::max<uint32_t>(info.preferred_simd_width, 1u);
    caps.subgroup_size = std::max<uint32_t>(info.subgroup_size, 1u);
    caps.max_total_threads_per_group = std::max<uint32_t>(info.max_total_threads_per_group, 1u);
    caps.max_threads_per_group = {std::max<uint32_t>(info.max_threads_per_group[0], 1u),
                                  std::max<uint32_t>(info.max_threads_per_group[1], 1u),
                                  std::max<uint32_t>(info.max_threads_per_group[2], 1u)};
    return caps;
}

GfxParallelismCaps make_default_caps(GpuBackend backend) {
    GfxParallelismCaps caps{};
    caps.backend = backend;
    caps.preferred_simd_width = 32;
    caps.subgroup_size = 32;
    if (backend == GpuBackend::Vulkan) {
        caps.device_key = "vulkan:default";
        caps.max_total_threads_per_group = 128;
        caps.max_threads_per_group = {128, 128, 64};
    } else {
        caps.device_key = "metal:default";
        caps.max_total_threads_per_group = 64;
        caps.max_threads_per_group = {64, 64, 64};
    }
    return caps;
}

}  // namespace

GfxParallelismCaps query_parallelism_caps(const GpuBufferManager* buffer_manager) {
    if (buffer_manager) {
        if (const auto info = buffer_manager->query_execution_device_info()) {
            return make_caps_from_device_info(*info);
        }
    }
    return make_default_caps(GpuBackend::Metal);
}

std::vector<MatMulParallelismPlan> enumerate_matmul_parallelism_candidates(const GfxParallelismCaps& caps,
                                                                           const ov::Shape& output_shape) {
    std::vector<MatMulParallelismPlan> plans;
    if (output_shape.size() < 2) {
        return plans;
    }

    const uint64_t batch = shape_prefix_product(output_shape);
    const uint64_t m = std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 2]));
    const uint64_t n = std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 1]));
    const uint64_t total = batch * m * n;
    if (total < 1024) {
        return plans;
    }

    const uint32_t wave = std::max<uint32_t>(std::max(caps.subgroup_size, caps.preferred_simd_width), 1u);
    const std::array<std::array<uint32_t, 2>, 8> candidates = {{
        {{8, 8}},
        {{8, 4}},
        {{4, 8}},
        {{4, 4}},
        {{8, 2}},
        {{2, 8}},
        {{4, 2}},
        {{2, 4}},
    }};

    for (const auto& candidate : candidates) {
        const uint32_t h = candidate[0];
        const uint32_t w = candidate[1];
        if (!supports_candidate(caps, h, w)) {
            continue;
        }
        const uint64_t threads = static_cast<uint64_t>(h) * static_cast<uint64_t>(w);
        if (threads > std::max<uint32_t>(wave * 2, 64u) && !(h == 8 && w == 8)) {
            continue;
        }
        MatMulParallelismPlan plan;
        plan.prefer_parallel = true;
        plan.variant = "matmul_" + std::to_string(h) + "x" + std::to_string(w);
        plan.dispatch.enabled = true;
        plan.dispatch.loop_dims = 5;
        plan.dispatch.tile_h = h;
        plan.dispatch.tile_w = w;
        plan.dispatch.threads_h = h;
        plan.dispatch.threads_w = w;
        plans.push_back(plan);
    }
    return plans;
}

MatMulParallelismPlan select_matmul_parallelism(const GfxParallelismCaps& caps, const ov::Shape& output_shape) {
    const auto key = make_cache_key(caps, output_shape);
    if (auto cached = MatMulTuningCache::instance().find(key)) {
        return *cached;
    }

    const auto plans = enumerate_matmul_parallelism_candidates(caps, output_shape);
    if (plans.empty()) {
        return {};
    }
    const auto& chosen = plans.front();
    MatMulTuningCache::instance().store(key, chosen);
    return chosen;
}

void remember_matmul_parallelism(const GfxParallelismCaps& caps,
                                 const ov::Shape& output_shape,
                                 const MatMulParallelismPlan& plan) {
    if (!plan.prefer_parallel) {
        return;
    }
    MatMulTuningCache::instance().store(make_cache_key(caps, output_shape), plan);
}

ChunkDispatchPlan select_chunk_dispatch_plan(const GfxParallelismCaps& caps,
                                             const std::string& op_kind,
                                             uint64_t total_elems,
                                             uint64_t work_per_elem) {
    const auto key = make_chunk_cache_key(caps, op_kind, total_elems, work_per_elem);
    if (auto cached = ChunkTuningCache::instance().find(key)) {
        return *cached;
    }

    ChunkDispatchPlan plan;
    plan.threads_per_group = std::min<uint32_t>(64u, std::max<uint32_t>(1u, caps.max_total_threads_per_group));

    uint32_t elems = 1024;
    if (caps.backend == GpuBackend::Metal) {
        elems = work_per_elem >= 256 ? 4096 : 8192;
    } else {
        if (work_per_elem >= 512) {
            elems = 1024;
        } else if (work_per_elem >= 128) {
            elems = 2048;
        } else {
            elems = 4096;
        }
    }

    if (total_elems <= 4096) {
        elems = static_cast<uint32_t>(std::max<uint64_t>(1024, bucketize(total_elems)));
    } else if (total_elems <= 16384) {
        elems = std::min<uint32_t>(elems, 4096u);
    }

    plan.elems_per_dispatch = elems;
    plan.variant = op_kind + "_chunk_" + std::to_string(plan.elems_per_dispatch);
    ChunkTuningCache::instance().store(key, plan);
    return plan;
}

Conv2DDirectPlan select_conv2d_direct_plan(const GfxParallelismCaps& caps,
                                           const ov::Shape& output_shape,
                                           uint64_t input_channels,
                                           uint64_t output_channels,
                                           uint64_t kernel_work,
                                           bool stride2) {
    const auto key =
        make_conv2d_direct_cache_key(caps, output_shape, input_channels, output_channels, kernel_work, stride2);
    if (auto cached = Conv2DDirectTuningCache::instance().find(key)) {
        return *cached;
    }

    Conv2DDirectPlan plan;
    const uint32_t wave = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
    plan.threads_per_group = std::min<uint32_t>(std::max<uint32_t>(wave, 64u),
                                                std::max<uint32_t>(1u, caps.max_total_threads_per_group));
    plan.output_channel_block = 1;
    plan.variant = "conv2d_direct_oc1_tg" + std::to_string(plan.threads_per_group);

    if (caps.backend == GpuBackend::Vulkan &&
        output_shape.size() == 4) {
        const uint64_t spatial = static_cast<uint64_t>(output_shape[2]) * static_cast<uint64_t>(output_shape[3]);
        const bool prefer_oc2_variant = !stride2 && spatial <= 1024 && input_channels >= 128 &&
                                        output_channels >= 128 && kernel_work >= 1152;
        const bool prefer_dense_stride2_variant = stride2 && spatial <= 2048 && spatial >= 512 &&
                                                  input_channels >= 96 && output_channels >= 96 &&
                                                  kernel_work >= 864;
        const bool prefer_dense_stride2_wide_variant = prefer_dense_stride2_variant &&
                                                       spatial >= 1024 &&
                                                       input_channels >= 128 &&
                                                       output_channels >= 128;
        const bool prefer_wide_stride2_variant = stride2 && spatial >= 4096 && kernel_work >= 512 &&
                                                 input_channels >= 32 && output_channels >= 32;
        const bool can_xy32x2 = caps.max_threads_per_group[0] >= 32 && caps.max_threads_per_group[1] >= 2 &&
                                caps.max_total_threads_per_group >= 64;
        const bool can_xy16x4 = caps.max_threads_per_group[0] >= 16 && caps.max_threads_per_group[1] >= 4 &&
                                caps.max_total_threads_per_group >= 64;
        const bool can_xy8x8 = caps.max_threads_per_group[0] >= 8 && caps.max_threads_per_group[1] >= 8 &&
                               caps.max_total_threads_per_group >= 64;
        if (kernel_work >= 288 && !prefer_oc2_variant) {
            if (stride2) {
                if (prefer_dense_stride2_wide_variant && can_xy32x2) {
                    plan.threads_per_group = 64;
                    plan.variant = "conv2d_direct_xy32x2_dense_s2";
                } else if (prefer_dense_stride2_variant && can_xy16x4) {
                    plan.threads_per_group = 64;
                    plan.variant = "conv2d_direct_xy16x4_dense_s2";
                } else if (prefer_wide_stride2_variant && can_xy32x2) {
                    plan.threads_per_group = 64;
                    plan.variant = "conv2d_direct_xy32x2_dense_s2";
                } else if (spatial >= 4096 && kernel_work >= 512 && can_xy16x4) {
                    plan.threads_per_group = 64;
                    plan.variant = "conv2d_direct_xy16x4";
                } else if (spatial >= 2048 && can_xy32x2) {
                    plan.threads_per_group = 64;
                    plan.variant = "conv2d_direct_xy32x2";
                } else if (spatial >= 1024 && can_xy8x8) {
                    plan.threads_per_group = 64;
                    plan.variant = "conv2d_direct_xy8x8";
                }
            } else if (kernel_work <= 288 && spatial >= 1024 && can_xy8x8) {
                plan.threads_per_group = 64;
                plan.variant = "conv2d_direct_xy8x8";
            } else if (spatial >= 16384 && kernel_work >= 768 && can_xy32x2) {
                plan.threads_per_group = 64;
                plan.variant = "conv2d_direct_xy32x2";
            } else if (spatial >= 2048 && can_xy16x4) {
                plan.threads_per_group = 64;
                plan.variant = "conv2d_direct_xy16x4";
            } else if (spatial >= 256 && can_xy8x8) {
                plan.threads_per_group = 64;
                plan.variant = "conv2d_direct_xy8x8";
            }
        }
        // Adreno regresses badly on large feature maps with the wider OC-block
        // kernel. Keep the variant available, but only allow it on small late
        // layers until real autotuning is wired on top of this selector.
        if (prefer_oc2_variant && plan.variant.rfind("conv2d_direct_xy", 0) != 0) {
            plan.output_channel_block = 2;
            plan.variant = "conv2d_direct_oc2_tg" + std::to_string(plan.threads_per_group);
        }
    } else if (caps.backend == GpuBackend::Metal &&
               output_shape.size() == 4 &&
               input_channels >= 64 &&
               output_channels >= 128 &&
               kernel_work >= 576) {
        plan.output_channel_block = 2;
        plan.variant = "conv2d_direct_oc2_tg" + std::to_string(plan.threads_per_group);
    }

    if (plan.output_channel_block == 1 && plan.variant.rfind("conv2d_direct_xy", 0) != 0) {
        plan.variant = "conv2d_direct_oc1_tg" + std::to_string(plan.threads_per_group);
    }

    Conv2DDirectTuningCache::instance().store(key, plan);
    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
