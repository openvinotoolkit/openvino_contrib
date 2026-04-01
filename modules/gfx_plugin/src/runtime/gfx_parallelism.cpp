// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_parallelism.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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
        static auto* cache = new MatMulTuningCache();
        return *cache;
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

struct ConvParallelCacheKey {
    GpuBackend backend = GpuBackend::Metal;
    std::string device_key;
    ov::Shape output_shape;
    uint64_t input_channels = 0;
    uint64_t output_channels = 0;
    uint64_t kernel_work = 0;
    bool stride2 = false;
    bool depthwise = false;

    bool operator==(const ConvParallelCacheKey& other) const {
        return backend == other.backend && device_key == other.device_key && output_shape == other.output_shape &&
               input_channels == other.input_channels && output_channels == other.output_channels &&
               kernel_work == other.kernel_work && stride2 == other.stride2 && depthwise == other.depthwise;
    }
};

struct ConvParallelCacheKeyHash {
    size_t operator()(const ConvParallelCacheKey& key) const {
        size_t h = static_cast<size_t>(key.backend);
        h ^= std::hash<std::string>{}(key.device_key) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (const auto dim : key.output_shape) {
            h ^= std::hash<size_t>{}(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        h ^= std::hash<uint64_t>{}(key.input_channels) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.output_channels) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>{}(key.kernel_work) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(key.stride2) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(key.depthwise) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class ConvParallelTuningCache {
public:
    static ConvParallelTuningCache& instance() {
        static auto* cache = new ConvParallelTuningCache();
        return *cache;
    }

    std::optional<ConvParallelismPlan> find(const ConvParallelCacheKey& key) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_entries.find(key);
        if (it == m_entries.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    void store(const ConvParallelCacheKey& key, const ConvParallelismPlan& plan) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_entries[key] = plan;
    }

private:
    mutable std::mutex m_mutex;
    std::unordered_map<ConvParallelCacheKey, ConvParallelismPlan, ConvParallelCacheKeyHash> m_entries;
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
        static auto* cache = new ChunkTuningCache();
        return *cache;
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
        static auto* cache = new Conv2DDirectTuningCache();
        return *cache;
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

struct MatMulDims {
    uint64_t batch = 1;
    uint64_t m = 1;
    uint64_t n = 1;
    uint64_t total = 1;
};

MatMulDims extract_matmul_dims(const ov::Shape& output_shape) {
    MatMulDims dims;
    if (output_shape.size() < 2) {
        return dims;
    }
    dims.batch = shape_prefix_product(output_shape);
    dims.m = std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 2]));
    dims.n = std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 1]));
    dims.total = dims.batch * dims.m * dims.n;
    return dims;
}

std::string make_device_key_fallback(const GfxParallelismCaps& caps) {
    std::ostringstream os;
    os << static_cast<int>(caps.backend) << ':' << gpu_device_family_name(caps.device_family) << ':'
       << caps.preferred_simd_width << ':' << caps.subgroup_size << ':'
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

ConvParallelCacheKey make_conv_parallel_cache_key(const GfxParallelismCaps& caps,
                                                  const ov::Shape& output_shape,
                                                  uint64_t input_channels,
                                                  uint64_t output_channels,
                                                  uint64_t kernel_work,
                                                  bool stride2,
                                                  bool depthwise) {
    ConvParallelCacheKey key;
    key.backend = caps.backend;
    key.device_key = caps.device_key.empty() ? make_device_key_fallback(caps) : caps.device_key;
    key.output_shape = output_shape;
    key.input_channels = input_channels;
    key.output_channels = output_channels;
    key.kernel_work = kernel_work;
    key.stride2 = stride2;
    key.depthwise = depthwise;
    return key;
}

uint64_t bucketize(uint64_t value) {
    uint64_t bucket = 1;
    while (bucket < std::max<uint64_t>(1, value)) {
        bucket <<= 1;
    }
    return bucket;
}

uint64_t ceil_div_u64(uint64_t value, uint64_t divisor) {
    return divisor == 0 ? 0 : ((value + divisor - 1) / divisor);
}

double clamp_double(double value, double lo, double hi) {
    return std::max(lo, std::min(value, hi));
}

uint32_t clamp_threadgroup_candidate(const GfxParallelismCaps& caps, uint32_t candidate) {
    const uint32_t max_x = std::max<uint32_t>(1u, caps.max_threads_per_group[0]);
    const uint32_t max_total = std::max<uint32_t>(1u, caps.max_total_threads_per_group);
    return std::max<uint32_t>(1u, std::min(candidate, std::min(max_x, max_total)));
}

std::vector<uint32_t> enumerate_hardware_relative_threadgroups(const GfxParallelismCaps& caps) {
    const uint32_t wave = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
    const uint32_t max_threads = clamp_threadgroup_candidate(caps, std::max<uint32_t>(1u, caps.max_total_threads_per_group));
    std::vector<uint32_t> candidates;
    auto add = [&](uint32_t value) {
        const uint32_t clamped = clamp_threadgroup_candidate(caps, value);
        if (std::find(candidates.begin(), candidates.end(), clamped) == candidates.end()) {
            candidates.push_back(clamped);
        }
    };

    add(max_threads);
    add(max_threads / 2);
    add(max_threads / 4);
    add(wave * 2);
    add(wave);
    add((wave + 1) / 2);
    add(1);

    candidates.erase(std::remove(candidates.begin(), candidates.end(), 0), candidates.end());
    std::sort(candidates.begin(), candidates.end(), std::greater<uint32_t>());
    return candidates;
}

uint32_t select_hardware_relative_threadgroup(const GfxParallelismCaps& caps,
                                              uint64_t total_elems,
                                              uint64_t work_per_elem,
                                              uint32_t default_threads) {
    const auto candidates = enumerate_hardware_relative_threadgroups(caps);
    if (candidates.empty()) {
        return std::max<uint32_t>(1u, default_threads);
    }

    const uint32_t wave = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
    const uint32_t clamped_default = clamp_threadgroup_candidate(caps, std::max<uint32_t>(1u, default_threads));
    const uint64_t work_bucket = bucketize(std::max<uint64_t>(1, work_per_elem));
    const uint64_t total_bucket = bucketize(std::max<uint64_t>(1, total_elems));

    uint32_t target = clamped_default;
    if (work_bucket >= 512) {
        target = wave;
    } else if (work_bucket >= 128) {
        target = wave + std::max<uint32_t>(1u, wave / 2);
    } else if (work_bucket <= 16 && total_bucket >= 4096) {
        target = clamp_threadgroup_candidate(caps, wave * 2);
    }
    if (total_bucket <= wave) {
        target = clamp_threadgroup_candidate(caps, static_cast<uint32_t>(std::min<uint64_t>(total_bucket, target)));
    }

    auto best = candidates.front();
    uint64_t best_score = std::numeric_limits<uint64_t>::max();
    for (const auto candidate : candidates) {
        const uint64_t distance = candidate > target ? candidate - target : target - candidate;
        const uint64_t oversubscribe_penalty = candidate > total_bucket ? (candidate - total_bucket) : 0;
        const uint64_t score = distance * 8 + oversubscribe_penalty;
        if (score < best_score) {
            best = candidate;
            best_score = score;
        }
    }
    return std::max<uint32_t>(1u, best);
}

double score_matmul_plan(const GfxParallelismCaps& caps,
                         const ov::Shape& output_shape,
                         const MatMulParallelismPlan& plan,
                         bool prefer_skinny_tiles) {
    const auto dims = extract_matmul_dims(output_shape);
    const uint32_t h = std::max<uint32_t>(1u, plan.dispatch.threads_h);
    const uint32_t w = std::max<uint32_t>(1u, plan.dispatch.threads_w);
    const uint32_t threads = std::max<uint32_t>(1u, h * w);
    const uint32_t wave = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
    uint32_t desired_threads =
        select_hardware_relative_threadgroup(caps, dims.total, std::max<uint64_t>(1, dims.m), wave);
    const double output_aspect =
        static_cast<double>(std::max<uint64_t>(dims.m, dims.n)) / static_cast<double>(std::max<uint64_t>(1, std::min(dims.m, dims.n)));
    const bool compute_dense = std::min<uint64_t>(dims.m, dims.n) >= static_cast<uint64_t>(wave) * 8u;
    if (dims.batch == 1 && compute_dense && output_aspect <= 2.0) {
        desired_threads = std::max(desired_threads, clamp_threadgroup_candidate(caps, wave * 4u));
    } else if (dims.batch == 1 && compute_dense && output_aspect <= 4.0) {
        desired_threads = std::max(desired_threads, clamp_threadgroup_candidate(caps, wave * 2u));
    }

    const double target_aspect = clamp_double(static_cast<double>(dims.n) / static_cast<double>(dims.m), 1.0 / 16.0, 16.0);
    const double tile_aspect = clamp_double(static_cast<double>(w) / static_cast<double>(h), 1.0 / 32.0, 32.0);
    const double aspect_error = std::abs(std::log2(clamp_double(tile_aspect / target_aspect, 1e-3, 1e3)));
    const double thread_error = std::abs(static_cast<double>(threads) - static_cast<double>(desired_threads));
    const double oversubscribe_h = h > dims.m ? static_cast<double>(h - dims.m) : 0.0;
    const double oversubscribe_w = w > dims.n ? static_cast<double>(w - dims.n) : 0.0;
    const double underfilled_wave_penalty =
        (threads < wave && dims.total >= static_cast<uint64_t>(wave) * 4u) ? static_cast<double>(wave - threads) : 0.0;
    const double wide_shape_penalty =
        (dims.n >= dims.m * 8u && h > 2u) ? static_cast<double>(h - 2u) * 8.0 : 0.0;
    const double tall_shape_penalty =
        (dims.m >= dims.n * 8u && w > 2u) ? static_cast<double>(w - 2u) * 8.0 : 0.0;
    const double skinny_bonus =
        (prefer_skinny_tiles && dims.n >= dims.m * 8u && w >= h) ? static_cast<double>(std::min<uint32_t>(w, desired_threads)) : 0.0;

    return aspect_error * 32.0 + thread_error + (oversubscribe_h + oversubscribe_w) * 64.0 +
           underfilled_wave_penalty * 2.0 + wide_shape_penalty + tall_shape_penalty - skinny_bonus;
}

void sort_matmul_candidates_by_shape(const GfxParallelismCaps& caps,
                                     const ov::Shape& output_shape,
                                     std::vector<MatMulParallelismPlan>& plans,
                                     bool prefer_skinny_tiles) {
    std::stable_sort(plans.begin(), plans.end(), [&](const MatMulParallelismPlan& lhs, const MatMulParallelismPlan& rhs) {
        const double lhs_score = score_matmul_plan(caps, output_shape, lhs, prefer_skinny_tiles);
        const double rhs_score = score_matmul_plan(caps, output_shape, rhs, prefer_skinny_tiles);
        if (lhs_score != rhs_score) {
            return lhs_score < rhs_score;
        }
        const uint32_t lhs_threads = std::max<uint32_t>(1u, lhs.dispatch.threads_h * lhs.dispatch.threads_w);
        const uint32_t rhs_threads = std::max<uint32_t>(1u, rhs.dispatch.threads_h * rhs.dispatch.threads_w);
        return lhs_threads > rhs_threads;
    });
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

struct ThreadPair {
    uint32_t h = 1;
    uint32_t w = 1;
};

ThreadPair choose_conv_thread_pair(const GfxParallelismCaps& caps,
                                   uint32_t target_threads,
                                   uint64_t out_h,
                                   uint64_t out_w,
                                   uint64_t spatial,
                                   uint64_t kernel_work,
                                   bool stride2,
                                   bool depthwise) {
    const uint32_t max_h = std::max<uint32_t>(1u, std::min<uint32_t>(caps.max_threads_per_group[1], target_threads));
    const uint32_t max_w = std::max<uint32_t>(1u, std::min<uint32_t>(caps.max_threads_per_group[0], target_threads));
    const double spatial_aspect = static_cast<double>(std::max<uint64_t>(1, out_w)) /
                                  static_cast<double>(std::max<uint64_t>(1, out_h));
    double target_aspect = spatial_aspect;
    if (stride2) {
        target_aspect *= 1.5;
    }
    if (depthwise) {
        target_aspect *= 0.5;
    }
    if (kernel_work >= 1024) {
        target_aspect = (target_aspect + 1.0) * 0.5;
    }
    target_aspect = clamp_double(target_aspect, 0.5, 4.0);

    ThreadPair best{};
    double best_score = std::numeric_limits<double>::infinity();
    for (uint32_t h = 1; h <= max_h; ++h) {
        for (uint32_t w = 1; w <= max_w; ++w) {
            const uint32_t threads = h * w;
            if (threads == 0 || threads > target_threads || threads > caps.max_total_threads_per_group) {
                continue;
            }
            const double aspect = static_cast<double>(w) / static_cast<double>(h);
            const double aspect_error = std::abs(std::log2(clamp_double(aspect / target_aspect, 1e-3, 1e3)));
            const double util_penalty = static_cast<double>(target_threads - threads);
            const double oversubscribe_penalty = threads > spatial ? static_cast<double>(threads - spatial) : 0.0;
            const double line_penalty =
                ((h == 1 || w == 1) && threads > 1 && out_h > 1 && out_w > 1) ? 6.0 : 0.0;
            const double score = util_penalty * 16.0 + aspect_error * 10.0 + oversubscribe_penalty * 2.0 +
                                 line_penalty;
            if (score < best_score) {
                best = ThreadPair{h, w};
                best_score = score;
            }
        }
    }
    return best;
}

ConvParallelismPlan make_conv_parallelism_plan(const GfxParallelismCaps& caps,
                                               const ov::Shape& output_shape,
                                               uint64_t input_channels,
                                               uint64_t output_channels,
                                               uint64_t kernel_work,
                                               bool stride2,
                                               bool depthwise,
                                               uint32_t default_threads) {
    ConvParallelismPlan plan;
    if (output_shape.size() < 4) {
        return plan;
    }

    const uint64_t out_h = std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 2]));
    const uint64_t out_w = std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 1]));
    const uint64_t spatial = shape_prefix_product(output_shape) * out_h * out_w;
    const uint64_t effective_work = std::max<uint64_t>(kernel_work, std::max<uint64_t>(1, input_channels));
    const uint32_t clamped_default = clamp_threadgroup_candidate(caps, std::max<uint32_t>(1u, default_threads));
    uint32_t target_threads = select_hardware_relative_threadgroup(caps, spatial, effective_work, clamped_default);
    const bool dense_parallel_workload =
        !depthwise && spatial >= static_cast<uint64_t>(clamped_default) * 4u && effective_work >= clamped_default;
    if (dense_parallel_workload) {
        target_threads = std::max(target_threads, clamped_default);
    }
    const auto pair =
        choose_conv_thread_pair(caps, target_threads, out_h, out_w, spatial, kernel_work, stride2, depthwise);

    plan.prefer_parallel = (shape_prefix_product(output_shape) * output_channels * out_h * out_w) >= 256;
    plan.variant = "conv_parallel_" + std::to_string(pair.h) + "x" + std::to_string(pair.w);
    plan.dispatch.enabled = true;
    plan.dispatch.loop_dims = 5;
    plan.dispatch.tile_h = pair.h;
    plan.dispatch.tile_w = pair.w;
    plan.dispatch.threads_h = pair.h;
    plan.dispatch.threads_w = pair.w;
    return plan;
}

GfxParallelismCaps make_caps_from_device_info(const GpuExecutionDeviceInfo& info) {
    GfxParallelismCaps caps{};
    caps.backend = info.backend;
    caps.device_family = info.device_family;
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
    caps.device_family = backend == GpuBackend::Metal ? GpuDeviceFamily::Apple : GpuDeviceFamily::Generic;
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

void append_matmul_candidate(const GfxParallelismCaps& caps,
                             uint32_t h,
                             uint32_t w,
                             uint32_t wave,
                             std::vector<MatMulParallelismPlan>& plans) {
    if (!supports_candidate(caps, h, w)) {
        return;
    }
    const uint64_t threads = static_cast<uint64_t>(h) * static_cast<uint64_t>(w);
    if (threads > std::max<uint32_t>(wave * 2, 64u) && !(h == 8 && w == 8)) {
        return;
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

std::vector<MatMulParallelismPlan> enumerate_matmul_parallelism_candidates_generic(const GfxParallelismCaps& caps,
                                                                                   const ov::Shape& output_shape,
                                                                                   bool include_skinny_candidates) {
    std::vector<MatMulParallelismPlan> plans;
    const auto dims = extract_matmul_dims(output_shape);
    if (output_shape.size() < 2) {
        return plans;
    }
    if (dims.total < 1024) {
        return plans;
    }

    const uint32_t wave = std::max<uint32_t>(std::max(caps.subgroup_size, caps.preferred_simd_width), 1u);
    const std::array<std::array<uint32_t, 2>, 8> base_candidates = {{
        {{8, 8}},
        {{8, 4}},
        {{4, 8}},
        {{4, 4}},
        {{8, 2}},
        {{2, 8}},
        {{4, 2}},
        {{2, 4}},
    }};

    for (const auto& candidate : base_candidates) {
        append_matmul_candidate(caps, candidate[0], candidate[1], wave, plans);
    }

    if (include_skinny_candidates) {
        const std::array<std::array<uint32_t, 2>, 10> skinny_candidates = {{
            {{1, 16}},
            {{16, 1}},
            {{1, 8}},
            {{8, 1}},
            {{2, 16}},
            {{16, 2}},
            {{1, 4}},
            {{4, 1}},
            {{1, 2}},
            {{2, 1}},
        }};
        for (const auto& candidate : skinny_candidates) {
            append_matmul_candidate(caps, candidate[0], candidate[1], wave, plans);
        }
    }
    return plans;
}

ChunkDispatchPlan select_chunk_dispatch_plan_generic(const GfxParallelismCaps& caps,
                                                     const std::string& op_kind,
                                                     uint64_t total_elems,
                                                     uint64_t work_per_elem) {
    ChunkDispatchPlan plan;
    const uint32_t wave = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
    plan.threads_per_group = std::min<uint32_t>(std::max<uint32_t>(wave, 64u),
                                                std::max<uint32_t>(1u, caps.max_total_threads_per_group));

    uint32_t elems = 1024;
    uint32_t max_elems = 16384;
    uint32_t target_dispatches = 16;
    if (caps.backend == GpuBackend::Metal) {
        elems = work_per_elem >= 256 ? 4096 : 8192;
        max_elems = 16384;
        target_dispatches = work_per_elem >= 256 ? 8u : 12u;
    } else {
        if (work_per_elem >= 1024) {
            elems = 4096;
            max_elems = 16384;
            target_dispatches = 8;
        } else if (work_per_elem >= 256) {
            elems = 8192;
            max_elems = 32768;
            target_dispatches = 8;
        } else if (work_per_elem >= 64) {
            elems = 16384;
            max_elems = 32768;
            target_dispatches = 12;
        } else {
            elems = 32768;
            max_elems = 65536;
            target_dispatches = 16;
        }
    }

    if (total_elems <= 16384) {
        elems = static_cast<uint32_t>(std::max<uint64_t>(1024, bucketize(total_elems)));
    } else {
        const uint64_t min_elems_for_budget = bucketize(ceil_div_u64(total_elems, target_dispatches));
        elems = std::max<uint32_t>(elems,
                                   static_cast<uint32_t>(std::min<uint64_t>(max_elems, min_elems_for_budget)));
    }
    elems = std::min<uint32_t>(elems, max_elems);

    plan.elems_per_dispatch = elems;
    plan.variant = op_kind + "_chunk_" + std::to_string(plan.elems_per_dispatch);
    return plan;
}

Conv2DDirectPlan select_conv2d_direct_plan_generic(const GfxParallelismCaps& caps,
                                                   const ov::Shape& output_shape,
                                                   uint64_t input_channels,
                                                   uint64_t output_channels,
                                                   uint64_t kernel_work,
                                                   bool stride2) {
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
    return plan;
}

class ParallelismStrategy {
public:
    virtual ~ParallelismStrategy() = default;

    virtual std::vector<MatMulParallelismPlan> enumerate_matmul(const GfxParallelismCaps& caps,
                                                                const ov::Shape& output_shape) const {
        return enumerate_matmul_parallelism_candidates_generic(caps, output_shape, /*include_skinny_candidates=*/false);
    }

    virtual ChunkDispatchPlan select_chunk_dispatch(const GfxParallelismCaps& caps,
                                                    const std::string& op_kind,
                                                    uint64_t total_elems,
                                                    uint64_t work_per_elem) const = 0;

    virtual ConvParallelismPlan select_conv_parallel(const GfxParallelismCaps& caps,
                                                     const ov::Shape& output_shape,
                                                     uint64_t input_channels,
                                                     uint64_t output_channels,
                                                     uint64_t kernel_work,
                                                     bool stride2,
                                                     bool depthwise) const = 0;

    virtual Conv2DDirectPlan select_conv2d_direct(const GfxParallelismCaps& caps,
                                                  const ov::Shape& output_shape,
                                                  uint64_t input_channels,
                                                  uint64_t output_channels,
                                                  uint64_t kernel_work,
                                                  bool stride2) const = 0;
};

class MetalParallelismStrategy final : public ParallelismStrategy {
public:
    ChunkDispatchPlan select_chunk_dispatch(const GfxParallelismCaps& caps,
                                            const std::string& op_kind,
                                            uint64_t total_elems,
                                            uint64_t work_per_elem) const override {
        return select_chunk_dispatch_plan_generic(caps, op_kind, total_elems, work_per_elem);
    }

    ConvParallelismPlan select_conv_parallel(const GfxParallelismCaps& caps,
                                             const ov::Shape& output_shape,
                                             uint64_t input_channels,
                                             uint64_t output_channels,
                                             uint64_t kernel_work,
                                             bool stride2,
                                             bool depthwise) const override {
        const uint32_t wave = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
        return make_conv_parallelism_plan(caps,
                                          output_shape,
                                          input_channels,
                                          output_channels,
                                          kernel_work,
                                          stride2,
                                          depthwise,
                                          wave);
    }

    Conv2DDirectPlan select_conv2d_direct(const GfxParallelismCaps& caps,
                                          const ov::Shape& output_shape,
                                          uint64_t input_channels,
                                          uint64_t output_channels,
                                          uint64_t kernel_work,
                                          bool stride2) const override {
        return select_conv2d_direct_plan_generic(caps,
                                                 output_shape,
                                                 input_channels,
                                                 output_channels,
                                                 kernel_work,
                                                 stride2);
    }
};

class VulkanParallelismStrategyBase : public ParallelismStrategy {
public:
    std::vector<MatMulParallelismPlan> enumerate_matmul(const GfxParallelismCaps& caps,
                                                        const ov::Shape& output_shape) const override {
        auto plans = enumerate_matmul_parallelism_candidates_generic(caps, output_shape, /*include_skinny_candidates=*/false);
        tune_matmul_candidates(caps, output_shape, plans);
        return plans;
    }

    ChunkDispatchPlan select_chunk_dispatch(const GfxParallelismCaps& caps,
                                            const std::string& op_kind,
                                            uint64_t total_elems,
                                            uint64_t work_per_elem) const final {
        auto plan = select_chunk_dispatch_plan_generic(caps, op_kind, total_elems, work_per_elem);
        tune_chunk_dispatch(caps, op_kind, total_elems, work_per_elem, plan);
        return plan;
    }

    ConvParallelismPlan select_conv_parallel(const GfxParallelismCaps& caps,
                                             const ov::Shape& output_shape,
                                             uint64_t input_channels,
                                             uint64_t output_channels,
                                             uint64_t kernel_work,
                                             bool stride2,
                                             bool depthwise) const final {
        const uint32_t wave = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
        auto plan = make_conv_parallelism_plan(caps,
                                               output_shape,
                                               input_channels,
                                               output_channels,
                                               kernel_work,
                                               stride2,
                                               depthwise,
                                               wave);
        tune_conv_parallel(caps,
                           output_shape,
                           input_channels,
                           output_channels,
                           kernel_work,
                           stride2,
                           depthwise,
                           plan);
        return plan;
    }

    Conv2DDirectPlan select_conv2d_direct(const GfxParallelismCaps& caps,
                                          const ov::Shape& output_shape,
                                          uint64_t input_channels,
                                          uint64_t output_channels,
                                          uint64_t kernel_work,
                                          bool stride2) const final {
        auto plan = select_conv2d_direct_plan_generic(caps,
                                                      output_shape,
                                                      input_channels,
                                                      output_channels,
                                                      kernel_work,
                                                      stride2);
        tune_conv2d_direct(caps, output_shape, input_channels, output_channels, kernel_work, stride2, plan);
        return plan;
    }

protected:
    virtual void tune_matmul_candidates(const GfxParallelismCaps& caps,
                                        const ov::Shape& output_shape,
                                        std::vector<MatMulParallelismPlan>& plans) const {
        sort_matmul_candidates_by_shape(caps, output_shape, plans, /*prefer_skinny_tiles=*/false);
    }

    virtual void tune_chunk_dispatch(const GfxParallelismCaps& /*caps*/,
                                     const std::string& /*op_kind*/,
                                     uint64_t /*total_elems*/,
                                     uint64_t /*work_per_elem*/,
                                     ChunkDispatchPlan& /*plan*/) const {}

    virtual void tune_conv_parallel(const GfxParallelismCaps& /*caps*/,
                                    const ov::Shape& /*output_shape*/,
                                    uint64_t /*input_channels*/,
                                    uint64_t /*output_channels*/,
                                    uint64_t /*kernel_work*/,
                                    bool /*stride2*/,
                                    bool /*depthwise*/,
                                    ConvParallelismPlan& /*plan*/) const {}

    virtual void tune_conv2d_direct(const GfxParallelismCaps& /*caps*/,
                                    const ov::Shape& /*output_shape*/,
                                    uint64_t /*input_channels*/,
                                    uint64_t /*output_channels*/,
                                    uint64_t /*kernel_work*/,
                                    bool /*stride2*/,
                                    Conv2DDirectPlan& /*plan*/) const {}
};

class GenericVulkanParallelismStrategy final : public VulkanParallelismStrategyBase {};

class AdrenoVulkanParallelismStrategy final : public VulkanParallelismStrategyBase {};

class BroadcomV3DParallelismStrategy final : public VulkanParallelismStrategyBase {
protected:
    std::vector<MatMulParallelismPlan> enumerate_matmul(const GfxParallelismCaps& caps,
                                                        const ov::Shape& output_shape) const override {
        const auto dims = extract_matmul_dims(output_shape);
        const bool include_skinny_candidates = dims.batch == 1 && dims.n >= dims.m * 8u;
        auto plans =
            enumerate_matmul_parallelism_candidates_generic(caps, output_shape, include_skinny_candidates);
        tune_matmul_candidates(caps, output_shape, plans);
        return plans;
    }

    void tune_matmul_candidates(const GfxParallelismCaps& caps,
                                const ov::Shape& output_shape,
                                std::vector<MatMulParallelismPlan>& plans) const override {
        const auto dims = extract_matmul_dims(output_shape);
        const bool prefer_skinny_tiles = dims.batch == 1 && dims.n >= dims.m * 8u;
        sort_matmul_candidates_by_shape(caps, output_shape, plans, prefer_skinny_tiles);
    }

    void tune_chunk_dispatch(const GfxParallelismCaps& caps,
                             const std::string& /*op_kind*/,
                             uint64_t total_elems,
                             uint64_t work_per_elem,
                             ChunkDispatchPlan& plan) const override {
        plan.threads_per_group =
            select_hardware_relative_threadgroup(caps, total_elems, work_per_elem, plan.threads_per_group);
    }

    void tune_conv2d_direct(const GfxParallelismCaps& caps,
                            const ov::Shape& output_shape,
                            uint64_t /*input_channels*/,
                            uint64_t /*output_channels*/,
                            uint64_t kernel_work,
                            bool /*stride2*/,
                            Conv2DDirectPlan& plan) const override {
        plan.output_channel_block = 1;
        plan.threads_per_group = select_hardware_relative_threadgroup(
            caps,
            shape_prefix_product(output_shape) *
                std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 2])) *
                std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 1])),
            kernel_work,
            plan.threads_per_group);
        plan.variant = "conv2d_direct_oc1_tg" + std::to_string(plan.threads_per_group);
    }

    void tune_conv_parallel(const GfxParallelismCaps& caps,
                            const ov::Shape& output_shape,
                            uint64_t input_channels,
                            uint64_t output_channels,
                            uint64_t kernel_work,
                            bool stride2,
                            bool depthwise,
                            ConvParallelismPlan& plan) const override {
        const uint32_t wave = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
        const uint64_t out_h =
            output_shape.size() >= 2 ? std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 2])) : 1;
        const uint64_t out_w =
            output_shape.size() >= 1 ? std::max<uint64_t>(1, static_cast<uint64_t>(output_shape[output_shape.size() - 1])) : 1;
        const uint64_t spatial = shape_prefix_product(output_shape) * out_h * out_w;
        uint32_t default_threads = wave;
        const bool compute_dense = kernel_work >= 288 && input_channels >= wave * 2u && output_channels >= wave * 2u;
        if (!depthwise && compute_dense && spatial >= 1024) {
            default_threads = clamp_threadgroup_candidate(caps, stride2 ? wave * 4u : wave * 2u);
            if (!stride2 &&
                kernel_work >= 512 &&
                spatial >= 4096 &&
                input_channels >= wave * 4u &&
                output_channels >= wave * 4u) {
                default_threads = clamp_threadgroup_candidate(caps, wave * 4u);
            }
        }
        plan = make_conv_parallelism_plan(caps,
                                          output_shape,
                                          input_channels,
                                          output_channels,
                                          kernel_work,
                                          stride2,
                                          depthwise,
                                          default_threads);
    }
};

const ParallelismStrategy& select_parallelism_strategy(const GfxParallelismCaps& caps) {
    static const MetalParallelismStrategy metal_strategy{};
    static const GenericVulkanParallelismStrategy generic_vulkan_strategy{};
    static const AdrenoVulkanParallelismStrategy adreno_vulkan_strategy{};
    static const BroadcomV3DParallelismStrategy broadcom_v3d_strategy{};

    if (caps.backend == GpuBackend::Metal) {
        return metal_strategy;
    }
    switch (caps.device_family) {
    case GpuDeviceFamily::QualcommAdreno:
        return adreno_vulkan_strategy;
    case GpuDeviceFamily::BroadcomV3D:
        return broadcom_v3d_strategy;
    case GpuDeviceFamily::Apple:
    case GpuDeviceFamily::Generic:
    default:
        return generic_vulkan_strategy;
    }
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
    return select_parallelism_strategy(caps).enumerate_matmul(caps, output_shape);
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
    MatMulTuningCache::instance().store(make_cache_key(caps, output_shape), plan);
}

ConvParallelismPlan select_conv_parallelism(const GfxParallelismCaps& caps,
                                            const ov::Shape& output_shape,
                                            uint64_t input_channels,
                                            uint64_t output_channels,
                                            uint64_t kernel_work,
                                            bool stride2,
                                            bool depthwise) {
    const auto key = make_conv_parallel_cache_key(caps,
                                                  output_shape,
                                                  input_channels,
                                                  output_channels,
                                                  kernel_work,
                                                  stride2,
                                                  depthwise);
    if (auto cached = ConvParallelTuningCache::instance().find(key)) {
        return *cached;
    }

    ConvParallelismPlan plan = select_parallelism_strategy(caps).select_conv_parallel(caps,
                                                                                       output_shape,
                                                                                       input_channels,
                                                                                       output_channels,
                                                                                       kernel_work,
                                                                                       stride2,
                                                                                       depthwise);
    ConvParallelTuningCache::instance().store(key, plan);
    return plan;
}

ChunkDispatchPlan select_chunk_dispatch_plan(const GfxParallelismCaps& caps,
                                             const std::string& op_kind,
                                             uint64_t total_elems,
                                             uint64_t work_per_elem) {
    const auto key = make_chunk_cache_key(caps, op_kind, total_elems, work_per_elem);
    if (auto cached = ChunkTuningCache::instance().find(key)) {
        return *cached;
    }

    ChunkDispatchPlan plan =
        select_parallelism_strategy(caps).select_chunk_dispatch(caps, op_kind, total_elems, work_per_elem);
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

    Conv2DDirectPlan plan = select_parallelism_strategy(caps).select_conv2d_direct(caps,
                                                                                    output_shape,
                                                                                    input_channels,
                                                                                    output_channels,
                                                                                    kernel_work,
                                                                                    stride2);
    Conv2DDirectTuningCache::instance().store(key, plan);
    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
