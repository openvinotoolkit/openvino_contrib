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

#include "runtime/gpu_buffer_manager.hpp"

namespace ov {
namespace gfx_plugin {

const char *conv_channel_block_accumulation_name(
    ConvChannelBlockAccumulation mode) {
  switch (mode) {
  case ConvChannelBlockAccumulation::Serial:
    return "serial";
  case ConvChannelBlockAccumulation::Fused:
  default:
    return "fused";
  }
}

namespace {

struct MatMulCacheKey {
  std::string profile_key;
  ov::Shape output_shape;

  bool operator==(const MatMulCacheKey &other) const {
    return profile_key == other.profile_key &&
           output_shape == other.output_shape;
  }
};

struct MatMulCacheKeyHash {
  size_t operator()(const MatMulCacheKey &key) const {
    size_t h = std::hash<std::string>{}(key.profile_key);
    h ^= 0x9e3779b9 + (h << 6) +
         (h >> 2);
    for (const auto dim : key.output_shape) {
      h ^= std::hash<size_t>{}(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

class MatMulTuningCache {
public:
  static MatMulTuningCache &instance() {
    static auto *cache = new MatMulTuningCache();
    return *cache;
  }

  std::optional<MatMulParallelismPlan> find(const MatMulCacheKey &key) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_entries.find(key);
    if (it == m_entries.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  void store(const MatMulCacheKey &key, const MatMulParallelismPlan &plan) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_entries[key] = plan;
  }

private:
  mutable std::mutex m_mutex;
  std::unordered_map<MatMulCacheKey, MatMulParallelismPlan, MatMulCacheKeyHash>
      m_entries;
};

struct ConvParallelCacheKey {
  std::string profile_key;
  bool supports_conv_output_channel_blocking = false;
  bool supports_conv_channel_block_spatial_tiling = false;
  bool conv_spatial_micro_tile_requires_large_output_area = false;
  ov::Shape output_shape;
  uint64_t input_channels = 0;
  uint64_t output_channels = 0;
  uint64_t kernel_work = 0;
  bool stride2 = false;
  bool depthwise = false;

  bool operator==(const ConvParallelCacheKey &other) const {
    return profile_key == other.profile_key &&
           supports_conv_output_channel_blocking ==
               other.supports_conv_output_channel_blocking &&
           supports_conv_channel_block_spatial_tiling ==
               other.supports_conv_channel_block_spatial_tiling &&
           conv_spatial_micro_tile_requires_large_output_area ==
               other.conv_spatial_micro_tile_requires_large_output_area &&
           output_shape == other.output_shape &&
           input_channels == other.input_channels &&
           output_channels == other.output_channels &&
           kernel_work == other.kernel_work && stride2 == other.stride2 &&
           depthwise == other.depthwise;
  }
};

struct ConvParallelCacheKeyHash {
  size_t operator()(const ConvParallelCacheKey &key) const {
    size_t h = std::hash<std::string>{}(key.profile_key);
    h ^= std::hash<bool>{}(key.supports_conv_output_channel_blocking) +
         0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(key.supports_conv_channel_block_spatial_tiling) +
         0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(
             key.conv_spatial_micro_tile_requires_large_output_area) +
         0x9e3779b9 + (h << 6) + (h >> 2);
    for (const auto dim : key.output_shape) {
      h ^= std::hash<size_t>{}(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    h ^= std::hash<uint64_t>{}(key.input_channels) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<uint64_t>{}(key.output_channels) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<uint64_t>{}(key.kernel_work) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<bool>{}(key.stride2) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(key.depthwise) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

class ConvParallelTuningCache {
public:
  static ConvParallelTuningCache &instance() {
    static auto *cache = new ConvParallelTuningCache();
    return *cache;
  }

  std::optional<ConvParallelismPlan>
  find(const ConvParallelCacheKey &key) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_entries.find(key);
    if (it == m_entries.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  void store(const ConvParallelCacheKey &key, const ConvParallelismPlan &plan) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_entries[key] = plan;
  }

private:
  mutable std::mutex m_mutex;
  std::unordered_map<ConvParallelCacheKey, ConvParallelismPlan,
                     ConvParallelCacheKeyHash>
      m_entries;
};

struct ChunkCacheKey {
  std::string profile_key;
  std::string op_kind;
  uint64_t total_bucket = 0;
  uint64_t work_bucket = 0;

  bool operator==(const ChunkCacheKey &other) const {
    return profile_key == other.profile_key &&
           op_kind == other.op_kind && total_bucket == other.total_bucket &&
           work_bucket == other.work_bucket;
  }
};

struct ChunkCacheKeyHash {
  size_t operator()(const ChunkCacheKey &key) const {
    size_t h = std::hash<std::string>{}(key.profile_key);
    h ^= std::hash<std::string>{}(key.op_kind) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<uint64_t>{}(key.total_bucket) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<uint64_t>{}(key.work_bucket) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    return h;
  }
};

class ChunkTuningCache {
public:
  static ChunkTuningCache &instance() {
    static auto *cache = new ChunkTuningCache();
    return *cache;
  }

  std::optional<ChunkDispatchPlan> find(const ChunkCacheKey &key) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_entries.find(key);
    if (it == m_entries.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  void store(const ChunkCacheKey &key, const ChunkDispatchPlan &plan) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_entries[key] = plan;
  }

private:
  mutable std::mutex m_mutex;
  std::unordered_map<ChunkCacheKey, ChunkDispatchPlan, ChunkCacheKeyHash>
      m_entries;
};

uint64_t shape_prefix_product(const ov::Shape &shape) {
  if (shape.size() <= 2) {
    return 1;
  }
  uint64_t batch = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch *= std::max<uint64_t>(1, static_cast<uint64_t>(shape[i]));
  }
  return batch;
}

bool supports_candidate(const GfxParallelismCaps &caps, uint32_t h,
                        uint32_t w) {
  const uint64_t total = static_cast<uint64_t>(h) * static_cast<uint64_t>(w);
  return h > 0 && w > 0 && total <= caps.max_total_threads_per_group &&
         h <= caps.max_threads_per_group[1] &&
         w <= caps.max_threads_per_group[0];
}

struct MatMulDims {
  uint64_t batch = 1;
  uint64_t m = 1;
  uint64_t n = 1;
  uint64_t total = 1;
};

MatMulDims extract_matmul_dims(const ov::Shape &output_shape) {
  MatMulDims dims;
  if (output_shape.size() < 2) {
    return dims;
  }
  dims.batch = shape_prefix_product(output_shape);
  dims.m = std::max<uint64_t>(
      1, static_cast<uint64_t>(output_shape[output_shape.size() - 2]));
  dims.n = std::max<uint64_t>(
      1, static_cast<uint64_t>(output_shape[output_shape.size() - 1]));
  dims.total = dims.batch * dims.m * dims.n;
  return dims;
}

std::string make_profile_key_fallback(const GfxParallelismCaps &caps) {
  std::ostringstream os;
  os << "parallelism:" << caps.preferred_simd_width << ':'
     << caps.subgroup_size << ':'
     << caps.max_total_threads_per_group << ':' << caps.max_threads_per_group[0]
     << ':' << caps.max_threads_per_group[1] << ':'
     << caps.max_threads_per_group[2] << ':'
     << (caps.supports_conv_output_channel_blocking ? 1 : 0) << ':'
     << (caps.supports_conv_channel_block_spatial_tiling ? 1 : 0) << ':'
     << (caps.sort_matmul_tiles_by_shape ? 1 : 0) << ':'
     << (caps.enable_skinny_matmul_tiles ? 1 : 0) << ':'
     << (caps.scale_conv_threads_for_large_spatial ? 1 : 0) << ':'
     << (caps.scale_conv_threads_for_dense_reduction ? 1 : 0) << ':'
     << (caps.scale_conv_threads_for_pointwise_reduction ? 1 : 0) << ':'
     << (caps.conv_spatial_micro_tile_requires_large_output_area ? 1 : 0)
     << ':' << (caps.chunk_dispatch.retune_threads_to_workload ? 1 : 0);
  return os.str();
}

MatMulCacheKey make_cache_key(const GfxParallelismCaps &caps,
                              const ov::Shape &output_shape) {
  MatMulCacheKey key;
  key.profile_key =
      caps.profile_key.empty() ? make_profile_key_fallback(caps)
                               : caps.profile_key;
  key.output_shape = output_shape;
  return key;
}

ConvParallelCacheKey make_conv_parallel_cache_key(
    const GfxParallelismCaps &caps, const ov::Shape &output_shape,
    uint64_t input_channels, uint64_t output_channels, uint64_t kernel_work,
    bool stride2, bool depthwise) {
  ConvParallelCacheKey key;
  key.profile_key =
      caps.profile_key.empty() ? make_profile_key_fallback(caps)
                               : caps.profile_key;
  key.supports_conv_output_channel_blocking =
      caps.supports_conv_output_channel_blocking;
  key.supports_conv_channel_block_spatial_tiling =
      caps.supports_conv_channel_block_spatial_tiling;
  key.conv_spatial_micro_tile_requires_large_output_area =
      caps.conv_spatial_micro_tile_requires_large_output_area;
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

uint32_t clamp_threadgroup_candidate(const GfxParallelismCaps &caps,
                                     uint32_t candidate) {
  const uint32_t max_x = std::max<uint32_t>(1u, caps.max_threads_per_group[0]);
  const uint32_t max_total =
      std::max<uint32_t>(1u, caps.max_total_threads_per_group);
  return std::max<uint32_t>(1u,
                            std::min(candidate, std::min(max_x, max_total)));
}

std::vector<uint32_t>
enumerate_hardware_relative_threadgroups(const GfxParallelismCaps &caps) {
  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint32_t max_threads = clamp_threadgroup_candidate(
      caps, std::max<uint32_t>(1u, caps.max_total_threads_per_group));
  std::vector<uint32_t> candidates;
  auto add = [&](uint32_t value) {
    const uint32_t clamped = clamp_threadgroup_candidate(caps, value);
    if (std::find(candidates.begin(), candidates.end(), clamped) ==
        candidates.end()) {
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

  candidates.erase(std::remove(candidates.begin(), candidates.end(), 0),
                   candidates.end());
  std::sort(candidates.begin(), candidates.end(), std::greater<uint32_t>());
  return candidates;
}

uint32_t select_hardware_relative_threadgroup(const GfxParallelismCaps &caps,
                                              uint64_t total_elems,
                                              uint64_t work_per_elem,
                                              uint32_t default_threads) {
  const auto candidates = enumerate_hardware_relative_threadgroups(caps);
  if (candidates.empty()) {
    return std::max<uint32_t>(1u, default_threads);
  }

  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint32_t clamped_default = clamp_threadgroup_candidate(
      caps, std::max<uint32_t>(1u, default_threads));
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
    target = clamp_threadgroup_candidate(
        caps, static_cast<uint32_t>(std::min<uint64_t>(total_bucket, target)));
  }

  auto best = candidates.front();
  uint64_t best_score = std::numeric_limits<uint64_t>::max();
  for (const auto candidate : candidates) {
    const uint64_t distance =
        candidate > target ? candidate - target : target - candidate;
    const uint64_t oversubscribe_penalty =
        candidate > total_bucket ? (candidate - total_bucket) : 0;
    const uint64_t score = distance * 8 + oversubscribe_penalty;
    if (score < best_score) {
      best = candidate;
      best_score = score;
    }
  }
  return std::max<uint32_t>(1u, best);
}

double score_matmul_plan(const GfxParallelismCaps &caps,
                         const ov::Shape &output_shape,
                         const MatMulParallelismPlan &plan,
                         bool prefer_skinny_tiles) {
  const auto dims = extract_matmul_dims(output_shape);
  const uint32_t h = std::max<uint32_t>(1u, plan.dispatch.threads_h);
  const uint32_t w = std::max<uint32_t>(1u, plan.dispatch.threads_w);
  const uint32_t threads = std::max<uint32_t>(1u, h * w);
  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  uint32_t desired_threads = select_hardware_relative_threadgroup(
      caps, dims.total, std::max<uint64_t>(1, dims.m), wave);
  const double output_aspect =
      static_cast<double>(std::max<uint64_t>(dims.m, dims.n)) /
      static_cast<double>(std::max<uint64_t>(1, std::min(dims.m, dims.n)));
  const bool compute_dense =
      std::min<uint64_t>(dims.m, dims.n) >= static_cast<uint64_t>(wave) * 8u;
  if (dims.batch == 1 && compute_dense && output_aspect <= 2.0) {
    desired_threads =
        std::max(desired_threads, clamp_threadgroup_candidate(caps, wave * 4u));
  } else if (dims.batch == 1 && compute_dense && output_aspect <= 4.0) {
    desired_threads =
        std::max(desired_threads, clamp_threadgroup_candidate(caps, wave * 2u));
  }

  const double target_aspect =
      clamp_double(static_cast<double>(dims.n) / static_cast<double>(dims.m),
                   1.0 / 16.0, 16.0);
  const double tile_aspect = clamp_double(
      static_cast<double>(w) / static_cast<double>(h), 1.0 / 32.0, 32.0);
  const double aspect_error =
      std::abs(std::log2(clamp_double(tile_aspect / target_aspect, 1e-3, 1e3)));
  const double thread_error = std::abs(static_cast<double>(threads) -
                                       static_cast<double>(desired_threads));
  const double oversubscribe_h =
      h > dims.m ? static_cast<double>(h - dims.m) : 0.0;
  const double oversubscribe_w =
      w > dims.n ? static_cast<double>(w - dims.n) : 0.0;
  const double underfilled_wave_penalty =
      (threads < wave && dims.total >= static_cast<uint64_t>(wave) * 4u)
          ? static_cast<double>(wave - threads)
          : 0.0;
  const double wide_shape_penalty = (dims.n >= dims.m * 8u && h > 2u)
                                        ? static_cast<double>(h - 2u) * 8.0
                                        : 0.0;
  const double tall_shape_penalty = (dims.m >= dims.n * 8u && w > 2u)
                                        ? static_cast<double>(w - 2u) * 8.0
                                        : 0.0;
  const double skinny_bonus =
      (prefer_skinny_tiles && dims.n >= dims.m * 8u && w >= h)
          ? static_cast<double>(std::min<uint32_t>(w, desired_threads))
          : 0.0;

  return aspect_error * 32.0 + thread_error +
         (oversubscribe_h + oversubscribe_w) * 64.0 +
         underfilled_wave_penalty * 2.0 + wide_shape_penalty +
         tall_shape_penalty - skinny_bonus;
}

void sort_matmul_candidates_by_shape(const GfxParallelismCaps &caps,
                                     const ov::Shape &output_shape,
                                     std::vector<MatMulParallelismPlan> &plans,
                                     bool prefer_skinny_tiles) {
  std::stable_sort(
      plans.begin(), plans.end(),
      [&](const MatMulParallelismPlan &lhs, const MatMulParallelismPlan &rhs) {
        const double lhs_score =
            score_matmul_plan(caps, output_shape, lhs, prefer_skinny_tiles);
        const double rhs_score =
            score_matmul_plan(caps, output_shape, rhs, prefer_skinny_tiles);
        if (lhs_score != rhs_score) {
          return lhs_score < rhs_score;
        }
        const uint32_t lhs_threads = std::max<uint32_t>(
            1u, lhs.dispatch.threads_h * lhs.dispatch.threads_w);
        const uint32_t rhs_threads = std::max<uint32_t>(
            1u, rhs.dispatch.threads_h * rhs.dispatch.threads_w);
        return lhs_threads > rhs_threads;
      });
}

ChunkCacheKey make_chunk_cache_key(const GfxParallelismCaps &caps,
                                   const std::string &op_kind,
                                   uint64_t total_elems,
                                   uint64_t work_per_elem) {
  ChunkCacheKey key;
  key.profile_key =
      caps.profile_key.empty() ? make_profile_key_fallback(caps)
                               : caps.profile_key;
  key.op_kind = op_kind;
  key.total_bucket = bucketize(total_elems);
  key.work_bucket = bucketize(work_per_elem);
  return key;
}

struct ThreadPair {
  uint32_t h = 1;
  uint32_t w = 1;
};

struct SpatialMicroTile {
  uint32_t h = 1;
  uint32_t w = 1;
};

struct ConvWorkloadProfile {
  uint64_t out_h = 1;
  uint64_t out_w = 1;
  uint64_t spatial = 1;
  uint64_t output_area = 1;
  uint32_t wave = 1;
  uint32_t max_threads = 1;
  uint32_t balanced_threads = 1;
  uint32_t dense_threads = 1;
  bool spatially_large = false;
  bool spatially_huge = false;
  bool channel_reuse = false;
  bool dense_reduction = false;
  bool very_dense_reduction = false;
  bool ultra_dense_reduction = false;
  bool pointwise_or_light_reduction = false;
  bool pointwise_dense_reduction = false;
};

ConvWorkloadProfile make_conv_workload_profile(
    const GfxParallelismCaps &caps, const ov::Shape &output_shape,
    uint64_t input_channels, uint64_t output_channels, uint64_t kernel_work,
    bool depthwise) {
  ConvWorkloadProfile profile{};
  profile.out_h =
      output_shape.size() >= 2
          ? std::max<uint64_t>(
                1, static_cast<uint64_t>(output_shape[output_shape.size() - 2]))
          : 1;
  profile.out_w =
      output_shape.size() >= 1
          ? std::max<uint64_t>(
                1, static_cast<uint64_t>(output_shape[output_shape.size() - 1]))
          : 1;
  profile.output_area = profile.out_h * profile.out_w;
  profile.spatial = shape_prefix_product(output_shape) * profile.output_area;
  profile.wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  profile.max_threads = clamp_threadgroup_candidate(
      caps, std::max<uint32_t>(1u, caps.max_total_threads_per_group));
  profile.balanced_threads = clamp_threadgroup_candidate(
      caps, std::max<uint32_t>(profile.wave * 4u, profile.max_threads / 4u));
  profile.dense_threads = clamp_threadgroup_candidate(
      caps, std::max<uint32_t>(profile.wave * 8u, profile.max_threads / 2u));
  const uint64_t wave = std::max<uint64_t>(1, profile.wave);
  profile.spatially_large = profile.spatial >= wave * wave;
  profile.spatially_huge = profile.spatial >= wave * wave * wave;
  profile.channel_reuse = !depthwise && input_channels >= wave * 2u &&
                          output_channels >= wave * 2u;
  profile.dense_reduction =
      profile.channel_reuse && kernel_work >= wave * wave;
  profile.very_dense_reduction =
      profile.channel_reuse && kernel_work >= wave * wave * 2u &&
      input_channels >= wave * 4u && output_channels >= wave * 4u;
  profile.ultra_dense_reduction =
      profile.channel_reuse && kernel_work >= wave * wave * 4u &&
      input_channels >= wave * 8u && output_channels >= wave * 8u;
  profile.pointwise_or_light_reduction =
      !depthwise && profile.spatially_huge && kernel_work <= wave * 4u &&
      output_channels >= wave * 2u;
  profile.pointwise_dense_reduction =
      !depthwise && profile.spatially_huge && kernel_work >= wave * 8u &&
      output_channels >= wave * 4u;
  return profile;
}

ThreadPair choose_conv_thread_pair(const GfxParallelismCaps &caps,
                                   uint32_t target_threads, uint64_t out_h,
                                   uint64_t out_w, uint64_t spatial,
                                   uint64_t kernel_work, bool stride2,
                                   bool depthwise) {
  const uint32_t max_h = std::max<uint32_t>(
      1u, std::min<uint32_t>(caps.max_threads_per_group[1], target_threads));
  const uint32_t max_w = std::max<uint32_t>(
      1u, std::min<uint32_t>(caps.max_threads_per_group[0], target_threads));
  const double spatial_aspect =
      static_cast<double>(std::max<uint64_t>(1, out_w)) /
      static_cast<double>(std::max<uint64_t>(1, out_h));
  double target_aspect = spatial_aspect;
  if (stride2) {
    target_aspect *= 1.5;
  }
  if (depthwise) {
    target_aspect *= 0.5;
  }
  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  if (kernel_work >= static_cast<uint64_t>(wave) * wave * 4u) {
    target_aspect = (target_aspect + 1.0) * 0.5;
  }
  target_aspect = clamp_double(target_aspect, 0.5, 4.0);

  ThreadPair best{};
  double best_score = std::numeric_limits<double>::infinity();
  for (uint32_t h = 1; h <= max_h; ++h) {
    for (uint32_t w = 1; w <= max_w; ++w) {
      const uint32_t threads = h * w;
      if (threads == 0 || threads > target_threads ||
          threads > caps.max_total_threads_per_group) {
        continue;
      }
      const double aspect = static_cast<double>(w) / static_cast<double>(h);
      const double aspect_error =
          std::abs(std::log2(clamp_double(aspect / target_aspect, 1e-3, 1e3)));
      const double util_penalty = static_cast<double>(target_threads - threads);
      const double oversubscribe_penalty =
          threads > spatial ? static_cast<double>(threads - spatial) : 0.0;
      const double line_penalty =
          ((h == 1 || w == 1) && threads > 1 && out_h > 1 && out_w > 1) ? 6.0
                                                                        : 0.0;
      const double score = util_penalty * 16.0 + aspect_error * 10.0 +
                           oversubscribe_penalty * 2.0 + line_penalty;
      if (score < best_score) {
        best = ThreadPair{h, w};
        best_score = score;
      }
    }
  }
  return best;
}

uint32_t select_conv_accumulator_budget(const GfxParallelismCaps &caps);

uint64_t min_output_area_for_conv_spatial_micro_tile(
    const GfxParallelismCaps &caps, uint32_t wave) {
  if (!caps.conv_spatial_micro_tile_requires_large_output_area) {
    return 1;
  }
  return static_cast<uint64_t>(std::max<uint32_t>(1u, wave)) *
         static_cast<uint64_t>(
             std::max<uint32_t>(1u, caps.max_total_threads_per_group));
}

bool can_use_conv_spatial_micro_tile_for_area(const GfxParallelismCaps &caps,
                                              uint64_t output_area,
                                              uint32_t wave) {
  return output_area >= min_output_area_for_conv_spatial_micro_tile(caps, wave);
}

uint32_t select_conv_output_channel_block(
    const GfxParallelismCaps &caps, uint64_t spatial, uint64_t input_channels,
    uint64_t output_channels, uint64_t kernel_work, uint64_t output_area,
    bool stride2, bool depthwise) {
  if (!caps.supports_conv_output_channel_blocking || depthwise ||
      spatial == 0 || input_channels == 0 || output_channels < 2) {
    return 1;
  }

  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const bool has_reuse = spatial >= static_cast<uint64_t>(wave) &&
                         kernel_work >= static_cast<uint64_t>(wave) &&
                         input_channels >= static_cast<uint64_t>(wave);
  if (!has_reuse) {
    return 1;
  }

  // Channel blocking is a shared dense-reduction policy. The lowering can
  // choose fused or serial accumulation separately, so the block width follows
  // the hardware-relative wave size and accumulator budget rather than a
  // per-device correctness cap.
  const uint32_t accumulator_budget = select_conv_accumulator_budget(caps);
  uint32_t base_block =
      std::max<uint32_t>(1u, std::min<uint32_t>(wave / 4u, accumulator_budget));
  const uint32_t threadgroup_waves = std::max<uint32_t>(
      1u, std::max<uint32_t>(1u, caps.max_total_threads_per_group) / wave);
  const bool can_spatial_micro_tile =
      caps.supports_conv_channel_block_spatial_tiling && !stride2 &&
      can_use_conv_spatial_micro_tile_for_area(caps, output_area, wave);
  const bool pointwise_reduction = kernel_work <= input_channels;
  const uint64_t dense_spatial_output_area =
      static_cast<uint64_t>(wave) * static_cast<uint64_t>(wave) * 32u;
  const bool in_dense_spatial_reuse_window =
      !stride2 && (pointwise_reduction || output_area <= dense_spatial_output_area);
  const bool compact_wave_dense_reuse =
      !can_spatial_micro_tile &&
      in_dense_spatial_reuse_window &&
      threadgroup_waves >= wave &&
      wave <= accumulator_budget * 2u &&
      input_channels >= static_cast<uint64_t>(wave) * 2u &&
      output_channels >= static_cast<uint64_t>(wave) * 2u &&
      kernel_work >= static_cast<uint64_t>(wave) * 8u;
  if (compact_wave_dense_reuse) {
    base_block =
        std::max<uint32_t>(base_block,
                           std::min<uint32_t>(accumulator_budget, wave / 2u));
  }
  const uint64_t compact_output_area =
      static_cast<uint64_t>(wave) * static_cast<uint64_t>(wave) * 8u;
  const bool dense_downsample_reduction =
      stride2 && kernel_work >= static_cast<uint64_t>(wave) * wave &&
      input_channels >= static_cast<uint64_t>(wave) * 2u &&
      output_channels >= static_cast<uint64_t>(wave) * 2u &&
      output_area <= compact_output_area;
  const uint32_t max_block =
      dense_downsample_reduction
          ? std::max<uint32_t>(
                base_block,
                std::min<uint32_t>(accumulator_budget,
                                   std::max<uint32_t>(base_block, wave / 2u)))
          : base_block;
  uint32_t block = max_block;
  while (block > 1u && output_channels % block != 0u) {
    block >>= 1u;
  }
  return std::max<uint32_t>(block, 1u);
}

ConvChannelBlockAccumulation
select_conv_channel_block_accumulation(const GfxParallelismCaps &caps,
                                       uint64_t kernel_work,
                                       uint32_t output_channel_block,
                                       const SpatialMicroTile &micro_tile) {
  if (output_channel_block <= 1u) {
    return ConvChannelBlockAccumulation::Fused;
  }
  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint32_t fused_accumulators =
      output_channel_block * std::max<uint32_t>(micro_tile.h, 1u) *
      std::max<uint32_t>(micro_tile.w, 1u);
  if (fused_accumulators <= select_conv_accumulator_budget(caps)) {
    return ConvChannelBlockAccumulation::Fused;
  }
  const bool constrained_device = caps.max_total_threads_per_group <= 256u;
  const uint64_t dense_reduction_threshold =
      static_cast<uint64_t>(wave) * fused_accumulators * 8u;
  return constrained_device && kernel_work >= dense_reduction_threshold
             ? ConvChannelBlockAccumulation::Serial
             : ConvChannelBlockAccumulation::Fused;
}

uint32_t select_conv_accumulator_budget(const GfxParallelismCaps &caps) {
  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint32_t threadgroup_waves = std::max<uint32_t>(
      1u, std::max<uint32_t>(1u, caps.max_total_threads_per_group) / wave);
  return std::max<uint32_t>(
      1u, std::min<uint32_t>(wave / 2u, threadgroup_waves * 2u));
}

SpatialMicroTile select_conv_spatial_micro_tile(
    const GfxParallelismCaps &caps, const ThreadPair &threads, uint64_t out_h,
    uint64_t out_w, uint64_t spatial, uint64_t kernel_work, bool stride2,
    bool depthwise, uint32_t output_channel_block) {
  if (depthwise || output_channel_block <= 1u ||
      !caps.supports_conv_channel_block_spatial_tiling ||
      spatial <= static_cast<uint64_t>(threads.h) * threads.w) {
    return {};
  }

  const uint32_t channel_block = std::max<uint32_t>(1u, output_channel_block);
  const uint32_t accumulator_budget = select_conv_accumulator_budget(caps);
  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint64_t output_area = out_h * out_w;
  if (!can_use_conv_spatial_micro_tile_for_area(caps, output_area, wave)) {
    return {};
  }
  const bool dense_reduction =
      kernel_work >= static_cast<uint64_t>(accumulator_budget) * channel_block;
  if (!dense_reduction) {
    return {};
  }
  const uint32_t max_spatial_lanes =
      std::max<uint32_t>(1u, accumulator_budget / channel_block);
  if (max_spatial_lanes <= 1u) {
    return {};
  }

  const double output_aspect =
      static_cast<double>(std::max<uint64_t>(1, out_w)) /
      static_cast<double>(std::max<uint64_t>(1, out_h));
  double target_micro_aspect = clamp_double(output_aspect, 0.5, 4.0);
  if (stride2) {
    target_micro_aspect = clamp_double(target_micro_aspect * 1.5, 0.5, 4.0);
  }

  SpatialMicroTile best{};
  double best_score = std::numeric_limits<double>::infinity();
  const uint32_t max_micro_h = std::max<uint32_t>(
      1u,
      std::min<uint32_t>(max_spatial_lanes,
                         static_cast<uint32_t>(std::min<uint64_t>(4, out_h))));
  const uint32_t max_micro_w = std::max<uint32_t>(
      1u,
      std::min<uint32_t>(max_spatial_lanes,
                         static_cast<uint32_t>(std::min<uint64_t>(4, out_w))));
  for (uint32_t micro_h = 1; micro_h <= max_micro_h; ++micro_h) {
    for (uint32_t micro_w = 1; micro_w <= max_micro_w; ++micro_w) {
      const uint32_t lanes = micro_h * micro_w;
      if (lanes == 0 || lanes > max_spatial_lanes) {
        continue;
      }
      const uint64_t tile_h = static_cast<uint64_t>(threads.h) * micro_h;
      const uint64_t tile_w = static_cast<uint64_t>(threads.w) * micro_w;
      if (tile_h > out_h * 2u || tile_w > out_w * 2u) {
        continue;
      }
      const double micro_aspect =
          static_cast<double>(micro_w) / static_cast<double>(micro_h);
      const double aspect_error = std::abs(std::log2(
          clamp_double(micro_aspect / target_micro_aspect, 1e-3, 1e3)));
      const double lane_bonus = static_cast<double>(lanes - 1u) * 8.0;
      const double edge_penalty =
          static_cast<double>((tile_h - (out_h % tile_h)) % tile_h +
                              (tile_w - (out_w % tile_w)) % tile_w) /
          static_cast<double>(std::max<uint64_t>(1, tile_h + tile_w));
      const double score = aspect_error * 4.0 + edge_penalty * 2.0 - lane_bonus;
      if (score < best_score) {
        best = SpatialMicroTile{micro_h, micro_w};
        best_score = score;
      }
    }
  }
  return best;
}

ConvParallelismPlan make_conv_parallelism_plan(
    const GfxParallelismCaps &caps, const ov::Shape &output_shape,
    uint64_t input_channels, uint64_t output_channels, uint64_t kernel_work,
    bool stride2, bool depthwise, uint32_t default_threads) {
  ConvParallelismPlan plan;
  if (output_shape.size() < 4) {
    return plan;
  }

  const auto profile =
      make_conv_workload_profile(caps, output_shape, input_channels,
                                 output_channels, kernel_work, depthwise);
  const uint64_t effective_work =
      std::max<uint64_t>(kernel_work, std::max<uint64_t>(1, input_channels));
  const uint32_t clamped_default = clamp_threadgroup_candidate(
      caps, std::max<uint32_t>(1u, default_threads));
  uint32_t target_threads = select_hardware_relative_threadgroup(
      caps, profile.spatial, effective_work, clamped_default);
  const bool dense_parallel_workload =
      !depthwise &&
      profile.spatial >= static_cast<uint64_t>(clamped_default) * 4u &&
      effective_work >= clamped_default;
  if (dense_parallel_workload) {
    target_threads = std::max(target_threads, clamped_default);
  }
  const auto pair =
      choose_conv_thread_pair(caps, target_threads, profile.out_h,
                              profile.out_w, profile.spatial, kernel_work,
                              stride2, depthwise);
  const uint32_t output_channel_block = select_conv_output_channel_block(
      caps, profile.spatial, input_channels, output_channels, kernel_work,
      profile.output_area, stride2, depthwise);
  const auto micro_tile = select_conv_spatial_micro_tile(
      caps, pair, profile.out_h, profile.out_w, profile.spatial, kernel_work,
      stride2, depthwise, output_channel_block);
  const auto channel_block_accumulation =
      select_conv_channel_block_accumulation(caps, kernel_work,
                                             output_channel_block, micro_tile);

  plan.prefer_parallel =
      (shape_prefix_product(output_shape) * output_channels *
       profile.output_area) >= 256;
  plan.variant =
      "conv_parallel_" + std::to_string(pair.h) + "x" + std::to_string(pair.w);
  plan.dispatch.enabled = true;
  plan.dispatch.loop_dims = 5;
  plan.dispatch.tile_h = pair.h * micro_tile.h;
  plan.dispatch.tile_w = pair.w * micro_tile.w;
  plan.dispatch.threads_h = pair.h;
  plan.dispatch.threads_w = pair.w;
  plan.output_channel_block = output_channel_block;
  plan.channel_block_accumulation = channel_block_accumulation;
  plan.dispatch.channel_block = plan.output_channel_block;
  if (micro_tile.h > 1u || micro_tile.w > 1u) {
    plan.variant += "_tile" + std::to_string(plan.dispatch.tile_h) + "x" +
                    std::to_string(plan.dispatch.tile_w);
  }
  if (plan.output_channel_block > 1u) {
    plan.variant += "_oc" + std::to_string(plan.output_channel_block);
    if (plan.channel_block_accumulation ==
        ConvChannelBlockAccumulation::Serial) {
      plan.variant += "_serial_acc";
    }
  }
  return plan;
}

GfxParallelismCaps
make_caps_from_device_info(const GpuExecutionDeviceInfo &info) {
  GfxParallelismCaps caps = info.parallelism_profile;
  if (caps.profile_key.empty()) {
    caps.profile_key = info.device_key;
  }
  caps.preferred_simd_width = std::max<uint32_t>(info.preferred_simd_width, 1u);
  caps.subgroup_size = std::max<uint32_t>(info.subgroup_size, 1u);
  caps.max_total_threads_per_group =
      std::max<uint32_t>(info.max_total_threads_per_group, 1u);
  caps.max_threads_per_group = {
      std::max<uint32_t>(info.max_threads_per_group[0], 1u),
      std::max<uint32_t>(info.max_threads_per_group[1], 1u),
      std::max<uint32_t>(info.max_threads_per_group[2], 1u)};
  caps.supports_conv_output_channel_blocking =
      info.supports_conv_output_channel_blocking;
  caps.supports_conv_channel_block_spatial_tiling =
      info.supports_conv_channel_block_spatial_tiling;
  return caps;
}

GfxParallelismCaps make_default_caps() {
  GfxParallelismCaps caps{};
  caps.profile_key = "parallelism:default";
  caps.preferred_simd_width = 32;
  caps.subgroup_size = 32;
  caps.max_total_threads_per_group = 128;
  caps.max_threads_per_group = {128, 128, 64};
  caps.chunk_dispatch = make_opencl_chunk_dispatch_profile();
  return caps;
}

void append_matmul_candidate(const GfxParallelismCaps &caps, uint32_t h,
                             uint32_t w, uint32_t wave,
                             std::vector<MatMulParallelismPlan> &plans) {
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

std::vector<MatMulParallelismPlan>
enumerate_matmul_parallelism_candidates_generic(
    const GfxParallelismCaps &caps, const ov::Shape &output_shape,
    bool include_skinny_candidates) {
  std::vector<MatMulParallelismPlan> plans;
  const auto dims = extract_matmul_dims(output_shape);
  if (output_shape.size() < 2) {
    return plans;
  }
  if (dims.total < 1024) {
    return plans;
  }

  const uint32_t wave = std::max<uint32_t>(
      std::max(caps.subgroup_size, caps.preferred_simd_width), 1u);
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

  for (const auto &candidate : base_candidates) {
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
    for (const auto &candidate : skinny_candidates) {
      append_matmul_candidate(caps, candidate[0], candidate[1], wave, plans);
    }
  }
  return plans;
}

ChunkDispatchPlan select_chunk_dispatch_plan_generic(
    const GfxParallelismCaps &caps, const std::string &op_kind,
    uint64_t total_elems, uint64_t work_per_elem) {
  ChunkDispatchPlan plan;
  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  plan.threads_per_group = std::min<uint32_t>(
      std::max<uint32_t>(wave, 64u),
      std::max<uint32_t>(1u, caps.max_total_threads_per_group));

  const auto &chunk_profile = caps.chunk_dispatch;
  const GpuChunkDispatchBand *band = &chunk_profile.light;
  if (work_per_elem >= chunk_profile.very_heavy.min_work_per_elem) {
    band = &chunk_profile.very_heavy;
  } else if (work_per_elem >= chunk_profile.heavy.min_work_per_elem) {
    band = &chunk_profile.heavy;
  } else if (work_per_elem >= chunk_profile.medium.min_work_per_elem) {
    band = &chunk_profile.medium;
  }

  uint32_t elems = std::max<uint32_t>(1u, band->elems_per_dispatch);
  uint32_t max_elems = std::max<uint32_t>(
      elems, std::max<uint32_t>(1u, band->max_elems_per_dispatch));
  uint32_t target_dispatches =
      std::max<uint32_t>(1u, band->target_dispatches);

  if (total_elems <= chunk_profile.small_total_elems_threshold) {
    elems = static_cast<uint32_t>(std::max<uint64_t>(
        std::max<uint32_t>(1u, chunk_profile.small_min_elems_per_dispatch),
        bucketize(total_elems)));
  } else {
    const uint64_t min_elems_for_budget =
        bucketize(ceil_div_u64(total_elems, target_dispatches));
    elems = std::max<uint32_t>(elems, static_cast<uint32_t>(std::min<uint64_t>(
                                          max_elems, min_elems_for_budget)));
  }
  elems = std::min<uint32_t>(elems, max_elems);

  plan.elems_per_dispatch = elems;
  if (chunk_profile.retune_threads_to_workload) {
    plan.threads_per_group = select_hardware_relative_threadgroup(
        caps, total_elems, work_per_elem, plan.threads_per_group);
  }
  plan.variant = op_kind + "_chunk_" + std::to_string(plan.elems_per_dispatch);
  return plan;
}

uint32_t select_conv_default_threads(const GfxParallelismCaps &caps,
                                     const ConvWorkloadProfile &profile,
                                     uint64_t output_channels, bool stride2,
                                     bool depthwise) {
  uint32_t default_threads = profile.wave;
  if (caps.scale_conv_threads_for_large_spatial) {
    if (!depthwise && profile.spatially_huge) {
      const uint32_t huge_spatial_threads =
          output_channels >= static_cast<uint64_t>(profile.wave) * 2u
              ? profile.balanced_threads
              : profile.wave * 2u;
      default_threads = clamp_threadgroup_candidate(caps, huge_spatial_threads);
    } else if (!depthwise && profile.spatially_large &&
               output_channels >= static_cast<uint64_t>(profile.wave) * 2u) {
      default_threads = clamp_threadgroup_candidate(caps, profile.wave * 2u);
    }
  }
  if (caps.scale_conv_threads_for_dense_reduction && !depthwise &&
      profile.dense_reduction &&
      profile.spatial >=
          static_cast<uint64_t>(profile.wave) * profile.wave * 4u) {
    default_threads =
        clamp_threadgroup_candidate(caps, stride2 ? profile.wave * 4u
                                                  : profile.wave * 2u);
    if (!stride2 && profile.very_dense_reduction && profile.spatially_large) {
      default_threads = profile.dense_threads;
    } else if (stride2 && profile.very_dense_reduction &&
               profile.spatial >=
                   static_cast<uint64_t>(profile.wave) * profile.wave * 8u) {
      default_threads = profile.dense_threads;
    }
    if (profile.ultra_dense_reduction && profile.spatially_huge) {
      default_threads = profile.dense_threads;
    }
  }
  if (caps.scale_conv_threads_for_pointwise_reduction) {
    if (profile.pointwise_or_light_reduction) {
      default_threads = profile.balanced_threads;
    } else if (profile.pointwise_dense_reduction) {
      default_threads = profile.dense_threads;
    }
  }
  return default_threads;
}

} // namespace

GfxParallelismCaps
query_parallelism_caps(const GpuBufferManager *buffer_manager) {
  if (buffer_manager) {
    if (const auto info = buffer_manager->query_execution_device_info()) {
      return make_caps_from_device_info(*info);
    }
  }
  return make_default_caps();
}

std::vector<MatMulParallelismPlan>
enumerate_matmul_parallelism_candidates(const GfxParallelismCaps &caps,
                                        const ov::Shape &output_shape) {
  const auto dims = extract_matmul_dims(output_shape);
  const bool skinny_output = dims.batch == 1 && dims.n >= dims.m * 8u;
  auto plans = enumerate_matmul_parallelism_candidates_generic(
      caps, output_shape,
      caps.enable_skinny_matmul_tiles && skinny_output);
  if (caps.sort_matmul_tiles_by_shape) {
    sort_matmul_candidates_by_shape(
        caps, output_shape,
        plans, caps.enable_skinny_matmul_tiles && skinny_output);
  }
  return plans;
}

MatMulParallelismPlan select_matmul_parallelism(const GfxParallelismCaps &caps,
                                                const ov::Shape &output_shape) {
  const auto key = make_cache_key(caps, output_shape);
  if (auto cached = MatMulTuningCache::instance().find(key)) {
    return *cached;
  }

  const auto plans =
      enumerate_matmul_parallelism_candidates(caps, output_shape);
  if (plans.empty()) {
    return {};
  }
  const auto &chosen = plans.front();
  MatMulTuningCache::instance().store(key, chosen);
  return chosen;
}

void remember_matmul_parallelism(const GfxParallelismCaps &caps,
                                 const ov::Shape &output_shape,
                                 const MatMulParallelismPlan &plan) {
  MatMulTuningCache::instance().store(make_cache_key(caps, output_shape), plan);
}

ConvParallelismPlan select_conv_parallelism(const GfxParallelismCaps &caps,
                                            const ov::Shape &output_shape,
                                            uint64_t input_channels,
                                            uint64_t output_channels,
                                            uint64_t kernel_work, bool stride2,
                                            bool depthwise) {
  const auto key = make_conv_parallel_cache_key(
      caps, output_shape, input_channels, output_channels, kernel_work, stride2,
      depthwise);
  if (auto cached = ConvParallelTuningCache::instance().find(key)) {
    return *cached;
  }

  const auto profile =
      make_conv_workload_profile(caps, output_shape, input_channels,
                                 output_channels, kernel_work, depthwise);
  const uint32_t default_threads =
      select_conv_default_threads(caps, profile, output_channels, stride2,
                                  depthwise);
  ConvParallelismPlan plan =
      make_conv_parallelism_plan(caps, output_shape, input_channels,
                                 output_channels, kernel_work, stride2,
                                 depthwise, default_threads);
  ConvParallelTuningCache::instance().store(key, plan);
  return plan;
}

ChunkDispatchPlan select_chunk_dispatch_plan(const GfxParallelismCaps &caps,
                                             const std::string &op_kind,
                                             uint64_t total_elems,
                                             uint64_t work_per_elem) {
  const auto key =
      make_chunk_cache_key(caps, op_kind, total_elems, work_per_elem);
  if (auto cached = ChunkTuningCache::instance().find(key)) {
    return *cached;
  }

  ChunkDispatchPlan plan =
      select_chunk_dispatch_plan_generic(caps, op_kind, total_elems,
                                         work_per_elem);
  ChunkTuningCache::instance().store(key, plan);
  return plan;
}

} // namespace gfx_plugin
} // namespace ov
