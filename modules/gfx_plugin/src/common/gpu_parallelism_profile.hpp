// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace ov {
namespace gfx_plugin {

struct GpuChunkDispatchBand {
  uint64_t min_work_per_elem = 0;
  uint32_t elems_per_dispatch = 32768;
  uint32_t max_elems_per_dispatch = 65536;
  uint32_t target_dispatches = 16;
};

struct GpuChunkDispatchProfile {
  uint64_t small_total_elems_threshold = 16384;
  uint32_t small_min_elems_per_dispatch = 1024;
  GpuChunkDispatchBand light{};
  GpuChunkDispatchBand medium{};
  GpuChunkDispatchBand heavy{};
  GpuChunkDispatchBand very_heavy{};
  bool retune_threads_to_workload = false;
};

inline GpuChunkDispatchProfile make_opencl_chunk_dispatch_profile() {
  GpuChunkDispatchProfile profile{};
  profile.light = GpuChunkDispatchBand{0, 32768, 65536, 16};
  profile.medium = GpuChunkDispatchBand{64, 16384, 32768, 12};
  profile.heavy = GpuChunkDispatchBand{256, 8192, 32768, 8};
  profile.very_heavy = GpuChunkDispatchBand{1024, 4096, 16384, 8};
  return profile;
}

inline GpuChunkDispatchProfile make_metal_chunk_dispatch_profile() {
  GpuChunkDispatchProfile profile{};
  profile.light = GpuChunkDispatchBand{0, 8192, 16384, 12};
  profile.medium = GpuChunkDispatchBand{64, 8192, 16384, 12};
  profile.heavy = GpuChunkDispatchBand{256, 4096, 16384, 8};
  profile.very_heavy = GpuChunkDispatchBand{1024, 4096, 16384, 8};
  return profile;
}

struct GpuParallelismProfile {
  std::string profile_key;
  uint32_t preferred_simd_width = 1;
  uint32_t subgroup_size = 1;
  uint32_t max_total_threads_per_group = 1;
  std::array<uint32_t, 3> max_threads_per_group{{1, 1, 1}};
  bool supports_conv_output_channel_blocking = false;
  bool supports_conv_channel_block_spatial_tiling = false;
  bool sort_matmul_tiles_by_shape = true;
  bool enable_skinny_matmul_tiles = false;
  bool scale_conv_threads_for_large_spatial = false;
  bool scale_conv_threads_for_dense_reduction = false;
  bool scale_conv_threads_for_pointwise_reduction = false;
  bool conv_spatial_micro_tile_requires_large_output_area = false;
  GpuChunkDispatchProfile chunk_dispatch = make_opencl_chunk_dispatch_profile();
};

inline GpuParallelismProfile
make_opencl_parallelism_profile(std::string profile_key = "opencl:default") {
  GpuParallelismProfile profile{};
  profile.profile_key = profile_key;
  profile.preferred_simd_width = 32;
  profile.subgroup_size = 32;
  profile.max_total_threads_per_group = 128;
  profile.max_threads_per_group = {128, 128, 64};
  profile.chunk_dispatch = make_opencl_chunk_dispatch_profile();
  return profile;
}

inline GpuParallelismProfile
make_metal_parallelism_profile(std::string profile_key = "metal:default") {
  GpuParallelismProfile profile{};
  profile.profile_key = profile_key;
  profile.preferred_simd_width = 32;
  profile.subgroup_size = 32;
  profile.max_total_threads_per_group = 256;
  profile.max_threads_per_group = {256, 256, 64};
  profile.supports_conv_output_channel_blocking = true;
  profile.supports_conv_channel_block_spatial_tiling = true;
  profile.sort_matmul_tiles_by_shape = false;
  profile.chunk_dispatch = make_metal_chunk_dispatch_profile();
  return profile;
}

}  // namespace gfx_plugin
}  // namespace ov
