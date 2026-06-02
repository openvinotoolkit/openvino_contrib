// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <string>
#include <vector>

#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "runtime/gpu_device_info.hpp"

namespace ov {
namespace gfx_plugin {

class GpuBufferManager;

struct GfxParallelismCaps {
  GpuBackend backend = GpuBackend::Unknown;
  GpuDeviceFamily device_family = GpuDeviceFamily::Generic;
  std::string device_key;
  uint32_t preferred_simd_width = 1;
  uint32_t subgroup_size = 1;
  uint32_t max_total_threads_per_group = 1;
  std::array<uint32_t, 3> max_threads_per_group{{1, 1, 1}};
  bool supports_conv_output_channel_blocking = false;
  bool supports_conv_channel_block_spatial_tiling = false;
};

struct MatMulParallelismPlan {
  bool prefer_parallel = false;
  std::string variant;
  ParallelDispatchConfig dispatch{};
};

enum class ConvChannelBlockAccumulation {
  Fused,
  Serial,
};

const char *conv_channel_block_accumulation_name(
    ConvChannelBlockAccumulation mode);

struct ConvParallelismPlan {
  bool prefer_parallel = false;
  std::string variant;
  ParallelDispatchConfig dispatch{};
  uint32_t output_channel_block = 1;
  ConvChannelBlockAccumulation channel_block_accumulation =
      ConvChannelBlockAccumulation::Fused;
};

struct ChunkDispatchPlan {
  std::string variant;
  uint32_t elems_per_dispatch = 0;
  uint32_t threads_per_group = 64;
};

GfxParallelismCaps
query_parallelism_caps(const GpuBufferManager *buffer_manager);
std::vector<MatMulParallelismPlan>
enumerate_matmul_parallelism_candidates(const GfxParallelismCaps &caps,
                                        const ov::Shape &output_shape);
MatMulParallelismPlan select_matmul_parallelism(const GfxParallelismCaps &caps,
                                                const ov::Shape &output_shape);
void remember_matmul_parallelism(const GfxParallelismCaps &caps,
                                 const ov::Shape &output_shape,
                                 const MatMulParallelismPlan &plan);
ConvParallelismPlan select_conv_parallelism(const GfxParallelismCaps &caps,
                                            const ov::Shape &output_shape,
                                            uint64_t input_channels,
                                            uint64_t output_channels,
                                            uint64_t kernel_work, bool stride2,
                                            bool depthwise);
ChunkDispatchPlan select_chunk_dispatch_plan(const GfxParallelismCaps &caps,
                                             const std::string &op_kind,
                                             uint64_t total_elems,
                                             uint64_t work_per_elem);

} // namespace gfx_plugin
} // namespace ov
