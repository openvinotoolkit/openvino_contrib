// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string>

#include "common/gpu_dispatch_config.hpp"
#include "common/gpu_parallelism_profile.hpp"

namespace ov {
namespace gfx_plugin {

using GfxParallelismCaps = GpuParallelismProfile;

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

}  // namespace gfx_plugin
}  // namespace ov
