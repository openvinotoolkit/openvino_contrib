// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "common/gpu_backend.hpp"
#include "common/gpu_device_profile.hpp"
#include "common/gpu_parallelism_profile.hpp"

namespace ov {
namespace gfx_plugin {

struct GpuExecutionDeviceInfo {
  GpuBackend backend = GpuBackend::Unknown;
  GpuDeviceFamily device_family = GpuDeviceFamily::Generic;
  std::string device_key;
  std::string device_name;
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  uint32_t driver_version = 0;
  uint32_t api_version = 0;
  uint32_t preferred_simd_width = 1;
  uint32_t subgroup_size = 1;
  uint32_t max_total_threads_per_group = 1;
  std::array<uint32_t, 3> max_threads_per_group{{1, 1, 1}};
  uint64_t min_storage_buffer_offset_alignment = 1;
  uint64_t non_coherent_atom_size = 1;
  bool supports_storage_buffer_8bit = false;
  bool supports_storage_buffer_16bit = false;
  bool supports_shader_float16 = false;
  bool supports_shader_int8 = false;
  bool supports_conv_output_channel_blocking = false;
  bool supports_conv_channel_block_spatial_tiling = false;
  GpuParallelismProfile parallelism_profile;
};

} // namespace gfx_plugin
} // namespace ov
