// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>

#include "common/gpu_backend.hpp"
#include "common/gpu_device_profile.hpp"
#include "compiler/backend_target.hpp"
#include "compiler/operation_support.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct StageParallelismProfile {
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

struct StageSourceKernelDispatchPolicy {
  bool enabled = false;
  StageParallelismProfile fallback_parallelism{};
};

struct StageCompilerPolicy {
  GpuBackend backend = GpuBackend::Unknown;
  const StagePlacementPolicy *placement = nullptr;
  const PostOpFusionCapabilities *post_ops = nullptr;
  StageSourceKernelDispatchPolicy source_kernel_dispatch{};
};

struct PrecisionSensitiveFusionQuery {
  std::string group_kind;
  std::string stage_type;
  std::shared_ptr<const ov::Node> primary_node;
  ov::element::Type element_type = ov::element::dynamic;
  bool has_bias = false;
  std::optional<ActivationKind> activation;
  bool has_input_activation = false;
  bool has_batchnorm = false;
  GfxStageRuntimeTraits traits{};
};

StageCompilerPolicy make_stage_compiler_policy_from_capabilities(
    const BackendCapabilities &capabilities);

StageSourceKernelDispatchPolicy
make_stage_source_kernel_dispatch_policy(const BackendTarget &target);

StageCompilerPolicy resolve_stage_compiler_policy(GpuBackend backend);

bool allow_precision_sensitive_arithmetic_fusion(
    const StageCompilerPolicy &policy,
    const PrecisionSensitiveFusionQuery &query);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
