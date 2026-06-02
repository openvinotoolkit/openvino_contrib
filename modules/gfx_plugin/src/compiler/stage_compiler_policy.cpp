// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/stage_compiler_policy.hpp"

#include "compiler/backend_registry.hpp"

#include <string_view>

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_conv_family_stage(std::string_view stage_type) {
  return stage_type == "Convolution" || stage_type == "GroupConvolution";
}

bool is_vendor_image_placement(const GfxStagePlacementPlan &placement) {
  return placement.storage == GfxStageStorageKind::Image &&
         placement.uses_vendor_primitive && !placement.uses_custom_kernel;
}

StageParallelismProfile
make_stage_parallelism_profile(const BackendTarget &target) {
  StageParallelismProfile profile{};
  profile.backend = target.backend();
  profile.device_key = target.backend_id() + ":default";
  switch (target.backend()) {
  case GpuBackend::OpenCL:
    profile.device_family = GpuDeviceFamily::Generic;
    profile.preferred_simd_width = 32;
    profile.subgroup_size = 32;
    profile.max_total_threads_per_group = 128;
    profile.max_threads_per_group = {128, 128, 64};
    break;
  case GpuBackend::Metal:
    profile.device_family = GpuDeviceFamily::Apple;
    profile.preferred_simd_width = 32;
    profile.subgroup_size = 32;
    profile.max_total_threads_per_group = 256;
    profile.max_threads_per_group = {256, 256, 64};
    profile.supports_conv_output_channel_blocking = true;
    profile.supports_conv_channel_block_spatial_tiling = true;
    break;
  case GpuBackend::Unknown:
  default:
    break;
  }
  return profile;
}

} // namespace

StageSourceKernelDispatchPolicy
make_stage_source_kernel_dispatch_policy(const BackendTarget &target) {
  StageSourceKernelDispatchPolicy policy{};
  policy.enabled = target.backend() == GpuBackend::OpenCL;
  policy.fallback_parallelism = make_stage_parallelism_profile(target);
  return policy;
}

StageCompilerPolicy make_stage_compiler_policy_from_capabilities(
    const BackendCapabilities &capabilities) {
  StageCompilerPolicy policy{};
  policy.backend = capabilities.backend();
  policy.placement = capabilities.stage_placement();
  policy.post_ops = &capabilities.post_ops();
  policy.source_kernel_dispatch =
      make_stage_source_kernel_dispatch_policy(capabilities.target());
  return policy;
}

StageCompilerPolicy resolve_stage_compiler_policy(GpuBackend backend) {
  const auto backend_module =
      BackendRegistry::default_registry().resolve(backend);
  if (!backend_module) {
    return {};
  }
  return make_stage_compiler_policy_from_capabilities(
      backend_module->capabilities());
}

bool allow_precision_sensitive_arithmetic_fusion(
    const StageCompilerPolicy &policy,
    const PrecisionSensitiveFusionQuery &query) {
  if (!policy.placement || !policy.post_ops || !query.primary_node ||
      query.has_input_activation || query.has_batchnorm ||
      !is_conv_family_stage(query.stage_type)) {
    return false;
  }

  if (query.group_kind == "ConvBias") {
    if (!query.has_bias || query.activation.has_value()) {
      return false;
    }
  } else if (query.group_kind == "ConvActivation" ||
             query.group_kind == "ConvBiasActivation") {
    if (!query.activation.has_value() ||
        (query.group_kind == "ConvBiasActivation" && !query.has_bias) ||
        !policy.post_ops->allow_stage_activation_fusion(query.stage_type,
                                                        *query.activation)) {
      return false;
    }
  } else {
    return false;
  }

  StagePlacementQuery placement_query{};
  placement_query.backend = policy.backend;
  placement_query.stage_type = query.stage_type;
  placement_query.node = query.primary_node;
  placement_query.element_type = query.element_type;
  placement_query.traits = query.traits;
  return is_vendor_image_placement(
      policy.placement->select_placement(placement_query));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
