// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/stage_compiler_policy.hpp"

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

} // namespace

StageCustomKernelDispatchPolicy
make_stage_custom_kernel_dispatch_policy(
    const BackendExecutionCapabilities &execution) {
  StageCustomKernelDispatchPolicy policy{};
  policy.enabled = execution.custom_kernel_dispatch_enabled;
  policy.profile = execution.custom_kernel_dispatch_profile;
  return policy;
}

StageCompilerPolicy make_stage_compiler_policy_from_capabilities(
    const BackendCapabilities &capabilities) {
  StageCompilerPolicy policy{};
  policy.target = capabilities.target();
  policy.backend = capabilities.backend();
  policy.placement = capabilities.stage_placement();
  policy.post_ops = &capabilities.post_ops();
  policy.custom_kernel_dispatch =
      make_stage_custom_kernel_dispatch_policy(capabilities.execution());
  return policy;
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
