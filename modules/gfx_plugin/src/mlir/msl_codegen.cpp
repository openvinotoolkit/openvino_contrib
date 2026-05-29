// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen.hpp"

#include "mlir/msl_codegen_apple_mps.hpp"
#include "mlir/msl_codegen_apple_msl.hpp"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_mps_family_required_pooling_stage(std::string_view stage_type) {
  return stage_type == "MaxPool" || stage_type == "AvgPool";
}

bool source_has_owned_exact_external_buffer_abi(const KernelSource &source) {
  if (source.msl_source.empty() || !source.module) {
    return false;
  }

  if (module_has_mpsrt_ops_program(source.module)) {
    GfxMpsrtProgram program{};
    if (!read_module_mpsrt_program(source.module, program) || !program.valid) {
      OPENVINO_THROW("GFX MSL: invalid typed MPSRT program");
    }
    if (gfx_mpsrt_program_has_custom_dispatch_stage(program) &&
        (!program.external_buffer_abi.valid ||
         !program.external_buffer_abi.has_buffer_count ||
         !program.external_buffer_abi.has_output_buffer_count ||
         !program.external_buffer_abi.has_buffer_roles)) {
      OPENVINO_THROW(
          "GFX MSL: typed custom-dispatch MPSRT program is missing exact "
          "external-buffer ABI");
    }
  }

  const auto abi = read_module_mpsrt_external_buffer_abi(source.module);
  if (!abi.valid || !abi.has_buffer_count || !abi.has_output_buffer_count ||
      !abi.has_buffer_roles) {
    return false;
  }
  return true;
}

} // namespace

GfxMpsrtKernelSourcePlan configure_apple_metal_kernel_source_plan_for_stage(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const GpuBufferManager *buffer_manager, std::string_view stage_type,
    bool has_bias, bool has_activation, bool has_batchnorm,
    ActivationKind activation, const ov::element::Type &storage_type,
    bool has_runtime_slice_params,
    const std::optional<ov::Shape> &runtime_input_shape,
    const BiasParams *bias_params, const GfxStageRuntimeTraits &traits) {
  const bool conv_swish_vendor_candidate =
      has_activation && activation == ActivationKind::Swish &&
      (stage_type == "Convolution" || stage_type == "GroupConvolution" ||
       stage_type == "GroupConv2D");
  if (source.module && conv_swish_vendor_candidate) {
    auto vendor_source_plan =
        configure_apple_mps_vendor_kernel_source_plan_for_node(
            source, node, buffer_manager, stage_type, has_bias, has_activation,
            has_batchnorm, activation, bias_params, traits);
    if (vendor_source_plan.valid()) {
      return vendor_source_plan;
    }
  }

  if (source.module && is_mps_family_required_pooling_stage(stage_type)) {
    auto vendor_source_plan =
        configure_apple_mps_vendor_kernel_source_plan_for_node(
            source, node, buffer_manager, stage_type, has_bias, has_activation,
            has_batchnorm, activation, bias_params, traits);
    if (vendor_source_plan.valid()) {
      return vendor_source_plan;
    }
    OPENVINO_THROW(
        "GFX Metal: Pooling source planning requires a direct MPS/MPSGraph-"
        "family route. MSL Pooling fallback is disabled until it has explicit "
        "MPS-family rejection evidence and an op-owned narrow artifact.");
  }

  if (source_has_owned_exact_external_buffer_abi(source)) {
    return {};
  }

  if (source.module && !conv_swish_vendor_candidate) {
    auto vendor_source_plan =
        configure_apple_mps_vendor_kernel_source_plan_for_node(
            source, node, buffer_manager, stage_type, has_bias, has_activation,
            has_batchnorm, activation, bias_params, traits);
    if (vendor_source_plan.valid()) {
      return vendor_source_plan;
    }
  }

  return configure_apple_metal_msl_kernel_source_plan(
      source, node, buffer_manager, stage_type, storage_type,
      has_runtime_slice_params, runtime_input_shape, has_bias, has_activation,
      has_batchnorm);
}

} // namespace gfx_plugin
} // namespace ov
