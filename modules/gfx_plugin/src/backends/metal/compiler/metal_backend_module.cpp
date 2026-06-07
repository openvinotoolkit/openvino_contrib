// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/metal_backend_module.hpp"

#include <utility>

#include "backends/metal/compiler/apple_mlir_stage_hooks.hpp"
#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/metal/compiler/metal_stage_placement.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/static_backend_module.hpp"
#include "transforms/pipeline.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

transforms::PipelineOptions make_metal_pipeline_options() {
  transforms::PipelineOptions options;
  options.preserve_scaled_dot_product_attention = true;
  options.canonicalize_sigmoid_before_ranking = true;
  options.enable_llm_attention_fusions = true;
  return options;
}

FusionCapabilities make_metal_fusion_capabilities() {
  FusionCapabilities capabilities;
  capabilities.enable_generic_attention_fusion = false;
  capabilities.supports_vendor_attention_stage = true;
  capabilities.enable_conv_activation_fusion = true;
  capabilities.enable_precision_sensitive_arithmetic_fusion = false;
  return capabilities;
}

PostOpFusionCapabilities make_metal_post_op_fusion_capabilities() {
  PostOpFusionCapabilities capabilities;
  capabilities.enable_elu_activation_fusion = false;
  capabilities.enable_prelu_activation_fusion = false;
  capabilities.enable_gelu_activation_fusion = false;
  capabilities.enable_hswish_activation_fusion = false;
  capabilities.enable_hsigmoid_activation_fusion = false;
  capabilities.enable_sign_activation_fusion = false;
  return capabilities;
}

BackendExecutionCapabilities make_metal_execution_capabilities() {
  BackendExecutionCapabilities capabilities;
  capabilities.custom_kernel_dispatch_profile = make_metal_parallelism_profile();
  return capabilities;
}

} // namespace

std::shared_ptr<const BackendModule>
make_metal_backend_module(BackendTarget target) {
  ensure_apple_mlir_stage_backend_hooks_registered();

  StaticBackendModuleConfig config;
  config.target = std::move(target);
  config.operation_policy = make_metal_operation_support_policy();
  config.kernel_registry = make_metal_kernel_registry(config.target);
  config.pipeline_options = make_metal_pipeline_options();
  config.fusion_capabilities = make_metal_fusion_capabilities();
  config.post_op_fusion_capabilities = make_metal_post_op_fusion_capabilities();
  config.stage_placement_policy = make_metal_stage_placement_policy();
  config.execution_capabilities = make_metal_execution_capabilities();
  config.artifact_descriptor_resolver =
      make_metal_kernel_artifact_descriptor_resolver();
  config.artifact_payload_resolver =
      make_metal_kernel_artifact_payload_resolver();
  config.vendor_attention_artifact_resolver =
      make_metal_vendor_attention_artifact_resolver();
  return make_static_backend_module(std::move(config));
}

std::shared_ptr<const BackendModule> make_metal_backend_module() {
  return make_metal_backend_module(
      BackendTarget::from_backend(GpuBackend::Metal));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
