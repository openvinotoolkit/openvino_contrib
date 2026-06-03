// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_backend_module.hpp"

#include <utility>

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "backends/opencl/compiler/opencl_stage_placement.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/static_backend_module.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

PostOpFusionCapabilities make_opencl_post_op_fusion_capabilities() {
  PostOpFusionCapabilities capabilities;
  capabilities.enable_bias_fusion_for_group_convolution = false;
  capabilities.enable_batchnorm_fusion_for_group_convolution = false;
  capabilities.enable_activation_fusion_for_group_convolution = false;
  capabilities.enable_sigmoid_activation_fusion = false;
  capabilities.enable_tanh_activation_fusion = false;
  capabilities.enable_elu_activation_fusion = false;
  capabilities.enable_prelu_activation_fusion = false;
  capabilities.enable_gelu_activation_fusion = false;
  capabilities.enable_hswish_activation_fusion = false;
  capabilities.enable_hsigmoid_activation_fusion = false;
  capabilities.enable_abs_activation_fusion = false;
  capabilities.enable_sign_activation_fusion = false;
  return capabilities;
}

BackendExecutionCapabilities make_opencl_execution_capabilities() {
  BackendExecutionCapabilities capabilities;
  capabilities.source_kernel_dispatch_enabled = true;
  capabilities.fallback_parallelism = make_opencl_parallelism_profile();
  return capabilities;
}

} // namespace

std::shared_ptr<const BackendModule> make_opencl_backend_module() {
  StaticBackendModuleConfig config;
  config.target = BackendTarget::from_backend(GpuBackend::OpenCL);
  config.operation_policy = make_opencl_operation_support_policy();
  config.kernel_registry = make_opencl_kernel_registry(config.target);
  config.post_op_fusion_capabilities =
      make_opencl_post_op_fusion_capabilities();
  config.stage_placement_policy = make_opencl_stage_placement_policy();
  config.execution_capabilities = make_opencl_execution_capabilities();
  config.artifact_payload_resolver =
      make_opencl_kernel_artifact_payload_resolver();
  return make_static_backend_module(std::move(config));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
