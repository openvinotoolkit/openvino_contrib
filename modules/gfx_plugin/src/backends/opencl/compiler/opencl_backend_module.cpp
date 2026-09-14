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

GpuParallelismProfile
make_opencl_parallelism_profile_for_target(const BackendTarget &target) {
  if (target.device_profile() == "opencl_adreno") {
    return ov::gfx_plugin::make_opencl_adreno_parallelism_profile();
  }
  if (target.device_profile() == "opencl_broadcom_v3d") {
    return ov::gfx_plugin::make_opencl_broadcom_v3d_parallelism_profile();
  }
  return ov::gfx_plugin::make_opencl_parallelism_profile();
}

BackendExecutionCapabilities
make_opencl_execution_capabilities(const BackendTarget &target) {
  BackendExecutionCapabilities capabilities;
  capabilities.custom_kernel_dispatch_enabled = true;
  capabilities.custom_kernel_dispatch_profile =
      make_opencl_parallelism_profile_for_target(target);
  return capabilities;
}

} // namespace

std::shared_ptr<const BackendModule>
make_opencl_backend_module(BackendTarget target) {
  StaticBackendModuleConfig config;
  config.target = std::move(target);
  config.kernel_registry = make_opencl_kernel_registry(config.target);
  config.operation_policy =
      make_opencl_operation_support_policy(config.kernel_registry);
  config.post_op_fusion_capabilities =
      make_opencl_post_op_fusion_capabilities();
  config.stage_placement_policy = make_opencl_stage_placement_policy();
  config.execution_capabilities =
      make_opencl_execution_capabilities(config.target);
  config.artifact_descriptor_resolver =
      make_opencl_kernel_artifact_descriptor_resolver();
  config.artifact_payload_resolver =
      make_opencl_kernel_artifact_payload_resolver();
  config.cache_payload_encoder = make_opencl_cache_payload_encoder();
  config.cache_payload_decoder = make_opencl_cache_payload_decoder();
  return make_static_backend_module(std::move(config));
}

std::shared_ptr<const BackendModule> make_opencl_backend_module() {
  return make_opencl_backend_module(
      BackendTarget::from_backend(GpuBackend::OpenCL));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
