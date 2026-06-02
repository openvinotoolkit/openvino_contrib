// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/backend_registry.hpp"

#include <utility>

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/metal/compiler/metal_stage_placement.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "backends/opencl/compiler/opencl_stage_placement.hpp"
#include "compiler/backend_config.hpp"
#include "transforms/pipeline.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

class StaticBackendModule final : public BackendModule {
public:
  StaticBackendModule(
      BackendTarget target,
      std::shared_ptr<const OperationSupportPolicy> operation_policy,
      KernelRegistry kernel_registry,
      transforms::PipelineOptions pipeline_options = {},
      FusionCapabilities fusion_capabilities = {},
      PostOpFusionCapabilities post_op_fusion_capabilities = {},
      std::shared_ptr<const StagePlacementPolicy> stage_placement_policy = {},
      KernelArtifactPayloadResolver artifact_payload_resolver = {},
      PipelineVendorAttentionArtifactResolver vendor_attention_artifact_resolver =
          {})
      : m_id(target.backend_id()), m_target(std::move(target)),
        m_capabilities(m_target, std::move(operation_policy),
                       fusion_capabilities, post_op_fusion_capabilities,
                       std::move(stage_placement_policy)),
        m_legalizer(m_capabilities),
        m_kernel_registry(std::move(kernel_registry)),
        m_lowering_planner(m_target, m_kernel_registry),
        m_pipeline_options(pipeline_options),
        m_artifact_payload_resolver(std::move(artifact_payload_resolver)),
        m_vendor_attention_artifact_resolver(
            std::move(vendor_attention_artifact_resolver)) {}

  const std::string &id() const noexcept override { return m_id; }

  const BackendTarget &target() const noexcept override { return m_target; }

  const BackendCapabilities &capabilities() const noexcept override {
    return m_capabilities;
  }

  const OperationLegalizer &legalizer() const noexcept override {
    return m_legalizer;
  }

  const KernelRegistry &kernel_registry() const noexcept override {
    return m_kernel_registry;
  }

  const LoweringPlanner &lowering_planner() const noexcept override {
    return m_lowering_planner;
  }

  const transforms::PipelineOptions &
  pipeline_options() const noexcept override {
    return m_pipeline_options;
  }

  std::shared_ptr<const KernelArtifactPayload>
  materialize_artifact_payload(KernelArtifactDescriptor &descriptor,
                               const PlannedOperation &op) const override {
    if (!m_artifact_payload_resolver) {
      return {};
    }
    return m_artifact_payload_resolver(descriptor, op);
  }

  PipelineVendorAttentionArtifact materialize_vendor_attention_artifact(
      uint64_t stage_record_key,
      const PipelineVendorAttentionPlan &plan) const override {
    if (!m_vendor_attention_artifact_resolver) {
      return {};
    }
    return m_vendor_attention_artifact_resolver(stage_record_key, plan);
  }

private:
  std::string m_id;
  BackendTarget m_target;
  BackendCapabilities m_capabilities;
  OperationLegalizer m_legalizer;
  KernelRegistry m_kernel_registry;
  LoweringPlanner m_lowering_planner;
  transforms::PipelineOptions m_pipeline_options;
  KernelArtifactPayloadResolver m_artifact_payload_resolver;
  PipelineVendorAttentionArtifactResolver m_vendor_attention_artifact_resolver;
};

transforms::PipelineOptions make_attention_pipeline_options() {
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

bool backend_available_in_config(GpuBackend backend) noexcept {
  switch (backend) {
  case GpuBackend::Metal:
    return kGfxBackendMetalAvailable;
  case GpuBackend::OpenCL:
    return kGfxBackendOpenCLAvailable;
  case GpuBackend::Unknown:
  default:
    return false;
  }
}

std::vector<std::shared_ptr<const BackendModule>> make_default_modules() {
  std::vector<std::shared_ptr<const BackendModule>> modules;
  if (backend_available_in_config(GpuBackend::Metal)) {
    const auto metal_target = BackendTarget::from_backend(GpuBackend::Metal);
    modules.push_back(std::make_shared<StaticBackendModule>(
        metal_target, make_metal_operation_support_policy(),
        make_metal_kernel_registry(metal_target),
        make_attention_pipeline_options(), make_metal_fusion_capabilities(),
        make_post_op_fusion_capabilities(GpuBackend::Metal),
        make_metal_stage_placement_policy(),
        make_metal_kernel_artifact_payload_resolver(),
        make_metal_vendor_attention_artifact_resolver()));
  }

  if (backend_available_in_config(GpuBackend::OpenCL)) {
    const auto opencl_target = BackendTarget::from_backend(GpuBackend::OpenCL);
    modules.push_back(std::make_shared<StaticBackendModule>(
        opencl_target, make_opencl_operation_support_policy(),
        make_opencl_kernel_registry(opencl_target),
        transforms::PipelineOptions{}, FusionCapabilities{},
        make_post_op_fusion_capabilities(GpuBackend::OpenCL),
        make_opencl_stage_placement_policy(),
        make_opencl_kernel_artifact_payload_resolver()));
  }
  return modules;
}

} // namespace

BackendRegistry::BackendRegistry() : BackendRegistry(make_default_modules()) {}

BackendRegistry::BackendRegistry(
    std::vector<std::shared_ptr<const BackendModule>> modules)
    : m_modules(std::move(modules)) {}

const BackendRegistry &BackendRegistry::default_registry() {
  static const BackendRegistry registry;
  return registry;
}

std::shared_ptr<const BackendModule>
BackendRegistry::resolve(GpuBackend backend) const {
  for (const auto &module : m_modules) {
    if (module && module->target().backend() == backend) {
      return module;
    }
  }
  return {};
}

std::shared_ptr<const BackendModule>
BackendRegistry::resolve(const BackendTarget &target) const {
  const auto fingerprint = target.fingerprint();
  for (const auto &module : m_modules) {
    if (module &&
        module->target().is_compatible_with_fingerprint(fingerprint)) {
      return module;
    }
  }
  return {};
}

std::vector<BackendTarget> BackendRegistry::available_targets() const {
  std::vector<BackendTarget> targets;
  targets.reserve(m_modules.size());
  for (const auto &module : m_modules) {
    if (module) {
      targets.push_back(module->target());
    }
  }
  return targets;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
