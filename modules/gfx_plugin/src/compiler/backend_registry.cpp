// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/backend_registry.hpp"

#include <utility>

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "runtime/gfx_backend_utils.hpp"
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
      KernelArtifactPayloadResolver artifact_payload_resolver = {})
      : m_id(target.backend_id()), m_target(std::move(target)),
        m_capabilities(m_target, std::move(operation_policy),
                       fusion_capabilities, post_op_fusion_capabilities),
        m_legalizer(m_capabilities),
        m_kernel_registry(std::move(kernel_registry)),
        m_lowering_planner(m_target, m_kernel_registry),
        m_pipeline_options(pipeline_options),
        m_artifact_payload_resolver(std::move(artifact_payload_resolver)) {}

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

private:
  std::string m_id;
  BackendTarget m_target;
  BackendCapabilities m_capabilities;
  OperationLegalizer m_legalizer;
  KernelRegistry m_kernel_registry;
  LoweringPlanner m_lowering_planner;
  transforms::PipelineOptions m_pipeline_options;
  KernelArtifactPayloadResolver m_artifact_payload_resolver;
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

std::vector<std::shared_ptr<const BackendModule>> make_default_modules() {
  std::vector<std::shared_ptr<const BackendModule>> modules;
  if (backend_supported(GpuBackend::Metal)) {
    const auto target = BackendTarget::from_backend(GpuBackend::Metal);
    modules.push_back(std::make_shared<StaticBackendModule>(
        target, make_metal_operation_support_policy(),
        make_metal_kernel_registry(target), make_attention_pipeline_options(),
        make_metal_fusion_capabilities(),
        make_post_op_fusion_capabilities(GpuBackend::Metal),
        make_metal_kernel_artifact_payload_resolver()));
  }
  if (backend_supported(GpuBackend::OpenCL)) {
    const auto target = BackendTarget::from_backend(GpuBackend::OpenCL);
    modules.push_back(std::make_shared<StaticBackendModule>(
        target, make_opencl_operation_support_policy(),
        make_opencl_kernel_registry(target), transforms::PipelineOptions{},
        FusionCapabilities{},
        make_post_op_fusion_capabilities(GpuBackend::OpenCL),
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
