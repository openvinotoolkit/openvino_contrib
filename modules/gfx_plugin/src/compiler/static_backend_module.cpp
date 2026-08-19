// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/static_backend_module.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

class StaticBackendModule final : public BackendModule {
public:
  explicit StaticBackendModule(StaticBackendModuleConfig config)
      : m_id(config.target.backend_id()), m_target(std::move(config.target)),
        m_capabilities(
            m_target, std::move(config.operation_policy),
            config.fusion_capabilities, config.post_op_fusion_capabilities,
            std::move(config.stage_placement_policy),
            std::move(config.execution_capabilities),
            config.precision_capabilities, config.artifact_format_capabilities),
        m_legalizer(m_capabilities),
        m_kernel_registry(std::move(config.kernel_registry)),
        m_lowering_planner(m_target, m_kernel_registry),
        m_pipeline_options(config.pipeline_options),
        m_artifact_descriptor_resolver(
            std::move(config.artifact_descriptor_resolver)),
        m_artifact_payload_resolver(
            std::move(config.artifact_payload_resolver)),
        m_cache_payload_encoder(std::move(config.cache_payload_encoder)),
        m_cache_payload_decoder(std::move(config.cache_payload_decoder)),
        m_vendor_attention_artifact_resolver(
            std::move(config.vendor_attention_artifact_resolver)) {}

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

  bool finalize_artifact_descriptor(KernelArtifactDescriptor &descriptor,
                                    const PlannedOperation &op) const override {
    if (!m_artifact_descriptor_resolver) {
      finalize_kernel_artifact_descriptor_identity(descriptor);
      return true;
    }
    return m_artifact_descriptor_resolver(descriptor, op);
  }

  std::shared_ptr<const KernelArtifactPayload>
  materialize_artifact_payload(const KernelArtifactDescriptor &descriptor,
                               const PlannedOperation &op) const override {
    if (!m_artifact_payload_resolver) {
      return {};
    }
    return m_artifact_payload_resolver(descriptor, op);
  }

  CacheBackendPayloadRecord
  encode_cache_payload(const KernelArtifactDescriptor &descriptor,
                       const KernelArtifactPayloadRecord &payload_record) const override {
    if (!m_cache_payload_encoder) {
      return {};
    }
    return m_cache_payload_encoder(descriptor, payload_record);
  }

  std::shared_ptr<const KernelArtifactPayload>
  decode_cache_payload(const CacheBackendPayloadRecord &payload,
                       const KernelArtifactDescriptor &descriptor) const override {
    if (!m_cache_payload_decoder) {
      return {};
    }
    return m_cache_payload_decoder(payload, descriptor);
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
  KernelArtifactDescriptorResolver m_artifact_descriptor_resolver;
  KernelArtifactPayloadResolver m_artifact_payload_resolver;
  CacheBackendPayloadEncoder m_cache_payload_encoder;
  CacheBackendPayloadDecoder m_cache_payload_decoder;
  PipelineVendorAttentionArtifactResolver m_vendor_attention_artifact_resolver;
};

} // namespace

std::shared_ptr<const BackendModule>
make_static_backend_module(StaticBackendModuleConfig config) {
  return std::make_shared<StaticBackendModule>(std::move(config));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
