// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "compiler/cache_envelope.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/lowering_planner.hpp"
#include "compiler/operation_legalizer.hpp"
#include "compiler/operation_support.hpp"

namespace ov {
namespace gfx_plugin {

namespace transforms {
struct PipelineOptions;
}

namespace compiler {

class BackendModule {
public:
  virtual ~BackendModule() = default;

  virtual const std::string &id() const noexcept = 0;
  virtual const BackendTarget &target() const noexcept = 0;
  virtual const BackendCapabilities &capabilities() const noexcept = 0;
  virtual const OperationLegalizer &legalizer() const noexcept = 0;
  virtual const KernelRegistry &kernel_registry() const noexcept = 0;
  virtual const LoweringPlanner &lowering_planner() const noexcept = 0;
  virtual const transforms::PipelineOptions &
  pipeline_options() const noexcept = 0;
  virtual bool finalize_artifact_descriptor(
      KernelArtifactDescriptor &descriptor,
      const PlannedOperation &op) const = 0;
  virtual std::shared_ptr<const KernelArtifactPayload>
  materialize_artifact_payload(const KernelArtifactDescriptor &descriptor,
                               const PlannedOperation &op) const = 0;
  virtual CacheBackendPayloadRecord
  encode_cache_payload(const KernelArtifactDescriptor &descriptor,
                       const KernelArtifactPayloadRecord &payload_record) const = 0;
  virtual std::shared_ptr<const KernelArtifactPayload>
  decode_cache_payload(const CacheBackendPayloadRecord &payload,
                       const KernelArtifactDescriptor &descriptor) const = 0;
  virtual PipelineVendorAttentionArtifact materialize_vendor_attention_artifact(
      uint64_t stage_record_key,
      const PipelineVendorAttentionPlan &plan) const = 0;
};

class BackendRegistry final {
public:
  BackendRegistry();
  explicit BackendRegistry(
      std::vector<std::shared_ptr<const BackendModule>> modules);

  static const BackendRegistry &default_registry();

  std::shared_ptr<const BackendModule>
  resolve(const BackendTarget &target) const;
  std::vector<BackendTarget> available_targets() const;

private:
  std::vector<std::shared_ptr<const BackendModule>> m_modules;
};

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
