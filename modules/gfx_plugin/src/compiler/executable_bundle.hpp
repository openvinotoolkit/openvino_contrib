// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/manifest.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct ExecutableStageRecord {
  uint64_t stage_record_key = 0;
  std::string kernel_unit_id;
  std::string kernel_unit_kind;
  LoweringRouteKind execution_kind = LoweringRouteKind::Unsupported;
  size_t artifact_descriptor_index = 0;
};

enum class KernelArtifactOrigin {
  Unknown,
  Common,
  Metadata,
  VendorPrimitive,
  Generated,
  HandwrittenException,
};

enum class KernelArtifactPayloadKind {
  None,
  VendorDescriptor,
  MslSource,
  OpenClSource,
};

class KernelArtifactPayload {
public:
  virtual ~KernelArtifactPayload() = default;

  virtual KernelArtifactPayloadKind payload_kind() const noexcept = 0;
  virtual std::string_view backend_domain() const noexcept = 0;
  virtual std::string_view source_id() const noexcept = 0;
  virtual std::string_view entry_point() const noexcept = 0;
  virtual bool valid() const noexcept = 0;
};

struct KernelDescriptor {
  std::string kernel_id;
  std::string op_family;
  std::string backend_domain;
  KernelArtifactOrigin origin = KernelArtifactOrigin::Unknown;
  std::vector<std::string> tensor_roles;
  std::vector<std::string> scalar_roles;
  std::string layout_contract = "logical";
  std::string precision_contract = "inferred";
  std::string dispatch_contract = "manifest";
  std::string exception_ticket;
  std::string exception_reason;
  std::string exception_removal_condition;
};

struct KernelArtifactDescriptor {
  uint64_t stage_record_key = 0;
  std::string manifest_ref;
  std::string abi_fingerprint;
  std::string artifact_key;
  KernelDescriptor kernel;
  KernelArtifactPayloadKind payload_kind = KernelArtifactPayloadKind::None;
  std::string entry_point;
  std::string compile_options_key;
  uint32_t abi_arg_count = 0;
  uint32_t abi_output_arg_count = 0;
  bool optional_cache_payload_allowed = true;
};

struct KernelArtifactPayloadRecord {
  size_t artifact_descriptor_index = 0;
  std::string artifact_key;
  std::shared_ptr<const KernelArtifactPayload> payload;
};

struct ExecutableBundleVerificationResult {
  std::vector<std::string> diagnostics;

  bool valid() const noexcept { return diagnostics.empty(); }
};

struct ExecutableBundle {
  uint32_t schema_version = 1;
  std::string target_fingerprint;
  ManifestBundle manifest;
  MemoryPlan memory_plan;
  std::vector<ExecutableStageRecord> stages;
  std::vector<KernelArtifactDescriptor> artifact_descriptors;
  std::vector<KernelArtifactPayloadRecord> artifact_payloads;

  ExecutableBundleVerificationResult verify() const;
  bool valid() const;
  std::shared_ptr<const KernelArtifactPayload>
  find_artifact_payload(const std::string &artifact_key) const;
};

using KernelArtifactPayloadResolver =
    std::function<std::shared_ptr<const KernelArtifactPayload>(
        KernelArtifactDescriptor &descriptor, const PlannedOperation &op)>;

class ExecutableBundleBuilder final {
public:
  explicit ExecutableBundleBuilder(KernelArtifactPayloadResolver resolver = {});

  ExecutableBundle build(const ManifestBundle &manifest) const;
  ExecutableBundle build(const ManifestBundle &manifest,
                         const LoweringPlan &lowering_plan) const;

private:
  KernelArtifactPayloadResolver m_payload_resolver;
};

std::string_view
kernel_artifact_origin_to_string(KernelArtifactOrigin origin) noexcept;
std::string_view
kernel_artifact_payload_kind_to_string(KernelArtifactPayloadKind kind) noexcept;

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
