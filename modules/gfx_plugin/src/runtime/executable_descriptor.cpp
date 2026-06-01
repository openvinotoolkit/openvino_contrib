// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/executable_descriptor.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace ov {
namespace gfx_plugin {
namespace {

constexpr uint32_t kRuntimeExecutableDescriptorSchemaVersion = 1;

bool source_payload_kind(compiler::KernelArtifactPayloadKind kind) noexcept {
  return kind == compiler::KernelArtifactPayloadKind::MslSource ||
         kind == compiler::KernelArtifactPayloadKind::OpenClSource;
}

bool payload_kind_requires_materialized_payload(
    compiler::KernelArtifactPayloadKind kind) noexcept {
  return kind == compiler::KernelArtifactPayloadKind::VendorDescriptor ||
         source_payload_kind(kind);
}

bool source_origin(compiler::KernelArtifactOrigin origin) noexcept {
  return origin == compiler::KernelArtifactOrigin::Generated ||
         origin == compiler::KernelArtifactOrigin::HandwrittenException;
}

bool same_string(std::string_view lhs, const std::string &rhs) {
  return lhs.size() == rhs.size() && lhs.compare(rhs) == 0;
}

RuntimeStageExecutableDescriptor make_stage_descriptor(
    size_t stage_index, const compiler::ExecutableStageRecord &executable_stage,
    const compiler::KernelArtifactDescriptor &artifact,
    std::shared_ptr<const compiler::KernelArtifactPayload> payload) {
  RuntimeStageExecutableDescriptor descriptor;
  descriptor.stage_index = stage_index;
  descriptor.stage_record_key = executable_stage.stage_record_key;
  descriptor.artifact_descriptor_index =
      executable_stage.artifact_descriptor_index;
  descriptor.manifest_ref = artifact.manifest_ref;
  descriptor.abi_fingerprint = artifact.abi_fingerprint;
  descriptor.artifact_key = artifact.artifact_key;
  descriptor.backend_domain = artifact.kernel.backend_domain;
  descriptor.kernel_id = artifact.kernel.kernel_id;
  descriptor.op_family = artifact.kernel.op_family;
  descriptor.origin = artifact.kernel.origin;
  descriptor.payload_kind = artifact.payload_kind;
  descriptor.entry_point = artifact.entry_point;
  descriptor.compile_options_key = artifact.compile_options_key;
  descriptor.abi_arg_count = artifact.abi_arg_count;
  descriptor.abi_output_arg_count = artifact.abi_output_arg_count;
  descriptor.dispatch_contract = artifact.kernel.dispatch_contract;
  descriptor.layout_contract = artifact.kernel.layout_contract;
  descriptor.tensor_view_only = descriptor.layout_contract == "view_only";
  descriptor.tensor_roles = artifact.kernel.tensor_roles;
  descriptor.scalar_roles = artifact.kernel.scalar_roles;
  descriptor.exception_ticket = artifact.kernel.exception_ticket;
  descriptor.exception_reason = artifact.kernel.exception_reason;
  descriptor.exception_removal_condition =
      artifact.kernel.exception_removal_condition;
  descriptor.optional_cache_payload_allowed =
      artifact.optional_cache_payload_allowed;
  descriptor.payload = std::move(payload);
  return descriptor;
}

} // namespace

RuntimeExecutableDescriptorVerificationResult
RuntimeExecutableDescriptor::verify(
    const compiler::ExecutableBundle &executable) const {
  RuntimeExecutableDescriptorVerificationResult result;
  if (schema_version != kRuntimeExecutableDescriptorSchemaVersion ||
      target_fingerprint.empty()) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor header is incomplete");
  }
  if (target_fingerprint != executable.target_fingerprint) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor target drift");
  }
  if (stages.size() != executable.stages.size()) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor stage count drift");
  }

  std::unordered_set<uint64_t> stage_keys;
  const size_t count = std::min(stages.size(), executable.stages.size());
  for (size_t i = 0; i < count; ++i) {
    const auto &stage = stages[i];
    const auto &executable_stage = executable.stages[i];
    if (stage.stage_index != i) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage index drift at " +
          std::to_string(i));
    }
    if (stage.stage_record_key == 0 ||
        !stage_keys.insert(stage.stage_record_key).second) {
      result.diagnostics.push_back(
          "runtime executable descriptor duplicate or empty stage key at " +
          std::to_string(i));
    }
    if (stage.stage_record_key != executable_stage.stage_record_key ||
        stage.artifact_descriptor_index !=
            executable_stage.artifact_descriptor_index) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage drift at " + std::to_string(i));
    }
    if (stage.artifact_descriptor_index >=
        executable.artifact_descriptors.size()) {
      result.diagnostics.push_back(
          "runtime executable descriptor artifact index out of range at " +
          std::to_string(i));
      continue;
    }

    const auto &artifact =
        executable.artifact_descriptors[stage.artifact_descriptor_index];
    if (stage.manifest_ref != artifact.manifest_ref ||
        stage.abi_fingerprint != artifact.abi_fingerprint ||
        stage.artifact_key != artifact.artifact_key ||
        stage.backend_domain != artifact.kernel.backend_domain ||
        stage.kernel_id != artifact.kernel.kernel_id ||
        stage.op_family != artifact.kernel.op_family ||
        stage.origin != artifact.kernel.origin ||
        stage.payload_kind != artifact.payload_kind ||
        stage.entry_point != artifact.entry_point ||
        stage.compile_options_key != artifact.compile_options_key ||
        stage.abi_arg_count != artifact.abi_arg_count ||
        stage.abi_output_arg_count != artifact.abi_output_arg_count ||
        stage.dispatch_contract != artifact.kernel.dispatch_contract ||
        stage.layout_contract != artifact.kernel.layout_contract ||
        stage.tensor_view_only !=
            (artifact.kernel.layout_contract == "view_only") ||
        stage.tensor_roles != artifact.kernel.tensor_roles ||
        stage.scalar_roles != artifact.kernel.scalar_roles ||
        stage.exception_ticket != artifact.kernel.exception_ticket ||
        stage.exception_reason != artifact.kernel.exception_reason ||
        stage.exception_removal_condition !=
            artifact.kernel.exception_removal_condition ||
        stage.optional_cache_payload_allowed !=
            artifact.optional_cache_payload_allowed) {
      result.diagnostics.push_back(
          "runtime executable descriptor artifact drift at " +
          std::to_string(i));
    }
    if (stage.manifest_ref.empty() || stage.abi_fingerprint.empty() ||
        stage.artifact_key.empty() || stage.backend_domain.empty() ||
        stage.kernel_id.empty() || stage.op_family.empty()) {
      result.diagnostics.push_back(
          "runtime executable descriptor has incomplete identity at " +
          std::to_string(i));
    }
    if (stage.payload_kind ==
            compiler::KernelArtifactPayloadKind::VendorDescriptor &&
        stage.origin != compiler::KernelArtifactOrigin::VendorPrimitive) {
      result.diagnostics.push_back("runtime executable descriptor vendor "
                                   "payload is not vendor-owned at " +
                                   std::to_string(i));
    }
    if (source_payload_kind(stage.payload_kind) &&
        !source_origin(stage.origin)) {
      result.diagnostics.push_back("runtime executable descriptor source "
                                   "payload has non-source origin at " +
                                   std::to_string(i));
    }
    if (stage.origin == compiler::KernelArtifactOrigin::HandwrittenException &&
        (stage.exception_ticket.empty() || stage.exception_reason.empty() ||
         stage.exception_removal_condition.empty())) {
      result.diagnostics.push_back("runtime executable descriptor handwritten "
                                   "exception contract is incomplete at " +
                                   std::to_string(i));
    }
    if (stage.origin != compiler::KernelArtifactOrigin::HandwrittenException &&
        (!stage.exception_ticket.empty() || !stage.exception_reason.empty() ||
         !stage.exception_removal_condition.empty())) {
      result.diagnostics.push_back(
          "runtime executable descriptor non-exception stage carries exception "
          "contract at " +
          std::to_string(i));
    }
    if (source_payload_kind(stage.payload_kind) &&
        (stage.abi_arg_count == 0 || stage.abi_output_arg_count == 0)) {
      result.diagnostics.push_back(
          "runtime executable descriptor source ABI is incomplete at " +
          std::to_string(i));
    }
    const auto executable_payload =
        executable.find_artifact_payload(stage.artifact_key);
    if (stage.payload != executable_payload) {
      result.diagnostics.push_back(
          "runtime executable descriptor payload drift at " +
          std::to_string(i));
    }
    if (payload_kind_requires_materialized_payload(stage.payload_kind) &&
        !stage.payload) {
      result.diagnostics.push_back(
          "runtime executable descriptor requires a materialized payload at " +
          std::to_string(i));
    }
    if (stage.payload) {
      if (!stage.payload->valid() ||
          stage.payload->payload_kind() != stage.payload_kind ||
          !same_string(stage.payload->backend_domain(), stage.backend_domain) ||
          !same_string(stage.payload->source_id(), stage.kernel_id) ||
          !same_string(stage.payload->entry_point(), stage.entry_point)) {
        result.diagnostics.push_back(
            "runtime executable descriptor payload identity drift at " +
            std::to_string(i));
      }
    }
  }
  return result;
}

bool RuntimeExecutableDescriptor::valid(
    const compiler::ExecutableBundle &executable) const {
  return verify(executable).valid();
}

RuntimeExecutableDescriptor RuntimeExecutableDescriptorBuilder::build(
    const compiler::ExecutableBundle &executable) const {
  RuntimeExecutableDescriptor descriptor;
  descriptor.schema_version = kRuntimeExecutableDescriptorSchemaVersion;
  descriptor.target_fingerprint = executable.target_fingerprint;
  descriptor.stages.reserve(executable.stages.size());
  for (size_t i = 0; i < executable.stages.size(); ++i) {
    const auto &executable_stage = executable.stages[i];
    if (executable_stage.artifact_descriptor_index >=
        executable.artifact_descriptors.size()) {
      RuntimeStageExecutableDescriptor stage;
      stage.stage_index = i;
      stage.stage_record_key = executable_stage.stage_record_key;
      stage.artifact_descriptor_index =
          executable_stage.artifact_descriptor_index;
      descriptor.stages.push_back(std::move(stage));
      continue;
    }
    const auto &artifact =
        executable
            .artifact_descriptors[executable_stage.artifact_descriptor_index];
    descriptor.stages.push_back(make_stage_descriptor(
        i, executable_stage, artifact,
        executable.find_artifact_payload(artifact.artifact_key)));
  }
  return descriptor;
}

} // namespace gfx_plugin
} // namespace ov
