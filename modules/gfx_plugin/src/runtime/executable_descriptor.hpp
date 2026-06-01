// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "compiler/executable_bundle.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeStageExecutableDescriptor {
  size_t stage_index = 0;
  uint64_t stage_record_key = 0;
  size_t artifact_descriptor_index = 0;
  std::string manifest_ref;
  std::string abi_fingerprint;
  std::string artifact_key;
  std::string backend_domain;
  std::string kernel_id;
  std::string op_family;
  compiler::KernelArtifactOrigin origin =
      compiler::KernelArtifactOrigin::Unknown;
  compiler::KernelArtifactPayloadKind payload_kind =
      compiler::KernelArtifactPayloadKind::None;
  std::string entry_point;
  std::string compile_options_key;
  uint32_t abi_arg_count = 0;
  uint32_t abi_output_arg_count = 0;
  std::string dispatch_contract;
  std::vector<std::string> tensor_roles;
  std::vector<std::string> scalar_roles;
  std::string exception_ticket;
  std::string exception_reason;
  std::string exception_removal_condition;
  bool optional_cache_payload_allowed = true;
  std::shared_ptr<const compiler::KernelArtifactPayload> payload;
};

struct RuntimeExecutableDescriptorVerificationResult {
  std::vector<std::string> diagnostics;

  bool valid() const noexcept { return diagnostics.empty(); }
};

struct RuntimeExecutableDescriptor {
  uint32_t schema_version = 1;
  std::string target_fingerprint;
  std::vector<RuntimeStageExecutableDescriptor> stages;

  RuntimeExecutableDescriptorVerificationResult
  verify(const compiler::ExecutableBundle &executable) const;
  bool valid(const compiler::ExecutableBundle &executable) const;
};

class RuntimeExecutableDescriptorBuilder final {
public:
  RuntimeExecutableDescriptor
  build(const compiler::ExecutableBundle &executable) const;
};

} // namespace gfx_plugin
} // namespace ov
