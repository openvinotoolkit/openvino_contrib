// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/executable_bundle.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeTensorBindingContract {
  std::string logical_name;
  std::string memory_region_id;
  std::string role;
  std::string element_type;
  std::string partial_shape;
  std::string layout = "logical";
  std::string storage_kind = "device_buffer";
  std::string lifetime_class;
  std::string alias_group;
  std::string stateful_prebind_variable_id;
  std::string stateful_prebind_shape_rule = "none";
  int64_t stateful_prebind_shape_axis = -1;
  bool external_binding = false;
  bool host_visible = false;
};

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
  std::string layout_contract = "logical";
  std::string runtime_shape_rule = "static_or_descriptor";
  bool requires_runtime_shape_args = false;
  bool tensor_view_only = false;
  uint32_t submission_stage_weight = 1;
  uint64_t submission_macs_estimate = 0;
  bool submission_dependency_boundary = false;
  std::string stateful_effect = "none";
  std::string stateful_variable_id;
  std::vector<std::string> tensor_roles;
  std::vector<std::string> scalar_roles;
  std::string exception_ticket;
  std::string exception_reason;
  std::string exception_removal_condition;
  bool optional_cache_payload_allowed = true;
  std::shared_ptr<const compiler::KernelArtifactPayload> payload;
  std::vector<RuntimeTensorBindingContract> input_bindings;
  std::vector<RuntimeTensorBindingContract> output_bindings;
};

struct RuntimeMemoryRegionDescriptor {
  std::string region_id;
  std::string logical_tensor_name;
  std::string kind;
  std::string element_type;
  std::string partial_shape;
  std::string layout = "logical";
  std::string storage_kind = "device_buffer";
  std::string alias_group;
  size_t first_stage = 0;
  size_t last_stage = 0;
  bool external_binding = false;
  bool host_visible = false;
};

struct RuntimeMemoryAliasGroupDescriptor {
  std::string group_id;
  std::vector<std::string> region_ids;
  bool output_aliasing = false;
};

struct RuntimeTransientArenaDescriptor {
  std::string arena_id;
  std::string storage_kind = "device_buffer";
  std::vector<std::string> region_ids;
};

struct RuntimeMemoryPlanDescriptor {
  uint32_t schema_version = 1;
  std::string fingerprint;
  std::vector<RuntimeMemoryRegionDescriptor> regions;
  std::vector<RuntimeMemoryAliasGroupDescriptor> alias_groups;
  std::vector<RuntimeTransientArenaDescriptor> transient_arenas;
  bool hidden_host_copies_allowed = false;

  bool has_region(std::string_view region_id) const;
  bool has_alias_group(std::string_view group_id) const;
};

struct RuntimeExecutableDescriptorVerificationResult {
  std::vector<std::string> diagnostics;

  bool valid() const noexcept { return diagnostics.empty(); }
};

struct RuntimeExecutableDescriptor {
  uint32_t schema_version = 1;
  std::string target_fingerprint;
  RuntimeMemoryPlanDescriptor memory_plan;
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
