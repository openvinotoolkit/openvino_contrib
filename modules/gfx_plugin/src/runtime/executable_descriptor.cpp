// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/executable_descriptor.hpp"

#include <algorithm>
#include <cstddef>
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

const compiler::MemoryRegion *
find_memory_region(const compiler::MemoryPlan &plan,
                   std::string_view region_id) {
  for (const auto &region : plan.regions) {
    if (same_string(region_id, region.region_id)) {
      return &region;
    }
  }
  return nullptr;
}

RuntimeTensorBindingContract make_tensor_binding_contract(
    const compiler::TensorContract &contract,
    const compiler::MemoryPlan &memory_plan) {
  RuntimeTensorBindingContract binding;
  binding.logical_name = contract.logical_name;
  binding.memory_region_id = contract.memory_region_id;
  binding.role = std::string(compiler::tensor_contract_role_to_string(contract.role));
  binding.element_type = contract.element_type;
  binding.partial_shape = contract.partial_shape;
  binding.layout = contract.layout;
  binding.storage_kind = contract.storage_kind;
  binding.lifetime_class = contract.lifetime_class;
  if (const auto *region =
          find_memory_region(memory_plan, contract.memory_region_id)) {
    binding.alias_group = region->alias_group;
    binding.external_binding = region->external_binding;
    binding.host_visible = region->host_visible;
  }
  return binding;
}

std::vector<RuntimeTensorBindingContract> make_tensor_binding_contracts(
    const std::vector<compiler::TensorContract> &contracts,
    const compiler::MemoryPlan &memory_plan) {
  std::vector<RuntimeTensorBindingContract> bindings;
  bindings.reserve(contracts.size());
  for (const auto &contract : contracts) {
    bindings.push_back(make_tensor_binding_contract(contract, memory_plan));
  }
  return bindings;
}

RuntimeStageExecutableDescriptor make_stage_descriptor(
    size_t stage_index, const compiler::ExecutableStageRecord &executable_stage,
    const compiler::StageRecord &manifest_stage,
    const compiler::MemoryPlan &memory_plan,
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
  descriptor.input_bindings =
      make_tensor_binding_contracts(manifest_stage.inputs, memory_plan);
  descriptor.output_bindings =
      make_tensor_binding_contracts(manifest_stage.outputs, memory_plan);
  return descriptor;
}

RuntimeMemoryRegionDescriptor
make_memory_region_descriptor(const compiler::MemoryRegion &region) {
  RuntimeMemoryRegionDescriptor descriptor;
  descriptor.region_id = region.region_id;
  descriptor.logical_tensor_name = region.logical_tensor_name;
  descriptor.kind =
      std::string(compiler::memory_region_kind_to_string(region.kind));
  descriptor.element_type = region.element_type;
  descriptor.partial_shape = region.partial_shape;
  descriptor.layout = region.layout;
  descriptor.storage_kind = region.storage_kind;
  descriptor.alias_group = region.alias_group;
  descriptor.first_stage = region.lifetime.first_stage;
  descriptor.last_stage = region.lifetime.last_stage;
  descriptor.external_binding = region.external_binding;
  descriptor.host_visible = region.host_visible;
  return descriptor;
}

RuntimeMemoryAliasGroupDescriptor
make_memory_alias_group_descriptor(const compiler::AliasGroup &group) {
  RuntimeMemoryAliasGroupDescriptor descriptor;
  descriptor.group_id = group.group_id;
  descriptor.region_ids = group.region_ids;
  descriptor.output_aliasing = group.output_aliasing;
  return descriptor;
}

RuntimeTransientArenaDescriptor
make_transient_arena_descriptor(const compiler::TransientArena &arena) {
  RuntimeTransientArenaDescriptor descriptor;
  descriptor.arena_id = arena.arena_id;
  descriptor.storage_kind = arena.storage_kind;
  descriptor.region_ids = arena.region_ids;
  return descriptor;
}

RuntimeMemoryPlanDescriptor
make_memory_plan_descriptor(const compiler::MemoryPlan &plan) {
  RuntimeMemoryPlanDescriptor descriptor;
  descriptor.schema_version = plan.schema_version;
  descriptor.fingerprint = compiler::make_memory_plan_fingerprint(plan);
  descriptor.hidden_host_copies_allowed = plan.hidden_host_copies_allowed;
  descriptor.regions.reserve(plan.regions.size());
  for (const auto &region : plan.regions) {
    descriptor.regions.push_back(make_memory_region_descriptor(region));
  }
  descriptor.alias_groups.reserve(plan.alias_groups.size());
  for (const auto &group : plan.alias_groups) {
    descriptor.alias_groups.push_back(make_memory_alias_group_descriptor(group));
  }
  descriptor.transient_arenas.reserve(plan.transient_arenas.size());
  for (const auto &arena : plan.transient_arenas) {
    descriptor.transient_arenas.push_back(
        make_transient_arena_descriptor(arena));
  }
  return descriptor;
}

bool memory_region_matches(const RuntimeMemoryRegionDescriptor &descriptor,
                           const compiler::MemoryRegion &region) {
  return descriptor.region_id == region.region_id &&
         descriptor.logical_tensor_name == region.logical_tensor_name &&
         same_string(compiler::memory_region_kind_to_string(region.kind),
                     descriptor.kind) &&
         descriptor.element_type == region.element_type &&
         descriptor.partial_shape == region.partial_shape &&
         descriptor.layout == region.layout &&
         descriptor.storage_kind == region.storage_kind &&
         descriptor.alias_group == region.alias_group &&
         descriptor.first_stage == region.lifetime.first_stage &&
         descriptor.last_stage == region.lifetime.last_stage &&
         descriptor.external_binding == region.external_binding &&
         descriptor.host_visible == region.host_visible;
}

bool memory_alias_group_matches(
    const RuntimeMemoryAliasGroupDescriptor &descriptor,
    const compiler::AliasGroup &group) {
  return descriptor.group_id == group.group_id &&
         descriptor.region_ids == group.region_ids &&
         descriptor.output_aliasing == group.output_aliasing;
}

bool transient_arena_matches(const RuntimeTransientArenaDescriptor &descriptor,
                             const compiler::TransientArena &arena) {
  return descriptor.arena_id == arena.arena_id &&
         descriptor.storage_kind == arena.storage_kind &&
         descriptor.region_ids == arena.region_ids;
}

bool tensor_binding_matches(const RuntimeTensorBindingContract &binding,
                            const compiler::TensorContract &contract,
                            const compiler::MemoryPlan &memory_plan) {
  if (binding.logical_name != contract.logical_name ||
      binding.memory_region_id != contract.memory_region_id ||
      !same_string(compiler::tensor_contract_role_to_string(contract.role),
                   binding.role) ||
      binding.element_type != contract.element_type ||
      binding.partial_shape != contract.partial_shape ||
      binding.layout != contract.layout ||
      binding.storage_kind != contract.storage_kind ||
      binding.lifetime_class != contract.lifetime_class) {
    return false;
  }
  const auto *region = find_memory_region(memory_plan, contract.memory_region_id);
  if (!region) {
    return false;
  }
  return binding.alias_group == region->alias_group &&
         binding.external_binding == region->external_binding &&
         binding.host_visible == region->host_visible;
}

void append_tensor_binding_diagnostics(
    RuntimeExecutableDescriptorVerificationResult &result,
    const RuntimeStageExecutableDescriptor &stage,
    const compiler::StageRecord &manifest_stage,
    const compiler::MemoryPlan &memory_plan, size_t stage_index) {
  if (stage.input_bindings.size() != manifest_stage.inputs.size()) {
    result.diagnostics.push_back(
        "runtime executable descriptor input binding count drift at " +
        std::to_string(stage_index));
  }
  const size_t input_count =
      std::min(stage.input_bindings.size(), manifest_stage.inputs.size());
  for (size_t i = 0; i < input_count; ++i) {
    if (!tensor_binding_matches(stage.input_bindings[i],
                                manifest_stage.inputs[i], memory_plan)) {
      result.diagnostics.push_back(
          "runtime executable descriptor input binding drift at " +
          std::to_string(stage_index) + ":" + std::to_string(i));
    }
  }

  if (stage.output_bindings.size() != manifest_stage.outputs.size()) {
    result.diagnostics.push_back(
        "runtime executable descriptor output binding count drift at " +
        std::to_string(stage_index));
  }
  const size_t output_count =
      std::min(stage.output_bindings.size(), manifest_stage.outputs.size());
  for (size_t i = 0; i < output_count; ++i) {
    if (!tensor_binding_matches(stage.output_bindings[i],
                                manifest_stage.outputs[i], memory_plan)) {
      result.diagnostics.push_back(
          "runtime executable descriptor output binding drift at " +
          std::to_string(stage_index) + ":" + std::to_string(i));
    }
  }
}

void append_memory_plan_diagnostics(
    RuntimeExecutableDescriptorVerificationResult &result,
    const RuntimeMemoryPlanDescriptor &descriptor,
    const compiler::MemoryPlan &memory_plan) {
  const auto memory_plan_result = memory_plan.verify();
  for (const auto &diagnostic : memory_plan_result.diagnostics) {
    result.diagnostics.push_back(
        "runtime executable descriptor source memory plan invalid: " +
        diagnostic);
  }

  if (descriptor.schema_version != memory_plan.schema_version) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor memory plan schema drift");
  }
  if (descriptor.fingerprint !=
      compiler::make_memory_plan_fingerprint(memory_plan)) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor memory plan fingerprint drift");
  }
  if (descriptor.hidden_host_copies_allowed !=
      memory_plan.hidden_host_copies_allowed) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor memory plan host-copy policy drift");
  }

  if (descriptor.regions.size() != memory_plan.regions.size()) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor memory region count drift");
  }
  const size_t region_count =
      std::min(descriptor.regions.size(), memory_plan.regions.size());
  for (size_t i = 0; i < region_count; ++i) {
    if (!memory_region_matches(descriptor.regions[i],
                               memory_plan.regions[i])) {
      result.diagnostics.push_back(
          "runtime executable descriptor memory region drift at " +
          std::to_string(i));
    }
  }

  if (descriptor.alias_groups.size() != memory_plan.alias_groups.size()) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor memory alias group count drift");
  }
  const size_t alias_group_count =
      std::min(descriptor.alias_groups.size(), memory_plan.alias_groups.size());
  for (size_t i = 0; i < alias_group_count; ++i) {
    if (!memory_alias_group_matches(descriptor.alias_groups[i],
                                    memory_plan.alias_groups[i])) {
      result.diagnostics.push_back(
          "runtime executable descriptor memory alias group drift at " +
          std::to_string(i));
    }
  }

  if (descriptor.transient_arenas.size() !=
      memory_plan.transient_arenas.size()) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor transient arena count drift");
  }
  const size_t arena_count = std::min(descriptor.transient_arenas.size(),
                                     memory_plan.transient_arenas.size());
  for (size_t i = 0; i < arena_count; ++i) {
    if (!transient_arena_matches(descriptor.transient_arenas[i],
                                 memory_plan.transient_arenas[i])) {
      result.diagnostics.push_back(
          "runtime executable descriptor transient arena drift at " +
          std::to_string(i));
    }
  }
}

} // namespace

bool RuntimeMemoryPlanDescriptor::has_region(
    std::string_view region_id) const {
  return std::any_of(regions.begin(), regions.end(),
                     [&](const RuntimeMemoryRegionDescriptor &region) {
                       return region.region_id == region_id;
                     });
}

bool RuntimeMemoryPlanDescriptor::has_alias_group(
    std::string_view group_id) const {
  return std::any_of(alias_groups.begin(), alias_groups.end(),
                     [&](const RuntimeMemoryAliasGroupDescriptor &group) {
                       return group.group_id == group_id;
                     });
}

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
  append_memory_plan_diagnostics(result, memory_plan, executable.memory_plan);

  std::unordered_set<uint64_t> stage_keys;
  const size_t count = std::min(stages.size(), executable.stages.size());
  for (size_t i = 0; i < count; ++i) {
    const auto &stage = stages[i];
    const auto &executable_stage = executable.stages[i];
    if (i < executable.manifest.stages.size()) {
      append_tensor_binding_diagnostics(result, stage,
                                        executable.manifest.stages[i],
                                        executable.memory_plan, i);
    } else {
      result.diagnostics.push_back(
          "runtime executable descriptor manifest stage missing at " +
          std::to_string(i));
    }
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
  descriptor.memory_plan = make_memory_plan_descriptor(executable.memory_plan);
  descriptor.stages.reserve(executable.stages.size());
  for (size_t i = 0; i < executable.stages.size(); ++i) {
    const auto &executable_stage = executable.stages[i];
    if (i >= executable.manifest.stages.size() ||
        executable_stage.artifact_descriptor_index >=
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
        i, executable_stage, executable.manifest.stages[i],
        executable.memory_plan, artifact,
        executable.find_artifact_payload(artifact.artifact_key)));
  }
  return descriptor;
}

} // namespace gfx_plugin
} // namespace ov
