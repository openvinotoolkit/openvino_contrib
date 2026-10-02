// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/runtime_executable_descriptor_builder.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "compiler/pipeline_stage_runtime_descriptor_builder_detail.hpp"
#include "kernel_ir/gfx_runtime_shape_rule.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

constexpr uint32_t kRuntimeExecutableDescriptorSchemaVersion = 1;

bool source_payload_kind(KernelArtifactPayloadKind kind) noexcept {
  return runtime_descriptor_source_payload_kind(kind);
}

bool payload_kind_requires_materialized_payload(
    KernelArtifactPayloadKind kind) noexcept {
  return runtime_descriptor_payload_kind_requires_payload(kind);
}

struct LaunchPlanRoleSummary {
  size_t const_tensor_count = 0;
  size_t runtime_param_count = 0;
  size_t output_count = 0;
};

LaunchPlanRoleSummary
summarize_launch_plan_roles(const KernelLaunchPlanDescriptor &plan) {
  LaunchPlanRoleSummary summary;
  if (!plan.valid) {
    return summary;
  }
  for (const auto &role : plan.buffer_roles) {
    if (role == "const_tensor") {
      ++summary.const_tensor_count;
    } else if (role == "runtime_params") {
      ++summary.runtime_param_count;
    } else if (role == "tensor_output") {
      ++summary.output_count;
    }
  }
  return summary;
}

bool launch_plan_matches(const KernelLaunchPlanDescriptor &lhs,
                         const KernelLaunchPlanDescriptor &rhs) {
  return lhs.valid == rhs.valid && lhs.buffer_roles == rhs.buffer_roles &&
         lhs.direct_input_indices == rhs.direct_input_indices &&
         lhs.input_indices == rhs.input_indices &&
         lhs.input_arg_count == rhs.input_arg_count &&
         lhs.operand_kinds == rhs.operand_kinds &&
         lhs.operand_arg_indices == rhs.operand_arg_indices &&
         lhs.scalar_args == rhs.scalar_args &&
         lhs.scalar_arg_kinds == rhs.scalar_arg_kinds;
}

bool source_origin(KernelArtifactOrigin origin) noexcept {
  return origin == KernelArtifactOrigin::Generated ||
         origin == KernelArtifactOrigin::HandwrittenException;
}

bool const_tensor_payloads_match(
    const std::vector<KernelArtifactConstTensor> &lhs,
    const std::vector<KernelArtifactConstTensor> *rhs) {
  if (!rhs) {
    return lhs.empty();
  }
  if (lhs.size() != rhs->size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    const auto &a = lhs[i];
    const auto &b = (*rhs)[i];
    if (a.source_input_index != b.source_input_index ||
        a.logical_name != b.logical_name || a.element_type != b.element_type ||
        a.shape != b.shape || a.bytes != b.bytes) {
      return false;
    }
  }
  return true;
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

RuntimeTensorBindingContract
make_tensor_binding_contract(const compiler::TensorContract &contract,
                             const compiler::MemoryPlan &memory_plan) {
  RuntimeTensorBindingContract binding;
  binding.logical_name = contract.logical_name;
  binding.memory_region_id = contract.memory_region_id;
  binding.role =
      std::string(compiler::tensor_contract_role_to_string(contract.role));
  binding.element_type = contract.element_type;
  binding.partial_shape = contract.partial_shape;
  binding.layout = contract.layout;
  binding.storage_kind = contract.storage_kind;
  binding.lifetime_class = contract.lifetime_class;
  binding.stateful_prebind_variable_id = contract.stateful_prebind_variable_id;
  binding.stateful_prebind_shape_rule = contract.stateful_prebind_shape_rule;
  binding.stateful_prebind_shape_axis = contract.stateful_prebind_shape_axis;
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
    std::shared_ptr<const KernelArtifactPayload> payload,
    const std::vector<KernelArtifactConstTensor> *const_tensors) {
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
  descriptor.stage_name = manifest_stage.source_node_name;
  descriptor.origin = artifact.kernel.origin;
  descriptor.payload_kind = artifact.payload_kind;
  descriptor.entry_point = artifact.entry_point;
  descriptor.compile_options_key = artifact.compile_options_key;
  descriptor.abi_arg_count = artifact.abi_arg_count;
  descriptor.abi_output_arg_count = artifact.abi_output_arg_count;
  descriptor.dispatch_contract = artifact.kernel.dispatch_contract;
  descriptor.layout_contract = artifact.kernel.layout_contract;
  descriptor.runtime_shape_rule = artifact.kernel.runtime_shape_rule;
  descriptor.runtime_shape_i64_metadata =
      artifact.kernel.runtime_shape_i64_metadata;
  descriptor.requires_runtime_shape_args =
      artifact.kernel.requires_runtime_shape_args;
  descriptor.tensor_view_only = descriptor.layout_contract == "view_only";
  descriptor.submission_stage_weight = manifest_stage.submission.stage_weight;
  descriptor.submission_macs_estimate = manifest_stage.submission.macs_estimate;
  descriptor.submission_dependency_boundary =
      manifest_stage.submission.dependency_extension_boundary;
  descriptor.stateful_effect =
      std::string(compiler::stateful_effect_kind_to_string(
          manifest_stage.stateful_effect.kind));
  descriptor.stateful_variable_id = manifest_stage.stateful_effect.variable_id;
  descriptor.tensor_roles = artifact.kernel.tensor_roles;
  descriptor.scalar_roles = artifact.kernel.scalar_roles;
  descriptor.runtime_param_buffer_count = artifact.runtime_param_buffer_count;
  descriptor.runtime_param_payload_kind = artifact.runtime_param_payload_kind;
  descriptor.runtime_param_i64_metadata = artifact.runtime_param_i64_metadata;
  descriptor.runtime_param_reduce_keep_dims =
      artifact.runtime_param_reduce_keep_dims;
  descriptor.runtime_param_reduce_keep_dims_valid =
      artifact.runtime_param_reduce_keep_dims_valid;
  descriptor.launch_plan = artifact.launch_plan;
  descriptor.exception_ticket = artifact.kernel.exception_ticket;
  descriptor.exception_reason = artifact.kernel.exception_reason;
  descriptor.exception_removal_condition =
      artifact.kernel.exception_removal_condition;
  descriptor.optional_cache_payload_allowed =
      artifact.optional_cache_payload_allowed;
  if (const_tensors) {
    descriptor.const_tensors = *const_tensors;
  }
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
    descriptor.alias_groups.push_back(
        make_memory_alias_group_descriptor(group));
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
      binding.lifetime_class != contract.lifetime_class ||
      binding.stateful_prebind_variable_id !=
          contract.stateful_prebind_variable_id ||
      binding.stateful_prebind_shape_rule !=
          contract.stateful_prebind_shape_rule ||
      binding.stateful_prebind_shape_axis !=
          contract.stateful_prebind_shape_axis) {
    return false;
  }
  const auto *region =
      find_memory_region(memory_plan, contract.memory_region_id);
  if (!region) {
    return false;
  }
  return binding.alias_group == region->alias_group &&
         binding.external_binding == region->external_binding &&
         binding.host_visible == region->host_visible;
}

bool runtime_binding_complete(const RuntimeTensorBindingContract &binding) {
  return !binding.logical_name.empty() && !binding.memory_region_id.empty() &&
         !binding.role.empty() && !binding.element_type.empty() &&
         !binding.partial_shape.empty() && !binding.layout.empty() &&
         !binding.storage_kind.empty() && !binding.lifetime_class.empty() &&
         !binding.alias_group.empty();
}

bool runtime_stage_identity_complete(
    const RuntimeStageExecutableDescriptor &stage) {
  return stage.stage_record_key != 0 && !stage.manifest_ref.empty() &&
         !stage.abi_fingerprint.empty() && !stage.artifact_key.empty() &&
         !stage.backend_domain.empty() && !stage.kernel_id.empty() &&
         !stage.op_family.empty() && !stage.stage_name.empty() &&
         !stage.runtime_shape_rule.empty() &&
         stage.submission_stage_weight != 0u;
}

void append_materialized_binding_diagnostics(
    RuntimeExecutableDescriptorVerificationResult &result,
    const std::vector<RuntimeTensorBindingContract> &bindings,
    std::string_view direction, size_t materialized_stage_index) {
  for (size_t i = 0; i < bindings.size(); ++i) {
    if (!runtime_binding_complete(bindings[i])) {
      result.diagnostics.push_back(
          "runtime executable descriptor materialized " +
          std::string(direction) + " binding incomplete at " +
          std::to_string(materialized_stage_index) + ":" + std::to_string(i));
    }
  }
}

void append_materialized_descriptor_diagnostics(
    RuntimeExecutableDescriptorVerificationResult &result,
    const PipelineStageMaterializationPlan &materialized_stage,
    const RuntimeStageExecutableDescriptor &descriptor_stage,
    size_t materialized_stage_index) {
  if (!materialized_stage.materialized_descriptor_valid) {
    result.diagnostics.push_back(
        "runtime executable descriptor stage plan materialized descriptor "
        "missing at " +
        std::to_string(materialized_stage_index));
    return;
  }

  const auto &materialized = materialized_stage.materialized_descriptor;
  if (materialized.stage_record_key != descriptor_stage.stage_record_key ||
      materialized.stage_index != descriptor_stage.stage_index) {
    result.diagnostics.push_back(
        "runtime executable descriptor stage plan materialized descriptor "
        "stage identity drift at " +
        std::to_string(materialized_stage_index));
  }
  if (!runtime_stage_identity_complete(materialized)) {
    result.diagnostics.push_back(
        "runtime executable descriptor stage plan materialized descriptor "
        "identity incomplete at " +
        std::to_string(materialized_stage_index));
  }
  if (materialized_stage.kind !=
      PipelineStageMaterializationKind::SingleStage) {
    if (materialized.input_bindings.size() !=
        materialized_stage.io_plan.inputs.size()) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage plan materialized input "
          "binding count drift at " +
          std::to_string(materialized_stage_index));
    }
    if (materialized.output_bindings.size() !=
        materialized_stage.io_plan.outputs.size()) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage plan materialized output "
          "binding count drift at " +
          std::to_string(materialized_stage_index));
    }
  }

  if (materialized.launch_plan.valid) {
    const auto launch_roles = summarize_launch_plan_roles(
        materialized.launch_plan);
    if (materialized.launch_plan.buffer_roles.size() !=
        materialized.abi_arg_count) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage plan materialized "
          "launch-plan ABI count drift at " +
          std::to_string(materialized_stage_index));
    }
    if (launch_roles.output_count != materialized.abi_output_arg_count) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage plan materialized "
          "launch-plan output count drift at " +
          std::to_string(materialized_stage_index));
    }
  } else {
    if (materialized.abi_arg_count < materialized.input_bindings.size()) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage plan materialized tensor ABI "
          "count drift at " +
          std::to_string(materialized_stage_index));
    }
    if (materialized.abi_output_arg_count !=
        materialized.output_bindings.size()) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage plan materialized output ABI "
          "count drift at " +
          std::to_string(materialized_stage_index));
    }
  }

  append_materialized_binding_diagnostics(result, materialized.input_bindings,
                                          "input", materialized_stage_index);
  append_materialized_binding_diagnostics(result, materialized.output_bindings,
                                          "output", materialized_stage_index);

  if (materialized_stage.kind ==
      PipelineStageMaterializationKind::VendorAttention) {
    if (!materialized_stage.vendor_attention.valid()) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage plan vendor descriptor missing "
          "at " +
          std::to_string(materialized_stage_index));
    }
    if (materialized.payload_kind !=
            KernelArtifactPayloadKind::VendorDescriptor ||
        !materialized.payload) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage plan vendor materialized "
          "payload missing at " +
          std::to_string(materialized_stage_index));
    }
  }

  if (materialized_stage.kind ==
      PipelineStageMaterializationKind::FusedAttentionSequence) {
    if (materialized.origin != KernelArtifactOrigin::Common ||
        materialized.payload_kind != KernelArtifactPayloadKind::None ||
        materialized.payload) {
      result.diagnostics.push_back(
          "runtime executable descriptor stage plan fused materialized "
          "descriptor carries backend payload at " +
          std::to_string(materialized_stage_index));
    }
  }
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
    if (!memory_region_matches(descriptor.regions[i], memory_plan.regions[i])) {
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

void append_materialization_diagnostics(
    RuntimeExecutableDescriptorVerificationResult &result,
    const RuntimeExecutableDescriptor &descriptor) {
  for (size_t i = 0; i < descriptor.materialization_stages.size(); ++i) {
    const auto &materialized_stage = descriptor.materialization_stages[i];
    const auto runtime_stage_index =
        materialized_stage.io_plan.runtime_stage_index;
    if (materialized_stage.descriptor_stage_index != runtime_stage_index) {
      result.diagnostics.push_back("runtime executable descriptor stage plan "
                                   "descriptor index drift at " +
                                   std::to_string(i));
    }
    if (runtime_stage_index == PipelineStageIoPlan::npos ||
        runtime_stage_index >= descriptor.stages.size()) {
      result.diagnostics.push_back(
          "compiler-owned runtime descriptor materialized stage has "
          "invalid descriptor index at " +
          std::to_string(i));
      continue;
    }
    const auto &descriptor_stage = descriptor.stages[runtime_stage_index];
    if (descriptor_stage.op_family != materialized_stage.io_plan.op_family ||
        descriptor_stage.stage_name != materialized_stage.io_plan.stage_name) {
      result.diagnostics.push_back(
          "compiler-owned runtime descriptor materialized stage drift "
          "at " +
          std::to_string(i));
    }
    append_materialized_descriptor_diagnostics(result, materialized_stage,
                                               descriptor_stage, i);
    if (materialized_stage.kind ==
        PipelineStageMaterializationKind::FusedAttentionSequence) {
      const auto fused_stage_count =
          materialized_stage.fused_descriptor_stage_indices.size();
      if (fused_stage_count < 3 ||
          materialized_stage.fused_inner_stages.size() != fused_stage_count) {
        result.diagnostics.push_back(
            "compiler-owned runtime descriptor fused materialization stage "
            "count drift at " +
            std::to_string(i));
      }
      for (const auto fused_stage_index :
           materialized_stage.fused_descriptor_stage_indices) {
        if (fused_stage_index == PipelineStageIoPlan::npos ||
            fused_stage_index >= descriptor.stages.size()) {
          result.diagnostics.push_back(
              "compiler-owned runtime descriptor fused descriptor index "
              "out of range at " +
              std::to_string(i));
          break;
        }
        if (!runtime_stage_descriptor_is_materializable(
                descriptor.stages[fused_stage_index])) {
          result.diagnostics.push_back(
              "compiler-owned runtime descriptor fused child descriptor is "
              "not materializable at " +
              std::to_string(i) + ":" +
              std::to_string(fused_stage_index));
        }
      }
    }
  }

  for (size_t i = 0; i < descriptor.public_outputs.size(); ++i) {
    const auto &descriptor_output = descriptor.public_outputs[i];
    if (descriptor_output.kind == RuntimePublicOutputSourceKind::None ||
        descriptor_output.index == PipelineStageTensorRef::npos) {
      result.diagnostics.push_back(
          "runtime executable descriptor public output is incomplete at " +
          std::to_string(i));
      continue;
    }
    if (descriptor_output.kind == RuntimePublicOutputSourceKind::StageOutput &&
        descriptor_output.index >= descriptor.materialization_stages.size()) {
      result.diagnostics.push_back(
          "runtime executable descriptor public output stage index out of range at " +
          std::to_string(i));
    }
  }
}

} // namespace

namespace compiler {

RuntimeExecutableDescriptorVerificationResult
verify_runtime_executable_descriptor(
    const RuntimeExecutableDescriptor &descriptor,
    const ExecutableBundle &executable) {
  RuntimeExecutableDescriptorVerificationResult result;
  if (descriptor.schema_version != kRuntimeExecutableDescriptorSchemaVersion ||
      descriptor.target_fingerprint.empty()) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor header is incomplete");
  }
  if (descriptor.target_fingerprint != executable.target_fingerprint) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor target drift");
  }
  if (descriptor.stages.size() != executable.stages.size()) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor stage count drift");
  }
  append_memory_plan_diagnostics(result, descriptor.memory_plan,
                                 executable.memory_plan);

  std::unordered_set<uint64_t> stage_keys;
  const size_t count =
      std::min(descriptor.stages.size(), executable.stages.size());
  for (size_t i = 0; i < count; ++i) {
    const auto &stage = descriptor.stages[i];
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
        stage.stage_name != executable.manifest.stages[i].source_node_name ||
        stage.origin != artifact.kernel.origin ||
        stage.payload_kind != artifact.payload_kind ||
        stage.entry_point != artifact.entry_point ||
        stage.compile_options_key != artifact.compile_options_key ||
        stage.abi_arg_count != artifact.abi_arg_count ||
        stage.abi_output_arg_count != artifact.abi_output_arg_count ||
        stage.dispatch_contract != artifact.kernel.dispatch_contract ||
        stage.layout_contract != artifact.kernel.layout_contract ||
        stage.runtime_shape_rule != artifact.kernel.runtime_shape_rule ||
        stage.runtime_shape_i64_metadata !=
            artifact.kernel.runtime_shape_i64_metadata ||
        stage.requires_runtime_shape_args !=
            artifact.kernel.requires_runtime_shape_args ||
        stage.tensor_view_only !=
            (artifact.kernel.layout_contract == "view_only") ||
        stage.submission_stage_weight !=
            executable.manifest.stages[i].submission.stage_weight ||
        stage.submission_macs_estimate !=
            executable.manifest.stages[i].submission.macs_estimate ||
        stage.submission_dependency_boundary !=
            executable.manifest.stages[i]
                .submission.dependency_extension_boundary ||
        stage.stateful_effect !=
            compiler::stateful_effect_kind_to_string(
                executable.manifest.stages[i].stateful_effect.kind) ||
        stage.stateful_variable_id !=
            executable.manifest.stages[i].stateful_effect.variable_id ||
        stage.tensor_roles != artifact.kernel.tensor_roles ||
        stage.scalar_roles != artifact.kernel.scalar_roles ||
        stage.runtime_param_buffer_count !=
            artifact.runtime_param_buffer_count ||
        stage.runtime_param_payload_kind !=
            artifact.runtime_param_payload_kind ||
        stage.runtime_param_i64_metadata !=
            artifact.runtime_param_i64_metadata ||
        stage.runtime_param_reduce_keep_dims !=
            artifact.runtime_param_reduce_keep_dims ||
        stage.runtime_param_reduce_keep_dims_valid !=
            artifact.runtime_param_reduce_keep_dims_valid ||
        !launch_plan_matches(stage.launch_plan, artifact.launch_plan) ||
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
        stage.kernel_id.empty() || stage.op_family.empty() ||
        stage.stage_name.empty() || stage.runtime_shape_rule.empty() ||
        stage.submission_stage_weight == 0u) {
      result.diagnostics.push_back(
          "runtime executable descriptor has incomplete identity at " +
          std::to_string(i));
    }
    if (!descriptor_owns_runtime_shape_rule(stage.op_family,
                                            stage.runtime_shape_rule)) {
      result.diagnostics.push_back(
          "runtime executable descriptor runtime shape rule does not match op "
          "family at " +
          std::to_string(i));
    }
    if (stage.payload_kind == KernelArtifactPayloadKind::VendorDescriptor &&
        stage.origin != KernelArtifactOrigin::VendorPrimitive) {
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
    if (stage.origin == KernelArtifactOrigin::HandwrittenException &&
        (stage.exception_ticket.empty() || stage.exception_reason.empty() ||
         stage.exception_removal_condition.empty())) {
      result.diagnostics.push_back("runtime executable descriptor handwritten "
                                   "exception contract is incomplete at " +
                                   std::to_string(i));
    }
    if (stage.origin != KernelArtifactOrigin::HandwrittenException &&
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
    const auto executable_const_tensors =
        executable.find_artifact_const_tensors(stage.artifact_key);
    const auto launch_roles = summarize_launch_plan_roles(stage.launch_plan);
    if (source_payload_kind(stage.payload_kind)) {
      if (!stage.launch_plan.valid || stage.launch_plan.buffer_roles.empty()) {
        result.diagnostics.push_back(
            "runtime executable descriptor source launch plan is incomplete "
            "at " +
            std::to_string(i));
      }
      if (stage.launch_plan.buffer_roles.size() != stage.abi_arg_count) {
        result.diagnostics.push_back(
            "runtime executable descriptor source launch-plan ABI count drift "
            "at " +
            std::to_string(i));
      }
      if (launch_roles.output_count != stage.abi_output_arg_count) {
        result.diagnostics.push_back(
            "runtime executable descriptor source launch-plan output count "
            "drift at " +
            std::to_string(i));
      }
    }
    if (launch_roles.const_tensor_count != 0 &&
        stage.const_tensors.size() < launch_roles.const_tensor_count) {
      result.diagnostics.push_back(
          "runtime executable descriptor ConstTensor ABI is not "
          "descriptor-owned at " +
          std::to_string(i));
    }
    if (!const_tensor_payloads_match(stage.const_tensors,
                                     executable_const_tensors)) {
      result.diagnostics.push_back(
          "runtime executable descriptor const tensor payload drift at " +
          std::to_string(i));
    }
    if (artifact.runtime_param_buffer_count != 0 &&
        !descriptor_owns_runtime_param_payload(stage)) {
      result.diagnostics.push_back(
          "runtime executable descriptor RuntimeParams ABI is not "
          "descriptor-owned at " +
          std::to_string(i));
    }
    if (launch_roles.runtime_param_count != 0 &&
        artifact.runtime_param_buffer_count !=
            launch_roles.runtime_param_count) {
      result.diagnostics.push_back(
          "runtime executable descriptor RuntimeParams buffer count drift at " +
          std::to_string(i));
    }
    if (stage.runtime_param_i64_metadata !=
            artifact.runtime_param_i64_metadata ||
        stage.runtime_param_payload_kind !=
            artifact.runtime_param_payload_kind ||
        stage.runtime_param_reduce_keep_dims !=
            artifact.runtime_param_reduce_keep_dims ||
        stage.runtime_param_reduce_keep_dims_valid !=
            artifact.runtime_param_reduce_keep_dims_valid) {
      result.diagnostics.push_back(
          "runtime executable descriptor runtime-param metadata drift at " +
          std::to_string(i));
    }
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

bool runtime_executable_descriptor_valid(
    const RuntimeExecutableDescriptor &descriptor,
    const ExecutableBundle &executable) {
  return verify_runtime_executable_descriptor(descriptor, executable).valid();
}

RuntimeExecutableDescriptorVerificationResult
verify_runtime_executable_descriptor_materialization(
    const RuntimeExecutableDescriptor &descriptor) {
  RuntimeExecutableDescriptorVerificationResult result;
  if (!descriptor.materialization_finalized) {
    result.diagnostics.emplace_back(
        "runtime executable descriptor materialization contract is not "
        "finalized");
    return result;
  }
  append_materialization_diagnostics(result, descriptor);
  return result;
}

bool runtime_executable_descriptor_materialization_valid(
    const RuntimeExecutableDescriptor &descriptor) {
  return verify_runtime_executable_descriptor_materialization(descriptor)
      .valid();
}

RuntimeExecutableDescriptor RuntimeExecutableDescriptorBuilder::build(
    const ExecutableBundle &executable) const {
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
        executable.find_artifact_payload(artifact.artifact_key),
        executable.find_artifact_const_tensors(artifact.artifact_key)));
  }
  return descriptor;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
