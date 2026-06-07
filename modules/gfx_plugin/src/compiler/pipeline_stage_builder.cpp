// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/pipeline_stage_runtime_descriptor_builder_detail.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "compiler/backend_config.hpp"
#include "compiler/backend_registry.hpp"
#include "compiler/stage_compiler_policy.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

using RuntimeStageDescriptorMap =
    std::unordered_map<const ov::Node *,
                       const RuntimeStageExecutableDescriptor *>;

struct RuntimeDescriptorMaterializationDraft {
  std::vector<::ov::gfx_plugin::PipelineStageMaterializationPlan> stage_plans;
  std::vector<::ov::gfx_plugin::PipelineStagePublicOutputDesc> public_outputs;
  ::ov::gfx_plugin::PipelineStageRuntimeOptionsPlan runtime_options;
};

RuntimeStageDescriptorMap build_runtime_stage_descriptor_map(
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeExecutableDescriptor &runtime_descriptor) {
  OPENVINO_ASSERT(
      runtime_descriptor.stages.size() == ordered_ops.size(),
      "GFX: runtime executable descriptor stage count drift: descriptor=",
      runtime_descriptor.stages.size(), " ordered_ops=", ordered_ops.size());

  RuntimeStageDescriptorMap descriptors;
  descriptors.reserve(runtime_descriptor.stages.size());
  for (size_t i = 0; i < runtime_descriptor.stages.size(); ++i) {
    const auto &node = ordered_ops[i];
    const auto &descriptor = runtime_descriptor.stages[i];
    OPENVINO_ASSERT(node, "GFX: runtime model ordered op ", i, " is null");
    OPENVINO_ASSERT(
        descriptor.stage_index == i,
        "GFX: runtime executable descriptor stage index drift at ordered op ",
        i);
    OPENVINO_ASSERT(
        descriptor.op_family == node->get_type_name(),
        "GFX: runtime executable descriptor op-family drift at ordered op ", i,
        ": descriptor=", descriptor.op_family, " node=", node->get_type_name());
    const auto inserted = descriptors.emplace(node.get(), &descriptor);
    OPENVINO_ASSERT(inserted.second,
                    "GFX: duplicate runtime executable descriptor for node ",
                    node->get_friendly_name(), " (", node->get_type_name(),
                    ")");
  }
  return descriptors;
}

const RuntimeStageExecutableDescriptor *descriptor_for_node(
    const RuntimeStageDescriptorMap &descriptors,
    const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return nullptr;
  }
  const auto it = descriptors.find(node.get());
  return it == descriptors.end() ? nullptr : it->second;
}

size_t stage_index_for_node(const RuntimeStageDescriptorMap &descriptors,
                            const std::shared_ptr<const ov::Node> &node) {
  const auto *descriptor = descriptor_for_node(descriptors, node);
  OPENVINO_ASSERT(descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for stage index of op ",
                  node ? node->get_friendly_name() : std::string("<null>"),
                  " (", node ? node->get_type_name() : "<null>", ")");
  return descriptor->stage_index;
}

PipelineStageFusionContract fusion_contract_for_node(
    const RuntimeStageDescriptorMap &descriptors,
    const std::shared_ptr<const ov::Node> &node) {
  const auto *descriptor = descriptor_for_node(descriptors, node);
  OPENVINO_ASSERT(descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for fusion contract of op ",
                  node ? node->get_friendly_name() : std::string("<null>"),
                  " (", node ? node->get_type_name() : "<null>", ")");
  PipelineStageFusionContract contract;
  contract.op_family = !descriptor->op_family.empty()
                           ? descriptor->op_family
                           : (node ? node->get_type_name() : std::string{});
  contract.origin = descriptor->origin;
  contract.payload_kind = descriptor->payload_kind;
  if (node && node->get_output_size() > 0) {
    contract.element_type = node->get_output_element_type(0);
  }
  return contract;
}

RuntimeStageExecutableDescriptor make_runtime_vendor_attention_descriptor(
    const RuntimeStageExecutableDescriptor &base_descriptor,
    const KernelArtifactDescriptor &artifact,
    std::shared_ptr<const KernelArtifactPayload> payload) {
  RuntimeStageExecutableDescriptor descriptor = base_descriptor;
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
  descriptor.runtime_shape_rule = artifact.kernel.runtime_shape_rule;
  descriptor.runtime_shape_i64_metadata =
      artifact.kernel.runtime_shape_i64_metadata;
  descriptor.requires_runtime_shape_args =
      artifact.kernel.requires_runtime_shape_args;
  descriptor.tensor_view_only = false;
  descriptor.tensor_roles = artifact.kernel.tensor_roles;
  descriptor.scalar_roles = artifact.kernel.scalar_roles;
  descriptor.exception_ticket.clear();
  descriptor.exception_reason.clear();
  descriptor.exception_removal_condition.clear();
  descriptor.optional_cache_payload_allowed =
      artifact.optional_cache_payload_allowed;
  descriptor.payload = std::move(payload);
  return descriptor;
}

bool runtime_binding_complete(const RuntimeTensorBindingContract &binding) {
  return !binding.logical_name.empty() && !binding.memory_region_id.empty() &&
         !binding.role.empty() && !binding.element_type.empty() &&
         !binding.partial_shape.empty() && !binding.layout.empty() &&
         !binding.storage_kind.empty() && !binding.lifetime_class.empty() &&
         !binding.alias_group.empty();
}

bool all_runtime_bindings_complete(
    const std::vector<RuntimeTensorBindingContract> &bindings) {
  return std::all_of(bindings.begin(), bindings.end(),
                     runtime_binding_complete);
}

std::string materialized_stage_name(const PipelineStageIoPlan &io_plan) {
  return io_plan.node ? io_plan.node->get_friendly_name()
                      : std::string("<null>");
}

void assert_complete_materialized_binding(
    const RuntimeTensorBindingContract &binding, std::string_view direction,
    const PipelineStageIoPlan &io_plan, size_t binding_idx) {
  OPENVINO_ASSERT(
      runtime_binding_complete(binding),
      "GFX: compiler failed to materialize ", direction,
      " binding contract for pipeline stage ", materialized_stage_name(io_plan),
      " at index ", binding_idx,
      "; runtime must receive final ABI, memory regions and tensor layout");
}

void refresh_runtime_tensor_roles(RuntimeStageExecutableDescriptor &descriptor) {
  descriptor.tensor_roles.clear();
  descriptor.tensor_roles.reserve(descriptor.input_bindings.size() +
                                  descriptor.output_bindings.size());
  for (const auto &binding : descriptor.input_bindings) {
    descriptor.tensor_roles.push_back(binding.role);
  }
  for (const auto &binding : descriptor.output_bindings) {
    descriptor.tensor_roles.push_back(binding.role);
  }
}

RuntimeStageExecutableDescriptor make_runtime_vendor_attention_descriptor(
    const PipelineStageMaterializationPlan &plan,
    const RuntimeStageDescriptorMap &descriptors) {
  const auto *base_descriptor =
      descriptor_for_node(descriptors, plan.io_plan.node);
  OPENVINO_ASSERT(base_descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for vendor attention stage ",
                  plan.vendor_attention.name);
  OPENVINO_ASSERT(plan.vendor_attention_artifact.valid(),
                  "GFX: compiler did not provide vendor attention artifact "
                  "for runtime plan ",
                  plan.vendor_attention.name);
  OPENVINO_ASSERT(
      plan.vendor_attention_artifact.descriptor.stage_record_key ==
          base_descriptor->stage_record_key,
      "GFX: vendor attention artifact stage key drift for ",
      plan.vendor_attention.name);
  return make_runtime_vendor_attention_descriptor(
      *base_descriptor, plan.vendor_attention_artifact.descriptor,
      plan.vendor_attention_artifact.payload);
}

void materialize_descriptor_input_bindings_from_io_links(
    RuntimeStageExecutableDescriptor &descriptor,
    const PipelineStageMaterializationPlan &plan,
    const RuntimeStageDescriptorMap &descriptors) {
  if (descriptor.input_bindings.size() == plan.io_plan.inputs.size() &&
      all_runtime_bindings_complete(descriptor.input_bindings)) {
    return;
  }

  descriptor.input_bindings.assign(plan.io_plan.inputs.size(), {});
  for (size_t input_idx = 0; input_idx < plan.io_plan.inputs.size();
       ++input_idx) {
    const auto &input = plan.io_plan.inputs[input_idx];
    const auto *source_descriptor =
        descriptor_for_node(descriptors, input.node);
    if (!source_descriptor ||
        input.port >= source_descriptor->output_bindings.size()) {
      continue;
    }
    const auto &source_binding =
        source_descriptor->output_bindings[input.port];
    if (runtime_binding_complete(source_binding)) {
      descriptor.input_bindings[input_idx] = source_binding;
    }
  }
}

void materialize_descriptor_input_bindings_from_fused_children(
    RuntimeStageExecutableDescriptor &descriptor,
    const PipelineStageMaterializationPlan &plan,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeStageDescriptorMap &descriptors) {
  if (descriptor.input_bindings.size() != plan.io_plan.inputs.size()) {
    descriptor.input_bindings.assign(plan.io_plan.inputs.size(), {});
  }

  for (size_t fused_idx = 0;
       fused_idx < plan.fusion_group.node_indices.size() &&
       fused_idx < plan.fused_inner_stages.size();
       ++fused_idx) {
    const size_t node_idx = plan.fusion_group.node_indices[fused_idx];
    if (node_idx >= ordered_ops.size()) {
      continue;
    }
    const auto *child_descriptor =
        descriptor_for_node(descriptors, ordered_ops[node_idx]);
    if (!child_descriptor) {
      continue;
    }
    const auto &inner_stage = plan.fused_inner_stages[fused_idx];
    for (size_t input_idx = 0; input_idx < inner_stage.inputs.size();
         ++input_idx) {
      const auto &input_plan = inner_stage.inputs[input_idx];
      if (input_plan.kind != PipelineFusedInputPlan::Kind::External ||
          input_plan.index >= descriptor.input_bindings.size() ||
          input_idx >= child_descriptor->input_bindings.size()) {
        continue;
      }
      const auto &child_binding = child_descriptor->input_bindings[input_idx];
      if (runtime_binding_complete(child_binding) &&
          !runtime_binding_complete(
              descriptor.input_bindings[input_plan.index])) {
        descriptor.input_bindings[input_plan.index] = child_binding;
      }
    }
  }
}

void materialize_descriptor_output_bindings(
    RuntimeStageExecutableDescriptor &descriptor,
    const PipelineStageMaterializationPlan &plan,
    const RuntimeStageDescriptorMap &descriptors) {
  if (descriptor.output_bindings.size() == plan.io_plan.outputs.size() &&
      all_runtime_bindings_complete(descriptor.output_bindings)) {
    return;
  }

  descriptor.output_bindings.assign(plan.io_plan.outputs.size(), {});
  for (size_t output_idx = 0; output_idx < plan.io_plan.outputs.size();
       ++output_idx) {
    const auto &output = plan.io_plan.outputs[output_idx];
    const auto *source_descriptor =
        descriptor_for_node(descriptors, output.source_node);
    if (!source_descriptor ||
        output.source_port >= source_descriptor->output_bindings.size()) {
      continue;
    }
    const auto &source_binding =
        source_descriptor->output_bindings[output.source_port];
    if (runtime_binding_complete(source_binding)) {
      descriptor.output_bindings[output_idx] = source_binding;
    }
  }
}

void finalize_materialized_descriptor_io(
    RuntimeStageExecutableDescriptor &descriptor,
    const PipelineStageMaterializationPlan &plan,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeStageDescriptorMap &descriptors) {
  if (plan.kind == PipelineStageMaterializationKind::FusedAttentionSequence) {
    materialize_descriptor_input_bindings_from_fused_children(
        descriptor, plan, ordered_ops, descriptors);
  } else {
    materialize_descriptor_input_bindings_from_io_links(descriptor, plan,
                                                       descriptors);
  }

  for (size_t input_idx = 0; input_idx < descriptor.input_bindings.size();
       ++input_idx) {
    assert_complete_materialized_binding(descriptor.input_bindings[input_idx],
                                         "input", plan.io_plan, input_idx);
  }

  materialize_descriptor_output_bindings(descriptor, plan, descriptors);
  for (size_t output_idx = 0; output_idx < descriptor.output_bindings.size();
       ++output_idx) {
    assert_complete_materialized_binding(descriptor.output_bindings[output_idx],
                                         "output", plan.io_plan, output_idx);
  }

  descriptor.abi_arg_count =
      static_cast<uint32_t>(descriptor.input_bindings.size());
  descriptor.abi_output_arg_count =
      static_cast<uint32_t>(descriptor.output_bindings.size());
  refresh_runtime_tensor_roles(descriptor);
}

void finalize_fused_attention_sequence_descriptor(
    RuntimeStageExecutableDescriptor &descriptor,
    const PipelineStageMaterializationPlan &plan,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeStageDescriptorMap &descriptors) {
  uint32_t stage_weight = 0;
  uint64_t macs_estimate = 0;
  bool dependency_boundary = false;
  for (const auto node_idx : plan.fusion_group.node_indices) {
    if (node_idx >= ordered_ops.size()) {
      continue;
    }
    const auto *child_descriptor =
        descriptor_for_node(descriptors, ordered_ops[node_idx]);
    if (!child_descriptor) {
      continue;
    }
    stage_weight +=
        std::max<uint32_t>(child_descriptor->submission_stage_weight, 1u);
    macs_estimate += child_descriptor->submission_macs_estimate;
    dependency_boundary =
        dependency_boundary || child_descriptor->submission_dependency_boundary;
  }

  descriptor.op_family = "FusedAttentionSequence";
  descriptor.origin = KernelArtifactOrigin::Common;
  descriptor.payload_kind = KernelArtifactPayloadKind::None;
  descriptor.entry_point.clear();
  descriptor.compile_options_key.clear();
  descriptor.runtime_shape_rule = "static_or_descriptor";
  descriptor.requires_runtime_shape_args = false;
  descriptor.tensor_view_only = false;
  descriptor.scalar_roles.clear();
  descriptor.optional_cache_payload_allowed = false;
  descriptor.payload.reset();
  descriptor.submission_stage_weight = std::max<uint32_t>(stage_weight, 1u);
  descriptor.submission_macs_estimate = macs_estimate;
  descriptor.submission_dependency_boundary = dependency_boundary;
  descriptor.kernel_id =
      "materialized/fused_attention_sequence/" + descriptor.kernel_id;
  descriptor.artifact_key =
      "materialized/fused_attention_sequence/" + descriptor.artifact_key;
  descriptor.manifest_ref =
      descriptor.manifest_ref + "/materialized_fused_attention_sequence";
}

RuntimeStageExecutableDescriptor make_materialized_runtime_descriptor(
    const PipelineStageMaterializationPlan &plan,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeStageDescriptorMap &descriptors) {
  const auto *base_descriptor = descriptor_for_node(descriptors, plan.io_plan.node);
  OPENVINO_ASSERT(base_descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for materialized pipeline stage ",
                  materialized_stage_name(plan.io_plan));

  RuntimeStageExecutableDescriptor descriptor =
      plan.kind == PipelineStageMaterializationKind::VendorAttention
          ? make_runtime_vendor_attention_descriptor(plan, descriptors)
          : *base_descriptor;

  if (plan.kind == PipelineStageMaterializationKind::SingleStage) {
    return descriptor;
  }

  finalize_materialized_descriptor_io(descriptor, plan, ordered_ops,
                                      descriptors);

  if (plan.kind == PipelineStageMaterializationKind::FusedAttentionSequence) {
    finalize_fused_attention_sequence_descriptor(descriptor, plan, ordered_ops,
                                                descriptors);
  }

  descriptor.abi_fingerprint =
      descriptor.abi_fingerprint + "/materialized_io/" +
      std::to_string(descriptor.input_bindings.size()) + "i/" +
      std::to_string(descriptor.output_bindings.size()) + "o";
  return descriptor;
}

using RuntimeTensorRefMap =
    std::unordered_map<PipelineOutputPortKey,
                       ::ov::gfx_plugin::PipelineStageTensorRef,
                       PipelineOutputPortKeyHash>;

::ov::gfx_plugin::PipelineStageTensorRef make_runtime_tensor_ref(
    ::ov::gfx_plugin::PipelineStageTensorRefKind kind, size_t index,
    size_t port);

::ov::gfx_plugin::PipelineStageTensorRef resolve_runtime_tensor_ref(
    const RuntimeTensorRefMap &refs,
    const std::shared_ptr<const ov::Node> &node, size_t port);

::ov::gfx_plugin::PipelineStageInputLink
make_runtime_input_link(const PipelineStageInputLink &link,
                        ::ov::gfx_plugin::PipelineStageTensorRef source_ref) {
  ::ov::gfx_plugin::PipelineStageInputLink result;
  result.port = link.port;
  result.source_ref = source_ref;
  return result;
}

::ov::gfx_plugin::PipelineStageOutputAlias
make_runtime_output_alias(
    const PipelineStageOutputAlias &alias,
    ::ov::gfx_plugin::PipelineStageTensorRef source_ref) {
  ::ov::gfx_plugin::PipelineStageOutputAlias result;
  result.source_port = alias.source_port;
  result.output_port = alias.output_port;
  result.source_ref = source_ref;
  return result;
}

::ov::gfx_plugin::PipelineStageOutputDesc
make_runtime_output_desc(
    const PipelineStageOutputDesc &output,
    ::ov::gfx_plugin::PipelineStageTensorRef source_ref) {
  ::ov::gfx_plugin::PipelineStageOutputDesc result;
  result.shape = output.shape;
  result.type = output.type;
  result.is_model_output = output.is_model_output;
  result.source_port = output.source_port;
  result.source_ref = source_ref;
  result.direct_stateful_assign_variable_id =
      output.direct_stateful_assign_variable_id;
  return result;
}

::ov::gfx_plugin::PipelineStageIoPlan
make_runtime_io_plan(
    const PipelineStageIoPlan &plan, size_t runtime_plan_stage_index,
    const RuntimeTensorRefMap &refs) {
  ::ov::gfx_plugin::PipelineStageIoPlan result;
  if (plan.node) {
    result.stage_name = plan.node->get_friendly_name();
    result.op_family = plan.node->get_type_name();
  }
  result.runtime_stage_index = plan.runtime_stage_index;
  result.outputs.reserve(plan.outputs.size());
  for (size_t output_idx = 0; output_idx < plan.outputs.size(); ++output_idx) {
    const auto &output = plan.outputs[output_idx];
    auto source_ref =
        resolve_runtime_tensor_ref(refs, output.source_node, output.source_port);
    if (!source_ref.valid()) {
      source_ref = make_runtime_tensor_ref(
          ::ov::gfx_plugin::PipelineStageTensorRefKind::StageOutput,
          runtime_plan_stage_index, output_idx);
    }
    result.outputs.push_back(make_runtime_output_desc(output, source_ref));
  }
  result.inputs.reserve(plan.inputs.size());
  for (const auto &input : plan.inputs) {
    result.inputs.push_back(make_runtime_input_link(
        input, resolve_runtime_tensor_ref(refs, input.node, input.port)));
  }
  result.output_aliases.reserve(plan.output_aliases.size());
  for (const auto &alias : plan.output_aliases) {
    result.output_aliases.push_back(make_runtime_output_alias(
        alias,
        resolve_runtime_tensor_ref(refs, alias.node, alias.source_port)));
  }
  return result;
}

::ov::gfx_plugin::PipelineStageInputTransformPlan
make_runtime_input_transform(const PipelineInputTransformPlan &transform) {
  ::ov::gfx_plugin::PipelineStageInputTransformPlan result;
  result.source_shape = transform.source_shape;
  result.transpose_permutation = transform.transpose_permutation;
  return result;
}

::ov::gfx_plugin::PipelineStageInputTransformBinding
make_runtime_input_transform_binding(
    const PipelineStageInputTransformBinding &binding) {
  ::ov::gfx_plugin::PipelineStageInputTransformBinding result;
  result.input_idx = binding.input_idx;
  result.transform = make_runtime_input_transform(binding.transform);
  return result;
}

::ov::gfx_plugin::PipelineFusedInputPlan::Kind
make_runtime_fused_input_kind(PipelineFusedInputPlan::Kind kind) {
  switch (kind) {
  case PipelineFusedInputPlan::Kind::External:
    return ::ov::gfx_plugin::PipelineFusedInputPlan::Kind::External;
  case PipelineFusedInputPlan::Kind::Output:
    return ::ov::gfx_plugin::PipelineFusedInputPlan::Kind::Output;
  case PipelineFusedInputPlan::Kind::None:
  default:
    return ::ov::gfx_plugin::PipelineFusedInputPlan::Kind::None;
  }
}

::ov::gfx_plugin::PipelineFusedInnerStagePlan
make_runtime_fused_inner_stage_plan(
    const PipelineFusedInnerStagePlan &inner_stage) {
  ::ov::gfx_plugin::PipelineFusedInnerStagePlan result;
  result.output_indices = inner_stage.output_indices;
  result.inputs.reserve(inner_stage.inputs.size());
  for (const auto &input : inner_stage.inputs) {
    ::ov::gfx_plugin::PipelineFusedInputPlan runtime_input;
    runtime_input.kind = make_runtime_fused_input_kind(input.kind);
    runtime_input.index = input.index;
    result.inputs.push_back(runtime_input);
  }
  return result;
}

::ov::gfx_plugin::PipelineStagePostOpFusionPlan
make_runtime_post_op_plan(const PipelineStagePostOpFusionPlan &post_ops) {
  ::ov::gfx_plugin::PipelineStagePostOpFusionPlan result;
  if (post_ops.input_activation) {
    const auto &input_activation = *post_ops.input_activation;
    result.input_activation =
        ::ov::gfx_plugin::PipelineStageInputActivationFusionPlan{
            input_activation.input_idx, input_activation.kind,
            input_activation.alpha};
  }
  result.batchnorm = post_ops.batchnorm;
  result.bias = post_ops.bias;
  result.activation = post_ops.activation;
  result.activation_alpha = post_ops.activation_alpha;
  return result;
}

::ov::gfx_plugin::PipelineStageMaterializationKind
make_runtime_materialization_kind(PipelineStageMaterializationKind kind) {
  switch (kind) {
  case PipelineStageMaterializationKind::VendorAttention:
    return ::ov::gfx_plugin::PipelineStageMaterializationKind::VendorAttention;
  case PipelineStageMaterializationKind::FusedAttentionSequence:
    return ::ov::gfx_plugin::PipelineStageMaterializationKind::
        FusedAttentionSequence;
  case PipelineStageMaterializationKind::SingleStage:
  default:
    return ::ov::gfx_plugin::PipelineStageMaterializationKind::SingleStage;
  }
}

std::vector<size_t> make_runtime_fused_descriptor_stage_indices(
    const PipelineStageMaterializationPlan &plan,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeStageDescriptorMap &descriptors) {
  std::vector<size_t> result;
  result.reserve(plan.fusion_group.node_indices.size());
  for (const auto node_idx : plan.fusion_group.node_indices) {
    OPENVINO_ASSERT(node_idx < ordered_ops.size(),
                    "GFX: fused compiler stage references missing ordered op");
    result.push_back(stage_index_for_node(descriptors, ordered_ops[node_idx]));
  }
  return result;
}

::ov::gfx_plugin::PipelineVendorAttentionStagePlan
make_runtime_vendor_attention_plan(
    const PipelineStageMaterializationPlan &plan,
    const RuntimeStageDescriptorMap &descriptors) {
  ::ov::gfx_plugin::PipelineVendorAttentionStagePlan result;
  result.name = plan.vendor_attention.name;
  if (plan.kind != PipelineStageMaterializationKind::VendorAttention) {
    return result;
  }

  result.descriptor =
      make_runtime_vendor_attention_descriptor(plan, descriptors);
  return result;
}

::ov::gfx_plugin::PipelineStageMaterializationPlan
make_runtime_materialization_plan(
    const PipelineStageMaterializationPlan &plan,
    size_t runtime_plan_stage_index,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeStageDescriptorMap &descriptors,
    const RuntimeTensorRefMap &refs) {
  ::ov::gfx_plugin::PipelineStageMaterializationPlan result;
  result.kind = make_runtime_materialization_kind(plan.kind);
  result.io_plan =
      make_runtime_io_plan(plan.io_plan, runtime_plan_stage_index, refs);
  result.descriptor_stage_index = result.io_plan.runtime_stage_index;
  result.materialized_descriptor =
      make_materialized_runtime_descriptor(plan, ordered_ops, descriptors);
  result.materialized_descriptor_valid = true;
  result.vendor_attention = make_runtime_vendor_attention_plan(plan, descriptors);
  result.fused_node_indices = plan.fusion_group.node_indices;
  result.fused_descriptor_stage_indices =
      make_runtime_fused_descriptor_stage_indices(plan, ordered_ops,
                                                 descriptors);
  result.fused_inner_stages.reserve(plan.fused_inner_stages.size());
  for (const auto &inner_stage : plan.fused_inner_stages) {
    result.fused_inner_stages.push_back(
        make_runtime_fused_inner_stage_plan(inner_stage));
  }
  result.input_transforms.reserve(plan.input_transforms.size());
  for (const auto &input_transform : plan.input_transforms) {
    result.input_transforms.push_back(
        make_runtime_input_transform_binding(input_transform));
  }
  if (plan.residual_add) {
    result.residual_add =
        ::ov::gfx_plugin::PipelineStageResidualAddFusionPlan{};
  }
  result.post_ops = make_runtime_post_op_plan(plan.post_ops);
  return result;
}

::ov::gfx_plugin::PipelineStageTensorRef make_runtime_tensor_ref(
    ::ov::gfx_plugin::PipelineStageTensorRefKind kind, size_t index,
    size_t port) {
  ::ov::gfx_plugin::PipelineStageTensorRef ref;
  ref.kind = kind;
  ref.index = index;
  ref.port = port;
  return ref;
}

::ov::gfx_plugin::PipelineStageTensorRef make_runtime_tensor_ref(
    const ::ov::gfx_plugin::RuntimePublicOutputDescriptor &output) {
  ::ov::gfx_plugin::PipelineStageTensorRefKind kind =
      ::ov::gfx_plugin::PipelineStageTensorRefKind::None;
  switch (output.kind) {
  case ::ov::gfx_plugin::RuntimePublicOutputSourceKind::Parameter:
    kind = ::ov::gfx_plugin::PipelineStageTensorRefKind::Parameter;
    break;
  case ::ov::gfx_plugin::RuntimePublicOutputSourceKind::StageOutput:
    kind = ::ov::gfx_plugin::PipelineStageTensorRefKind::StageOutput;
    break;
  case ::ov::gfx_plugin::RuntimePublicOutputSourceKind::None:
  default:
    break;
  }
  return make_runtime_tensor_ref(kind, output.index, output.port);
}

size_t compact_stage_index_for_runtime_stage(
    const RuntimeDescriptorMaterializationDraft &draft,
    size_t runtime_stage_index) {
  for (size_t i = 0; i < draft.stage_plans.size(); ++i) {
    if (draft.stage_plans[i].io_plan.runtime_stage_index ==
        runtime_stage_index) {
      return i;
    }
  }
  return ::ov::gfx_plugin::PipelineStageTensorRef::npos;
}

void record_runtime_tensor_ref(RuntimeTensorRefMap &refs,
                               const std::shared_ptr<const ov::Node> &node,
                               size_t port,
                               ::ov::gfx_plugin::PipelineStageTensorRef ref) {
  if (!node || !ref.valid()) {
    return;
  }
  refs[{node.get(), port}] = ref;
}

::ov::gfx_plugin::PipelineStageTensorRef resolve_runtime_tensor_ref(
    const RuntimeTensorRefMap &refs,
    const std::shared_ptr<const ov::Node> &node, size_t port) {
  if (!node) {
    return {};
  }
  const auto it = refs.find({node.get(), port});
  return it == refs.end() ? ::ov::gfx_plugin::PipelineStageTensorRef{}
                          : it->second;
}

RuntimeTensorRefMap build_runtime_tensor_ref_map(
    const std::vector<PipelineStageMaterializationPlan> &stage_plans,
    const std::unordered_map<const ov::Node *, size_t> &param_index) {
  RuntimeTensorRefMap refs;
  refs.reserve(param_index.size() + stage_plans.size());

  for (const auto &entry : param_index) {
    const auto *node = entry.first;
    if (!node) {
      continue;
    }
    refs[{node, 0}] = make_runtime_tensor_ref(
        ::ov::gfx_plugin::PipelineStageTensorRefKind::Parameter, entry.second,
        0);
  }

  for (size_t stage_idx = 0; stage_idx < stage_plans.size(); ++stage_idx) {
    const auto &io_plan = stage_plans[stage_idx].io_plan;
    for (size_t output_idx = 0; output_idx < io_plan.outputs.size();
         ++output_idx) {
      auto ref = make_runtime_tensor_ref(
          ::ov::gfx_plugin::PipelineStageTensorRefKind::StageOutput, stage_idx,
          output_idx);
      const auto &output = io_plan.outputs[output_idx];
      record_runtime_tensor_ref(refs, io_plan.node, output_idx, ref);
      record_runtime_tensor_ref(refs, output.source_node, output.source_port,
                                ref);
    }
    for (auto &alias : io_plan.output_aliases) {
      const auto source_ref = make_runtime_tensor_ref(
          ::ov::gfx_plugin::PipelineStageTensorRefKind::StageOutput, stage_idx,
          alias.output_port);
      record_runtime_tensor_ref(refs, alias.node, alias.source_port,
                                source_ref);
      record_runtime_tensor_ref(refs, alias.node, alias.output_port,
                                source_ref);
    }
  }

  return refs;
}

void attach_runtime_public_output_refs(
    RuntimeDescriptorMaterializationDraft &draft,
    const std::vector<detail::PipelineStagePublicOutputSource> &public_outputs,
    const RuntimeTensorRefMap &refs,
    const RuntimeStageDescriptorMap &descriptors,
    const ::ov::gfx_plugin::RuntimeExecutableDescriptor *descriptor) {
  draft.public_outputs.clear();
  draft.public_outputs.reserve(public_outputs.size());

  for (size_t result_idx = 0; result_idx < public_outputs.size(); ++result_idx) {
    const auto &source_output = public_outputs[result_idx];
    ::ov::gfx_plugin::PipelineStagePublicOutputDesc public_output;
    const auto source_node = source_output.node;
    const size_t source_port = source_output.port;
    public_output.source_ref =
        resolve_runtime_tensor_ref(refs, source_node, source_port);
    if (!public_output.source_ref.valid()) {
      if (const auto *stage_descriptor =
              descriptor_for_node(descriptors, source_node)) {
        const auto compact_stage_index = compact_stage_index_for_runtime_stage(
            draft, stage_descriptor->stage_index);
        if (compact_stage_index !=
            ::ov::gfx_plugin::PipelineStageTensorRef::npos) {
          public_output.source_ref = make_runtime_tensor_ref(
              ::ov::gfx_plugin::PipelineStageTensorRefKind::StageOutput,
              compact_stage_index, source_port);
        }
      }
    }
    if (!public_output.source_ref.valid() && descriptor &&
        result_idx < descriptor->public_outputs.size()) {
      public_output.source_ref =
          make_runtime_tensor_ref(descriptor->public_outputs[result_idx]);
      public_output.shape = descriptor->public_outputs[result_idx].static_shape;
      public_output.type = descriptor->public_outputs[result_idx].static_type;
    }

    if (public_output.shape.empty()) {
      public_output.shape = source_output.shape;
    }
    if (public_output.type == ov::element::dynamic) {
      public_output.type = source_output.type;
    }
    OPENVINO_ASSERT(
        public_output.source_ref.valid(),
        "GFX: compiler failed to materialize public output binding for result ",
        draft.public_outputs.size(),
        " source=",
        source_node ? source_node->get_friendly_name() : std::string("<null>"),
        " (", source_node ? source_node->get_type_name() : "<null>",
        ") port=", source_port,
        " pipeline_stages=", draft.stage_plans.size(),
        " descriptor_public_outputs=",
        descriptor ? descriptor->public_outputs.size() : 0,
        "; runtime must not resolve public outputs from graph maps");
    draft.public_outputs.push_back(std::move(public_output));
  }
}

RuntimeDescriptorMaterializationDraft make_runtime_descriptor_materialization_draft(
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const std::vector<PipelineStageMaterializationPlan> &stage_plans,
    const std::vector<detail::PipelineStagePublicOutputSource> &public_outputs,
    const RuntimeStageDescriptorMap &descriptors,
    const StageCompilerPolicy &stage_compiler_policy,
    const std::unordered_map<const ov::Node *, size_t> &param_index,
    const ::ov::gfx_plugin::RuntimeExecutableDescriptor *runtime_descriptor) {
  const auto refs = build_runtime_tensor_ref_map(stage_plans, param_index);
  RuntimeDescriptorMaterializationDraft result;
  result.stage_plans.reserve(stage_plans.size());
  for (size_t stage_idx = 0; stage_idx < stage_plans.size(); ++stage_idx) {
    result.stage_plans.push_back(make_runtime_materialization_plan(
        stage_plans[stage_idx], stage_idx, ordered_ops, descriptors, refs));
  }
  result.runtime_options.custom_kernel_dispatch_enabled =
      stage_compiler_policy.custom_kernel_dispatch.enabled;
  result.runtime_options.custom_kernel_dispatch_profile =
      stage_compiler_policy.custom_kernel_dispatch.profile;
  attach_runtime_public_output_refs(result, public_outputs, refs, descriptors,
                                    runtime_descriptor);
  return result;
}

std::vector<RuntimePublicOutputDescriptor>
make_runtime_public_output_descriptors(
    const RuntimeDescriptorMaterializationDraft &draft) {
  std::vector<RuntimePublicOutputDescriptor> public_outputs;
  public_outputs.reserve(draft.public_outputs.size());
  for (const auto &output : draft.public_outputs) {
    RuntimePublicOutputDescriptor runtime_output;
    switch (output.source_ref.kind) {
    case PipelineStageTensorRefKind::Parameter:
      runtime_output.kind = RuntimePublicOutputSourceKind::Parameter;
      break;
    case PipelineStageTensorRefKind::StageOutput:
      runtime_output.kind = RuntimePublicOutputSourceKind::StageOutput;
      break;
    case PipelineStageTensorRefKind::None:
    default:
      runtime_output.kind = RuntimePublicOutputSourceKind::None;
      break;
    }
    runtime_output.index = output.source_ref.index;
    runtime_output.port = output.source_ref.port;
    runtime_output.static_shape = output.shape;
    runtime_output.static_type = output.type;
    public_outputs.push_back(std::move(runtime_output));
  }
  return public_outputs;
}

struct AttentionSequenceStagePlan {
  PipelineStageIoPlan io_plan;
  std::vector<PipelineFusedInnerStagePlan> inner_stages;
};

struct PipelineStageBuildState {
  std::vector<std::shared_ptr<ov::Node>> ordered_ops;
  std::vector<PipelineStageMaterializationPlan> stage_plans;
  RuntimeDescriptorMaterializationDraft materialization_draft;
  StageCompilerPolicy stage_compiler_policy;
  std::unordered_map<const ov::Node *, size_t> param_index;
};

std::optional<AttentionSequenceStagePlan> make_attention_sequence_stage_plan(
    const FusionGroup &group,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const PipelineStagePlanBuilder &stage_plan_builder,
    const PipelineOutputAliasMap &output_aliases,
    const RuntimeStageDescriptorMap &descriptors) {
  const size_t stage_count = group.node_indices.size();
  if (stage_count < 3) {
    return std::nullopt;
  }

  std::unordered_map<const ov::Node *, size_t> fused_stage_index;
  fused_stage_index.reserve(stage_count);
  for (size_t i = 0; i < stage_count; ++i) {
    const auto idx = group.node_indices[i];
    if (idx >= ordered_ops.size()) {
      return std::nullopt;
    }
    fused_stage_index[ordered_ops[idx].get()] = i;
  }

  std::vector<std::vector<size_t>> stage_output_slots(stage_count);
  size_t fused_output_count = 1;
  for (size_t i = 0; i < stage_count; ++i) {
    const size_t idx = group.node_indices[i];
    if (idx >= ordered_ops.size()) {
      return std::nullopt;
    }
    const auto &stage_node = ordered_ops[idx];
    stage_output_slots[i].reserve(stage_node->get_output_size());
    for (size_t port = 0; port < stage_node->get_output_size(); ++port) {
      if (i + 1 == stage_count && port == 0) {
        stage_output_slots[i].push_back(0);
      } else {
        stage_output_slots[i].push_back(fused_output_count++);
      }
    }
  }

  std::unordered_map<PipelineOutputPortKey, size_t, PipelineOutputPortKeyHash>
      external_map;
  std::vector<PipelineStageInputLink> fused_inputs;
  std::vector<PipelineFusedInnerStagePlan> inner_stages;
  inner_stages.reserve(stage_count);
  for (size_t i = 0; i < stage_count; ++i) {
    const size_t idx = group.node_indices[i];
    const auto &stage_node = ordered_ops[idx];
    PipelineFusedInnerStagePlan inner_stage;
    inner_stage.output_indices = stage_output_slots[i];
    inner_stage.inputs.reserve(stage_node->get_input_size());
    for (const auto &iv : stage_node->input_values()) {
      auto src_node = iv.get_node();
      const auto it_stage = fused_stage_index.find(src_node);
      if (it_stage != fused_stage_index.end()) {
        const size_t src_stage = it_stage->second;
        if (iv.get_index() >= stage_output_slots[src_stage].size()) {
          return std::nullopt;
        }
        inner_stage.inputs.push_back(
            {PipelineFusedInputPlan::Kind::Output,
             stage_output_slots[src_stage][iv.get_index()]});
        continue;
      }
      if (ov::as_type_ptr<const ov::op::v0::Constant>(
              iv.get_node_shared_ptr())) {
        inner_stage.inputs.push_back({PipelineFusedInputPlan::Kind::None, 0});
        continue;
      }

      auto remapped = stage_plan_builder.remap_input_link(
          output_aliases, iv.get_node_shared_ptr(), iv.get_index());
      PipelineOutputPortKey key{remapped.node.get(), remapped.port};
      if (external_map.find(key) != external_map.end()) {
        inner_stage.inputs.push_back(
            {PipelineFusedInputPlan::Kind::External, external_map.at(key)});
        continue;
      }
      const size_t ext_idx = fused_inputs.size();
      external_map.emplace(key, ext_idx);
      fused_inputs.push_back(std::move(remapped));
      inner_stage.inputs.push_back(
          {PipelineFusedInputPlan::Kind::External, ext_idx});
    }
    inner_stages.push_back(std::move(inner_stage));
  }

  const auto &final_node = ordered_ops[group.node_indices.back()];
  AttentionSequenceStagePlan result;
  result.io_plan = stage_plan_builder.make_fused_stage_plan(
      final_node, fused_output_count, stage_index_for_node(descriptors, final_node));
  result.io_plan.inputs = std::move(fused_inputs);
  result.inner_stages = std::move(inner_stages);

  for (size_t stage_idx = 0; stage_idx < stage_count; ++stage_idx) {
    const size_t node_idx = group.node_indices[stage_idx];
    const auto &out_node = ordered_ops[node_idx];
    for (size_t port = 0; port < out_node->get_output_size(); ++port) {
      const size_t slot = stage_output_slots[stage_idx][port];
      stage_plan_builder.describe_output(result.io_plan, slot, out_node, port);
      stage_plan_builder.append_output_alias(result.io_plan, out_node, port,
                                             slot);
    }
  }

  return result;
}

} // namespace

namespace detail {

PipelineStageGraphSnapshot
make_pipeline_stage_graph_snapshot(const std::shared_ptr<const ov::Model> &model,
                                   const FusionConfig &fusion_config) {
  OPENVINO_ASSERT(model,
                  "GFX: pipeline stage graph snapshot requires compiler model");
  PipelineStageGraphSnapshot snapshot;
  snapshot.ordered_ops = model->get_ordered_ops();
  snapshot.graph_op_count = model->get_ops().size();
  snapshot.model_outputs = collect_model_output_ports(*model);
  snapshot.param_index.reserve(model->inputs().size());
  for (size_t i = 0; i < model->inputs().size(); ++i) {
    snapshot.param_index[model->inputs()[i].get_node()] = i;
  }
  const auto &model_results = model->get_results();
  snapshot.public_outputs.reserve(model_results.size());
  for (const auto &model_result : model_results) {
    const auto source = model_result->input_value(0);
    PipelineStagePublicOutputSource public_output;
    public_output.node = source.get_node_shared_ptr();
    public_output.port = source.get_index();
    if (source.get_partial_shape().is_static()) {
      public_output.shape = source.get_shape();
    }
    public_output.type = source.get_element_type();
    snapshot.public_outputs.push_back(std::move(public_output));
  }
  snapshot.fusion_plan = build_fusion_plan(model, fusion_config);
  return snapshot;
}

FusionConfig make_pipeline_stage_fusion_config(
    const FusionCapabilities &fusion_capabilities, bool enable_fusion,
    bool debug_dump_ir) {
  FusionConfig fusion_config;
  fusion_config.enable_fusion = enable_fusion;
  fusion_config.debug_dump_ir = debug_dump_ir;
  fusion_config.enable_attention_fusion =
      fusion_capabilities.enable_generic_attention_fusion;
  fusion_config.enable_vendor_attention_fusion =
      fusion_capabilities.supports_vendor_attention_stage;
  fusion_config.enable_conv_activation_fusion =
      fusion_capabilities.enable_conv_activation_fusion;
  fusion_config.enable_conv_swish_fusion = true;
  return fusion_config;
}

RuntimeExecutableDescriptor build_pipeline_stage_runtime_descriptor(
    const PipelineStageBuildRequest &request) {
  OPENVINO_ASSERT(request.graph.valid(),
                  "GFX: pipeline stage builder requires compiler-owned graph "
                  "snapshot");
  OPENVINO_ASSERT(request.runtime_descriptor,
                  "GFX: pipeline stage builder requires compiler-owned runtime "
                  "executable descriptor");

  OPENVINO_ASSERT(request.target.backend() != GpuBackend::Unknown,
                  "GFX: pipeline stage builder requires concrete BackendTarget");
  OPENVINO_ASSERT(
      request.runtime_descriptor->target_fingerprint ==
          request.target.fingerprint(),
      "GFX: pipeline stage target mismatch: descriptor=",
      request.runtime_descriptor->target_fingerprint,
      " request=", request.target.fingerprint());
  OPENVINO_ASSERT(request.backend_registry,
                  "GFX: pipeline stage builder requires explicit BackendRegistry");
  const auto backend_module = request.backend_registry->resolve(request.target);
  OPENVINO_ASSERT(backend_module, "GFX: backend registry has no module for ",
                  request.target.debug_string());
  const auto &backend_capabilities = backend_module->capabilities();
  const auto &fusion_capabilities = backend_capabilities.fusion();
  const auto stage_compiler_policy =
      compiler::make_stage_compiler_policy_from_capabilities(
          backend_capabilities);

  const std::string backend_name =
      request.backend_name.empty() ? request.target.backend_id()
                                   : request.backend_name;
  gfx_log_info("StagePlan")
      << "Planning pipeline for backend=" << backend_name
      << " target=" << request.target.debug_string()
      << " ops=" << request.graph.graph_op_count;
  const auto build_start = request.compile_trace
                               ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  if (request.compile_trace) {
    request.compile_trace->set_counter(
        "compiler_graph_op_count",
        static_cast<uint64_t>(request.graph.graph_op_count));
  }

  PipelineStageBuildState result;
  result.stage_compiler_policy = stage_compiler_policy;
  result.param_index = request.graph.param_index;

  const auto &model_outputs = request.graph.model_outputs;
  const compiler::PipelineStagePlanBuilder stage_plan_builder(model_outputs);

  result.ordered_ops = request.graph.ordered_ops;
  const auto &ordered_ops = result.ordered_ops;
  const auto descriptors = build_runtime_stage_descriptor_map(
      ordered_ops, *request.runtime_descriptor);
  gfx_log_info("StagePlan") << "Ordered ops count=" << ordered_ops.size();
  result.stage_plans.reserve(ordered_ops.size());

  const bool has_unobserved_stage_edges = [&]() {
    for (const auto &node : ordered_ops) {
      if (!compiler::is_executable_stage_node(node)) {
        continue;
      }
      for (size_t port = 0; port < node->get_output_size(); ++port) {
        if (!compiler::is_model_output_port(model_outputs, node.get(), port) &&
            !node->output(port).get_target_inputs().empty()) {
          return true;
        }
      }
    }
    return false;
  }();

  compiler::PipelineFusionSelectionOptions fusion_options;
  fusion_options.enable_fusion =
      request.enable_fusion && has_unobserved_stage_edges;
  fusion_options.debug_dump_ir = gfx_log_debug_enabled();
  fusion_options.fusion_capabilities = fusion_capabilities;
  fusion_options.stage_compiler_policy = &stage_compiler_policy;
  fusion_options.fusion_contract_for =
      [&](const std::shared_ptr<const ov::Node> &stage_node) {
        return fusion_contract_for_node(descriptors, stage_node);
      };

  const auto fusion_selection = compiler::plan_pipeline_fusions(
      request.graph.fusion_plan, ordered_ops, model_outputs, fusion_options);
  const auto &planned_fused_indices = fusion_selection.planned_fused_indices;
  const auto &planned_fused_nodes = fusion_selection.planned_fused_nodes;
  const auto &fused_nodes = fusion_selection.fused_nodes;
  auto primary_fusion_group_for = [&](size_t primary_idx) {
    return compiler::primary_fusion_group_for(fusion_selection, primary_idx);
  };

  if (request.compile_trace) {
    request.compile_trace->set_counter(
        "fusion_group_count",
        static_cast<uint64_t>(fusion_selection.fusion_plan.groups.size()));
    if (fusion_selection.precision_sensitive_vendor_allow_count != 0) {
      request.compile_trace->increment_counter(
          "fusion_precision_sensitive_vendor_allow_count",
          static_cast<uint64_t>(
              fusion_selection.precision_sensitive_vendor_allow_count));
    }
    if (fusion_selection.precision_sensitive_arithmetic_skip_count != 0) {
      request.compile_trace->increment_counter(
          "fusion_precision_sensitive_arithmetic_skip_count",
          static_cast<uint64_t>(
              fusion_selection.precision_sensitive_arithmetic_skip_count));
    }
  }
  if (fusion_options.enable_fusion) {
    if (gfx_log_debug_enabled()) {
      for (const auto &group : fusion_selection.fusion_plan.groups) {
        std::string node_list;
        for (size_t i = 0; i < group.node_indices.size(); ++i) {
          const auto node_idx = group.node_indices[i];
          if (node_idx >= ordered_ops.size()) {
            continue;
          }
          const auto &fused_node = ordered_ops[node_idx];
          if (!node_list.empty()) {
            node_list += " | ";
          }
          node_list += "[" + std::to_string(node_idx) + "] " +
                       fused_node->get_friendly_name() + " (" +
                       fused_node->get_type_name() + ")";
        }
        gfx_log_debug("Fusion")
            << "group kind=" << group.kind
            << " size=" << group.node_indices.size() << " nodes=" << node_list;
      }
      gfx_log_debug("Fusion") << "Fusion enabled: groups="
                              << fusion_selection.fusion_plan.groups.size();
    }
  } else if (gfx_log_debug_enabled()) {
    gfx_log_debug("Fusion")
        << (request.enable_fusion
                ? "Fusion skipped: all stage edges are observable"
                : "Fusion disabled via GFX_ENABLE_FUSION");
  }

  std::unordered_map<const ov::Node *, std::shared_ptr<const ov::Node>>
      rms_residual_adds;
  std::unordered_set<const ov::Node *> rms_residual_add_nodes;
  for (const auto &node : ordered_ops) {
    auto add = compiler::find_rms_residual_add(node, model_outputs,
                                               planned_fused_nodes);
    if (!add) {
      continue;
    }
    if (!compiler::allow_stage_residual_add_fusion(
            fusion_contract_for_node(descriptors, node))) {
      continue;
    }
    rms_residual_adds.emplace(node.get(), add);
    rms_residual_add_nodes.insert(add.get());
  }
  if (request.compile_trace && !rms_residual_adds.empty()) {
    request.compile_trace->set_counter(
        "rms_residual_add_fusion_count",
        static_cast<uint64_t>(rms_residual_adds.size()));
  }
  if (gfx_log_debug_enabled() && !rms_residual_adds.empty()) {
    gfx_log_debug("Fusion")
        << "RMS residual Add fusion candidates=" << rms_residual_adds.size();
  }

  std::unordered_set<size_t> fused_indices;
  fused_indices.reserve(ordered_ops.size());
  compiler::PipelineOutputAliasMap fused_output_port_aliases;

  const auto absorbed_input_transform_plan =
      compiler::plan_absorbed_input_transforms(ordered_ops, model_outputs,
                                               fused_nodes);

  for (size_t op_index = 0; op_index < ordered_ops.size(); ++op_index) {
    const auto &node = ordered_ops[op_index];
    if (!compiler::is_executable_stage_node(node)) {
      continue;
    }

    if (absorbed_input_transform_plan.absorbed_nodes.count(node.get()) != 0) {
      continue;
    }

    if (rms_residual_add_nodes.count(node.get()) != 0) {
      continue;
    }

    if (fused_indices.count(op_index)) {
      continue;
    }
    const auto *primary_group = primary_fusion_group_for(op_index);
    if (planned_fused_indices.count(op_index) && primary_group == nullptr) {
      continue;
    }

    if (primary_group) {
      const auto *group = primary_group;
      if (group->kind == "VendorAttention") {
        auto plan =
            compiler::plan_vendor_attention_subgraph(*group, ordered_ops);
        const auto &final_node = group->node_indices.empty()
                                     ? std::shared_ptr<ov::Node>{}
                                     : ordered_ops[group->node_indices.back()];
        const auto *base_descriptor = descriptor_for_node(descriptors, final_node);
        auto artifact =
            plan && base_descriptor
                ? backend_module->materialize_vendor_attention_artifact(
                      base_descriptor->stage_record_key, *plan)
                : PipelineVendorAttentionArtifact{};
        if (plan && artifact.valid() && !group->node_indices.empty()) {
          PipelineStageMaterializationPlan stage_plan;
          stage_plan.kind = PipelineStageMaterializationKind::VendorAttention;
          stage_plan.vendor_attention = *plan;
          stage_plan.vendor_attention_artifact = std::move(artifact);
          stage_plan.fusion_group = *group;
          stage_plan.io_plan = stage_plan_builder.make_fused_stage_plan(
              final_node, 1, stage_index_for_node(descriptors, final_node));
          stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
              fused_output_port_aliases, plan->query.node, plan->query.port));
          stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
              fused_output_port_aliases, plan->key.node, plan->key.port));
          stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
              fused_output_port_aliases, plan->value.node, plan->value.port));
          stage_plan.io_plan.outputs[0].shape = plan->output_shape;
          stage_plan.io_plan.outputs[0].type = plan->element_type;
          stage_plan.io_plan.outputs[0].source_node = final_node;
          stage_plan.io_plan.outputs[0].source_port = 0;
          stage_plan_builder.merge_model_outputs(stage_plan.io_plan,
                                                 final_node.get());
          stage_plan_builder.append_output_alias(stage_plan.io_plan, final_node,
                                                 0, 0);
          stage_plan_builder.record_output_alias(fused_output_port_aliases,
                                                 final_node.get(), 0, 0);

          if (request.compile_trace) {
            request.compile_trace->increment_counter("fused_stage_count");
            request.compile_trace->increment_counter(
                "fused_node_count",
                static_cast<uint64_t>(group->node_indices.size()));
            request.compile_trace->increment_counter(
                "vendor_attention_stage_count");
          }

          result.stage_plans.emplace_back(std::move(stage_plan));
          for (const auto node_idx : group->node_indices) {
            if (node_idx < ordered_ops.size()) {
              fused_indices.insert(node_idx);
            }
          }
          continue;
        }
      }
      if (group->kind == "Attention" || group->kind == "AttentionScale" ||
          group->kind == "AttentionScaleMask" || group->kind == "NativeSDPA") {
        auto sequence_plan = make_attention_sequence_stage_plan(
            *group, ordered_ops, stage_plan_builder, fused_output_port_aliases,
            descriptors);
        if (sequence_plan) {
          PipelineStageMaterializationPlan stage_plan;
          stage_plan.kind =
              PipelineStageMaterializationKind::FusedAttentionSequence;
          stage_plan.fusion_group = *group;
          stage_plan.io_plan = std::move(sequence_plan->io_plan);
          stage_plan.fused_inner_stages =
              std::move(sequence_plan->inner_stages);

          result.stage_plans.emplace_back(std::move(stage_plan));
          for (const auto &alias : result.stage_plans.back().io_plan.output_aliases) {
            stage_plan_builder.record_output_alias(
                fused_output_port_aliases, alias.node.get(), alias.source_port,
                alias.output_port);
          }
          for (const auto node_idx : group->node_indices) {
            if (node_idx < ordered_ops.size()) {
              fused_indices.insert(node_idx);
            }
          }
          if (request.compile_trace) {
            request.compile_trace->increment_counter("fused_stage_count");
            request.compile_trace->increment_counter(
                "fused_node_count",
                static_cast<uint64_t>(group->node_indices.size()));
          }
          continue;
        }
      }
    }

    if (gfx_log_debug_enabled()) {
      gfx_log_debug("StagePlan")
          << "Planning stage for " << node->get_type_name()
          << " name=" << node->get_friendly_name();
    }
    if (primary_group &&
        compiler::pipeline_fusion_group_has_fp32_precision(*primary_group,
                                                           ordered_ops) &&
        !ov::fp16_compression_is_disabled(node)) {
      ov::disable_fp16_compression(node);
    }

    PipelineStageMaterializationPlan stage_plan;
    stage_plan.kind = PipelineStageMaterializationKind::SingleStage;
    stage_plan.io_plan = stage_plan_builder.make_stage_plan(
        node, stage_index_for_node(descriptors, node));
    const auto residual_it = rms_residual_adds.find(node.get());
    if (residual_it != rms_residual_adds.end() && residual_it->second) {
      stage_plan.residual_add =
          PipelineStageResidualAddFusionPlan{residual_it->second};
      const auto &add = residual_it->second;
      stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
          fused_output_port_aliases, add->input_value(0).get_node_shared_ptr(),
          add->input_value(0).get_index()));
      stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
          fused_output_port_aliases, node->input_value(1).get_node_shared_ptr(),
          node->input_value(1).get_index()));
      stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
          fused_output_port_aliases, add->input_value(1).get_node_shared_ptr(),
          add->input_value(1).get_index()));
    } else {
      const auto absorbed_it =
          absorbed_input_transform_plan.input_transforms.find(node.get());
      for (size_t input_idx = 0; input_idx < node->get_input_size();
           ++input_idx) {
        const auto &iv = node->input_value(input_idx);
        auto linked_node = iv.get_node_shared_ptr();
        size_t linked_port = iv.get_index();
        if (absorbed_it !=
            absorbed_input_transform_plan.input_transforms.end()) {
          auto transform_it = absorbed_it->second.find(input_idx);
          if (transform_it != absorbed_it->second.end()) {
            auto transpose =
                ov::as_type_ptr<const ov::op::v1::Transpose>(linked_node);
            OPENVINO_ASSERT(
                transpose,
                "GFX: absorbed transpose input is not a transpose for ",
                node->get_friendly_name());
            linked_node = transpose->input_value(0).get_node_shared_ptr();
            linked_port = transpose->input_value(0).get_index();
            stage_plan.input_transforms.push_back(
                PipelineStageInputTransformBinding{input_idx,
                                                   transform_it->second});
          }
        }
        stage_plan.io_plan.inputs.push_back(stage_plan_builder.remap_input_link(
            fused_output_port_aliases, linked_node, linked_port));
      }
    }

    const size_t idx = result.stage_plans.size();
    result.stage_plans.emplace_back(std::move(stage_plan));

    if (primary_group) {
      const auto *group = primary_group;
      auto &planned_stage = result.stage_plans[idx];
      auto &io_plan = planned_stage.io_plan;
      auto mark_fused = [&](size_t fused_idx,
                            bool aliases_stage_output) -> bool {
        if (fused_idx >= ordered_ops.size()) {
          return false;
        }
        fused_indices.insert(fused_idx);
        const auto &fused_node = ordered_ops[fused_idx];
        if (aliases_stage_output && fused_node->get_output_size() == 1) {
          stage_plan_builder.merge_model_outputs(io_plan, fused_node.get());
          stage_plan_builder.append_output_alias(io_plan, fused_node, 0, 0);
        }
        return true;
      };

      auto mark_fused_tail = [&](size_t start_idx) -> bool {
        for (size_t i = start_idx; i < group->node_indices.size(); ++i) {
          const bool aliases_stage_output =
              (i + 1 == group->node_indices.size());
          if (!mark_fused(group->node_indices[i], aliases_stage_output)) {
            return false;
          }
        }
        return true;
      };
      if (group->input_activation.has_value()) {
        const size_t input_idx = group->input_activation_input;
        bool input_activation_ok = false;
        const bool input_activation_exclusive =
            io_plan.node &&
            compiler::pipeline_input_activation_has_exclusive_consumer(
                *group, op_index, ordered_ops, model_outputs);
        if (input_activation_exclusive && group->node_indices.size() > 1 &&
            input_idx < io_plan.inputs.size()) {
          const size_t act_idx = group->node_indices[1];
          if (act_idx < ordered_ops.size()) {
            const auto &act_node = ordered_ops[act_idx];
            if (io_plan.inputs[input_idx].node.get() == act_node.get() &&
                act_node->get_input_size() == 1) {
              io_plan.inputs[input_idx].node =
                  act_node->input_value(0).get_node_shared_ptr();
              io_plan.inputs[input_idx].port =
                  act_node->input_value(0).get_index();
              planned_stage.post_ops.input_activation =
                  PipelineStageInputActivationFusionPlan{
                      input_idx, *group->input_activation,
                      group->input_activation_alpha};
              input_activation_ok = mark_fused(act_idx, false);
            }
          }
        }
        if (!input_activation_ok && gfx_log_debug_enabled()) {
          gfx_log_debug("Fusion")
              << "Failed input activation fusion planning for "
              << (node ? node->get_friendly_name() : std::string("<null>"));
        }
      }
      size_t next_post_op_idx = 1;
      bool post_ops_ok = true;
      if (compiler::pipeline_fusion_requires_batchnorm_payload(*group) &&
          !group->batchnorm.has_value()) {
        post_ops_ok = false;
      }
      if (compiler::pipeline_fusion_requires_bias_payload(*group) &&
          !group->bias.has_value()) {
        post_ops_ok = false;
      }
      std::vector<size_t> fused_post_ops;
      if (group->batchnorm.has_value()) {
        if (group->node_indices.size() <= next_post_op_idx) {
          post_ops_ok = false;
        } else {
          planned_stage.post_ops.batchnorm = *group->batchnorm;
          fused_post_ops.push_back(group->node_indices[next_post_op_idx]);
          ++next_post_op_idx;
        }
      }
      if (post_ops_ok && group->bias.has_value()) {
        if (group->node_indices.size() <= next_post_op_idx) {
          post_ops_ok = false;
        } else {
          planned_stage.post_ops.bias = *group->bias;
          fused_post_ops.push_back(group->node_indices[next_post_op_idx]);
          ++next_post_op_idx;
        }
      }
      bool activation_fused = false;
      if (post_ops_ok && group->activation.has_value() &&
          group->node_indices.size() > next_post_op_idx) {
        planned_stage.post_ops.activation = *group->activation;
        planned_stage.post_ops.activation_alpha = group->activation_alpha;
        activation_fused = mark_fused_tail(next_post_op_idx);
        post_ops_ok = activation_fused;
      }
      if (!post_ops_ok) {
        planned_stage.post_ops.batchnorm.reset();
        planned_stage.post_ops.bias.reset();
        planned_stage.post_ops.activation.reset();
      }
      for (size_t i = 0; i < fused_post_ops.size(); ++i) {
        const bool aliases_stage_output =
            !activation_fused && (i + 1 == fused_post_ops.size());
        post_ops_ok =
            mark_fused(fused_post_ops[i], aliases_stage_output) && post_ops_ok;
      }
    }
  }

  if (request.compile_trace) {
    request.compile_trace->set_counter(
        "pipeline_stage_count",
        static_cast<uint64_t>(result.stage_plans.size()));
    request.compile_trace->add_segment(
        "compile", "build_pipeline_stage_runtime_descriptor",
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - build_start)
                .count()));
  }
  gfx_log_info("StagePlan")
      << "Planned GFX " << backend_name << " pipeline with "
      << result.stage_plans.size() << " stages";
  result.materialization_draft = make_runtime_descriptor_materialization_draft(
      result.ordered_ops, result.stage_plans, request.graph.public_outputs,
      descriptors, result.stage_compiler_policy, result.param_index,
      request.runtime_descriptor);
  RuntimeExecutableDescriptor runtime_descriptor = *request.runtime_descriptor;
  runtime_descriptor.materialization_finalized = true;
  runtime_descriptor.materialization_stages =
      std::move(result.materialization_draft.stage_plans);
  runtime_descriptor.public_outputs =
      make_runtime_public_output_descriptors(result.materialization_draft);
  runtime_descriptor.runtime_options =
      std::move(result.materialization_draft.runtime_options);
  return runtime_descriptor;
}

} // namespace detail

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
