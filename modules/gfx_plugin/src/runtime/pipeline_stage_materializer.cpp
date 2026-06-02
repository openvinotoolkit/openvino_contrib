// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/pipeline_stage_materializer.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "common/gpu_backend.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/fused_output_lifetime_plan.hpp"
#include "runtime/fused_sequence_stage.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

RuntimeStageExecutableDescriptor make_vendor_attention_stage_descriptor(
    const RuntimeStageExecutableDescriptor &base_descriptor,
    const compiler::KernelArtifactDescriptor &artifact,
    std::shared_ptr<const compiler::KernelArtifactPayload> payload) {
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

} // namespace

PipelineStageMaterializer::PipelineStageMaterializer(
    const BackendStageFactory &stage_factory,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeExecutableDescriptor &runtime_descriptor,
    GpuStageRuntimeOptions runtime_options,
    compiler::PipelineVendorAttentionArtifactResolver
        vendor_attention_artifact_resolver)
    : m_stage_factory(stage_factory), m_runtime_descriptor(runtime_descriptor),
      m_runtime_options(std::move(runtime_options)),
      m_vendor_attention_artifact_resolver(
          std::move(vendor_attention_artifact_resolver)) {
  OPENVINO_ASSERT(
      runtime_descriptor.stages.size() == ordered_ops.size(),
      "GFX: runtime executable descriptor stage count drift: descriptor=",
      runtime_descriptor.stages.size(), " ordered_ops=", ordered_ops.size());

  m_descriptors_by_node.reserve(runtime_descriptor.stages.size());
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
    const auto inserted =
        m_descriptors_by_node.emplace(node.get(), &descriptor);
    OPENVINO_ASSERT(inserted.second,
                    "GFX: duplicate runtime executable descriptor for node ",
                    node->get_friendly_name(), " (", node->get_type_name(),
                    ")");
  }
}

const RuntimeStageExecutableDescriptor *
PipelineStageMaterializer::descriptor_for(
    const std::shared_ptr<const ov::Node> &node) const noexcept {
  if (!node) {
    return nullptr;
  }
  const auto it = m_descriptors_by_node.find(node.get());
  return it == m_descriptors_by_node.end() ? nullptr : it->second;
}

size_t PipelineStageMaterializer::stage_index_for(
    const std::shared_ptr<const ov::Node> &node) const {
  const auto *descriptor = descriptor_for(node);
  OPENVINO_ASSERT(descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for stage index of op ",
                  node ? node->get_friendly_name() : std::string("<null>"),
                  " (", node ? node->get_type_name() : "<null>", ")");
  return descriptor->stage_index;
}

compiler::PipelineStageFusionContract
PipelineStageMaterializer::fusion_contract_for(
    const std::shared_ptr<const ov::Node> &node) const {
  const auto *descriptor = descriptor_for(node);
  OPENVINO_ASSERT(descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for fusion contract of op ",
                  node ? node->get_friendly_name() : std::string("<null>"),
                  " (", node ? node->get_type_name() : "<null>", ")");
  compiler::PipelineStageFusionContract contract;
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

std::unique_ptr<GpuStage> PipelineStageMaterializer::create_stage(
    const std::shared_ptr<const ov::Node> &node) const {
  const auto *descriptor = descriptor_for(node);
  OPENVINO_ASSERT(descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for op ",
                  node ? node->get_friendly_name() : std::string("<null>"),
                  " (", node ? node->get_type_name() : "<null>", ")");
  auto stage = m_stage_factory.create_stage(node, descriptor);
  configure_stage(stage);
  return stage;
}

std::unique_ptr<GpuStage>
PipelineStageMaterializer::create_vendor_attention_stage(
    const compiler::PipelineVendorAttentionPlan &plan,
    const std::shared_ptr<const ov::Node> &final_node) const {
  const auto *base_descriptor = descriptor_for(final_node);
  OPENVINO_ASSERT(base_descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for vendor attention output op ",
                  final_node ? final_node->get_friendly_name()
                             : std::string("<null>"));
  OPENVINO_ASSERT(
      m_vendor_attention_artifact_resolver,
      "GFX: backend module does not expose compiler-owned vendor attention "
      "artifact resolver for ",
      backend_to_string(m_stage_factory.backend()));

  auto artifact =
      m_vendor_attention_artifact_resolver(base_descriptor->stage_record_key,
                                           plan);
  OPENVINO_ASSERT(artifact.valid(),
                  "GFX: backend module failed to materialize vendor attention "
                  "artifact for ",
                  plan.name);
  OPENVINO_ASSERT(
      artifact.descriptor.stage_record_key == base_descriptor->stage_record_key,
      "GFX: vendor attention artifact stage key drift for ", plan.name);
  RuntimeStageExecutableDescriptor descriptor =
      make_vendor_attention_stage_descriptor(*base_descriptor,
                                             artifact.descriptor,
                                             std::move(artifact.payload));

  auto stage = m_stage_factory.create_stage(final_node, &descriptor);
  configure_stage(stage);
  return stage;
}

std::optional<MaterializedFusedSequenceStage>
PipelineStageMaterializer::create_attention_sequence_stage(
    const FusionGroup &group,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const compiler::PipelineStagePlanBuilder &stage_plan_builder,
    const compiler::PipelineOutputAliasMap &output_aliases) const {
  const size_t stage_count = group.node_indices.size();
  if (stage_count < 3) {
    return std::nullopt;
  }

  std::unordered_map<const ov::Node *, size_t> stage_index;
  stage_index.reserve(stage_count);
  for (size_t i = 0; i < stage_count; ++i) {
    const auto idx = group.node_indices[i];
    if (idx >= ordered_ops.size()) {
      return std::nullopt;
    }
    stage_index[ordered_ops[idx].get()] = i;
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

  std::unordered_map<compiler::PipelineOutputPortKey, size_t,
                     compiler::PipelineOutputPortKeyHash>
      external_map;
  std::vector<compiler::PipelineStageInputLink> fused_inputs;
  std::vector<FusedStageInfo> fused_stages;
  std::vector<FusedOutputLifetimeStage> fused_lifetime_stages;
  fused_stages.reserve(stage_count);
  fused_lifetime_stages.reserve(stage_count);

  for (size_t i = 0; i < stage_count; ++i) {
    const size_t idx = group.node_indices[i];
    if (idx >= ordered_ops.size()) {
      return std::nullopt;
    }
    const auto &stage_node = ordered_ops[idx];
    auto stage = create_stage(stage_node);
    if (!stage) {
      return std::nullopt;
    }

    FusedStageInfo info;
    info.stage = std::move(stage);
    info.output_indices = stage_output_slots[i];
    info.inputs.reserve(stage_node->get_input_size());
    for (const auto &iv : stage_node->input_values()) {
      auto src_node = iv.get_node();
      const auto it_stage = stage_index.find(src_node);
      if (it_stage != stage_index.end()) {
        const size_t src_stage = it_stage->second;
        if (iv.get_index() >= stage_output_slots[src_stage].size()) {
          return std::nullopt;
        }
        info.inputs.push_back({FusedInputKind::Output,
                               stage_output_slots[src_stage][iv.get_index()]});
        continue;
      }
      if (ov::as_type_ptr<const ov::op::v0::Constant>(
              iv.get_node_shared_ptr())) {
        info.inputs.push_back({FusedInputKind::None, 0});
        continue;
      }
      size_t linked_port = iv.get_index();
      const auto remapped_input = stage_plan_builder.remap_input_link(
          output_aliases, iv.get_node_shared_ptr(), linked_port);
      linked_port = remapped_input.port;
      compiler::PipelineOutputPortKey key{src_node, linked_port};
      auto it_ext = external_map.find(key);
      size_t ext_idx = 0;
      if (it_ext == external_map.end()) {
        ext_idx = fused_inputs.size();
        external_map.emplace(key, ext_idx);
        fused_inputs.push_back(std::move(remapped_input));
      } else {
        ext_idx = it_ext->second;
      }
      info.inputs.push_back({FusedInputKind::External, ext_idx});
    }

    FusedOutputLifetimeStage lifetime_stage;
    lifetime_stage.output_indices = info.output_indices;
    lifetime_stage.descriptor = descriptor_for(stage_node);
    lifetime_stage.inputs.reserve(info.inputs.size());
    for (const auto &input : info.inputs) {
      FusedOutputLifetimeInputRef lifetime_input;
      lifetime_input.index = input.index;
      switch (input.kind) {
      case FusedInputKind::External:
        lifetime_input.kind = FusedOutputLifetimeInputRef::Kind::External;
        break;
      case FusedInputKind::Output:
        lifetime_input.kind = FusedOutputLifetimeInputRef::Kind::Output;
        break;
      case FusedInputKind::None:
      default:
        lifetime_input.kind = FusedOutputLifetimeInputRef::Kind::None;
        break;
      }
      lifetime_stage.inputs.push_back(lifetime_input);
    }
    fused_lifetime_stages.push_back(std::move(lifetime_stage));
    fused_stages.emplace_back(std::move(info));
  }

  if (fused_stages.size() != stage_count) {
    return std::nullopt;
  }

  const auto &final_node = ordered_ops[group.node_indices.back()];
  MaterializedFusedSequenceStage result;
  result.io_plan = stage_plan_builder.make_fused_stage_plan(
      final_node, fused_output_count, stage_index_for(final_node));
  result.output_lifetimes = build_fused_output_lifetime_plan(
      fused_lifetime_stages, m_runtime_descriptor.memory_plan,
      fused_output_count);
  result.io_plan.inputs = std::move(fused_inputs);

  for (size_t stage_idx = 0; stage_idx < stage_count; ++stage_idx) {
    const size_t node_idx = group.node_indices[stage_idx];
    if (node_idx >= ordered_ops.size()) {
      continue;
    }
    const auto &out_node = ordered_ops[node_idx];
    for (size_t port = 0; port < out_node->get_output_size(); ++port) {
      const size_t slot = stage_output_slots[stage_idx][port];
      stage_plan_builder.describe_output(result.io_plan, slot, out_node, port);
      stage_plan_builder.append_output_alias(result.io_plan, out_node, port,
                                             slot);
    }
  }

  result.stage = std::make_unique<FusedSequenceStage>(
      std::move(fused_stages),
      final_node ? final_node->get_friendly_name()
                 : std::string("fused_attention"),
      "FusedAttention");
  result.materialized_stage_count = stage_count;
  return result;
}

void PipelineStageMaterializer::configure_stage(
    const std::unique_ptr<GpuStage> &stage) const {
  if (stage) {
    stage->set_runtime_options(m_runtime_options);
  }
}

} // namespace gfx_plugin
} // namespace ov
