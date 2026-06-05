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
#include "runtime/fused_output_lifetime_plan.hpp"
#include "runtime/fused_sequence_stage.hpp"
#include "runtime/gfx_compile_profiling.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void apply_pipeline_stage_io_plan(PipelineStageDesc &stage_desc,
                                  PipelineStageIoPlan stage_plan) {
  static_cast<PipelineStageIoPlan &>(stage_desc) = std::move(stage_plan);
}

void apply_input_transform_plan(
    GpuStage &stage,
    const std::vector<PipelineStageInputTransformBinding> &input_transforms) {
  for (const auto &binding : input_transforms) {
    GfxInputTransform transform;
    transform.source_shape = binding.transform.source_shape;
    transform.transpose_permutation = binding.transform.transpose_permutation;
    stage.set_input_transform(binding.input_idx, transform);
  }
}

void apply_single_stage_fusion_plan(
    GpuStage &stage, const PipelineStageMaterializationPlan &plan) {
  apply_input_transform_plan(stage, plan.input_transforms);

  if (plan.residual_add) {
    OPENVINO_ASSERT(stage.fuse_residual_add(),
                    "GFX: runtime stage rejected compiler-owned residual Add "
                    "fusion plan for ",
                    plan.io_plan.node
                        ? plan.io_plan.node->get_friendly_name()
                        : std::string("<null>"));
  }

  if (plan.post_ops.input_activation) {
    const auto &input_activation = *plan.post_ops.input_activation;
    OPENVINO_ASSERT(
        stage.fuse_input_activation(input_activation.input_idx,
                                    input_activation.kind,
                                    input_activation.alpha),
        "GFX: runtime stage rejected compiler-owned input activation fusion "
        "plan for ",
        plan.io_plan.node ? plan.io_plan.node->get_friendly_name()
                          : std::string("<null>"));
  }
  if (plan.post_ops.batchnorm) {
    OPENVINO_ASSERT(stage.fuse_batchnorm(*plan.post_ops.batchnorm),
                    "GFX: runtime stage rejected compiler-owned BatchNorm "
                    "fusion plan for ",
                    plan.io_plan.node
                        ? plan.io_plan.node->get_friendly_name()
                        : std::string("<null>"));
  }
  if (plan.post_ops.bias) {
    OPENVINO_ASSERT(stage.fuse_bias(*plan.post_ops.bias),
                    "GFX: runtime stage rejected compiler-owned bias fusion "
                    "plan for ",
                    plan.io_plan.node
                        ? plan.io_plan.node->get_friendly_name()
                        : std::string("<null>"));
  }
  if (plan.post_ops.activation) {
    OPENVINO_ASSERT(stage.fuse_activation(*plan.post_ops.activation,
                                          plan.post_ops.activation_alpha),
                    "GFX: runtime stage rejected compiler-owned activation "
                    "fusion plan for ",
                    plan.io_plan.node
                        ? plan.io_plan.node->get_friendly_name()
                        : std::string("<null>"));
  }
}

FusedInputKind
to_runtime_fused_input_kind(PipelineFusedInputPlan::Kind kind) {
  switch (kind) {
  case PipelineFusedInputPlan::Kind::External:
    return FusedInputKind::External;
  case PipelineFusedInputPlan::Kind::Output:
    return FusedInputKind::Output;
  case PipelineFusedInputPlan::Kind::None:
  default:
    return FusedInputKind::None;
  }
}

FusedOutputLifetimeInputRef::Kind to_lifetime_fused_input_kind(
    PipelineFusedInputPlan::Kind kind) {
  switch (kind) {
  case PipelineFusedInputPlan::Kind::External:
    return FusedOutputLifetimeInputRef::Kind::External;
  case PipelineFusedInputPlan::Kind::Output:
    return FusedOutputLifetimeInputRef::Kind::Output;
  case PipelineFusedInputPlan::Kind::None:
  default:
    return FusedOutputLifetimeInputRef::Kind::None;
  }
}

bool binding_complete(const RuntimeTensorBindingContract &binding) {
  return !binding.logical_name.empty() && !binding.memory_region_id.empty() &&
         !binding.role.empty() && !binding.element_type.empty() &&
         !binding.partial_shape.empty() && !binding.layout.empty() &&
         !binding.storage_kind.empty() && !binding.lifetime_class.empty() &&
         !binding.alias_group.empty();
}

std::string materialized_stage_name(const PipelineStageIoPlan &io_plan) {
  return io_plan.node ? io_plan.node->get_friendly_name()
                      : std::string("<null>");
}

void assert_complete_materialized_binding(
    const RuntimeTensorBindingContract &binding, std::string_view direction,
    const PipelineStageIoPlan &io_plan, size_t binding_idx) {
  OPENVINO_ASSERT(
      binding_complete(binding),
      "GFX: compiler-owned materialized ", direction,
      " binding contract is incomplete for pipeline stage ",
      materialized_stage_name(io_plan), " at index ", binding_idx,
      "; runtime must not synthesize ABI, memory regions or tensor layout");
}

void assert_materialized_descriptor_complete(
    const RuntimeStageExecutableDescriptor &descriptor,
    const PipelineStageMaterializationPlan &plan) {
  OPENVINO_ASSERT(!descriptor.manifest_ref.empty() &&
                      !descriptor.abi_fingerprint.empty() &&
                      !descriptor.artifact_key.empty() &&
                      !descriptor.backend_domain.empty() &&
                      !descriptor.kernel_id.empty() &&
                      !descriptor.op_family.empty() &&
                      !descriptor.runtime_shape_rule.empty() &&
      descriptor.stage_record_key != 0,
                  "GFX: compiler-owned materialized descriptor identity is "
                  "incomplete for pipeline stage ",
                  materialized_stage_name(plan.io_plan));
  OPENVINO_ASSERT(!descriptor.stage_name.empty(),
                  "GFX: compiler-owned materialized descriptor stage name is "
                  "incomplete for pipeline stage ",
                  materialized_stage_name(plan.io_plan));

  if (plan.kind != PipelineStageMaterializationKind::SingleStage) {
    OPENVINO_ASSERT(
        descriptor.input_bindings.size() == plan.io_plan.inputs.size(),
        "GFX: compiler-owned materialized input binding count drift for ",
        materialized_stage_name(plan.io_plan), ": descriptor=",
        descriptor.input_bindings.size(), " plan=", plan.io_plan.inputs.size());
    OPENVINO_ASSERT(
        descriptor.output_bindings.size() == plan.io_plan.outputs.size(),
        "GFX: compiler-owned materialized output binding count drift for ",
        materialized_stage_name(plan.io_plan), ": descriptor=",
        descriptor.output_bindings.size(), " plan=",
        plan.io_plan.outputs.size());
  }

  OPENVINO_ASSERT(
      descriptor.abi_arg_count == descriptor.input_bindings.size(),
      "GFX: compiler-owned materialized input ABI count drift for ",
      materialized_stage_name(plan.io_plan));
  OPENVINO_ASSERT(
      descriptor.abi_output_arg_count == descriptor.output_bindings.size(),
      "GFX: compiler-owned materialized output ABI count drift for ",
      materialized_stage_name(plan.io_plan));

  for (size_t input_idx = 0; input_idx < descriptor.input_bindings.size();
       ++input_idx) {
    assert_complete_materialized_binding(descriptor.input_bindings[input_idx],
                                         "input", plan.io_plan, input_idx);
  }
  for (size_t output_idx = 0; output_idx < descriptor.output_bindings.size();
       ++output_idx) {
    assert_complete_materialized_binding(descriptor.output_bindings[output_idx],
                                         "output", plan.io_plan, output_idx);
  }
}

} // namespace

PipelineStageMaterializer::PipelineStageMaterializer(
    const BackendStageFactory &stage_factory,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const RuntimeExecutableDescriptor &runtime_descriptor,
    GpuStageRuntimeOptions runtime_options)
    : m_stage_factory(stage_factory), m_runtime_descriptor(runtime_descriptor),
      m_runtime_options(std::move(runtime_options)) {
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

std::unique_ptr<GpuStage> PipelineStageMaterializer::create_stage(
    const std::shared_ptr<const ov::Node> &node) const {
  const auto *descriptor = descriptor_for(node);
  OPENVINO_ASSERT(descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for op ",
                  node ? node->get_friendly_name() : std::string("<null>"),
                  " (", node ? node->get_type_name() : "<null>", ")");
  auto stage = m_stage_factory.create_stage(
      RuntimeStageMaterializationContext{node, *descriptor});
  configure_stage(stage);
  return stage;
}

std::unique_ptr<GpuStage>
PipelineStageMaterializer::create_vendor_attention_stage(
    const PipelineVendorAttentionStagePlan &plan,
    const std::shared_ptr<const ov::Node> &final_node,
    const RuntimeStageExecutableDescriptor *descriptor) const {
  const auto *base_descriptor = descriptor_for(final_node);
  OPENVINO_ASSERT(base_descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for vendor attention output op ",
                  final_node ? final_node->get_friendly_name()
                             : std::string("<null>"));
  OPENVINO_ASSERT(plan.valid(),
                  "GFX: compiler did not provide a valid runtime vendor "
                  "attention stage plan for ",
                  plan.name);
  OPENVINO_ASSERT(
      plan.descriptor.stage_record_key == base_descriptor->stage_record_key,
      "GFX: vendor attention artifact stage key drift for ", plan.name);
  OPENVINO_ASSERT(descriptor,
                  "GFX: materialized vendor attention runtime descriptor is "
                  "required for ",
                  plan.name);
  OPENVINO_ASSERT(descriptor->payload_kind ==
                          KernelArtifactPayloadKind::VendorDescriptor &&
                      descriptor->payload,
                  "GFX: materialized vendor attention descriptor has no vendor "
                  "payload for ",
                  plan.name);

  auto stage = m_stage_factory.create_stage(
      RuntimeStageMaterializationContext{final_node, *descriptor});
  configure_stage(stage);
  return stage;
}

std::optional<MaterializedFusedSequenceStage>
PipelineStageMaterializer::create_attention_sequence_stage(
    const PipelineStageMaterializationPlan &plan,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops) const {
  const auto &node_indices = plan.fused_node_indices;
  const size_t stage_count = node_indices.size();
  if (stage_count < 3 || plan.fused_inner_stages.size() != stage_count) {
    return std::nullopt;
  }

  std::vector<FusedStageInfo> fused_stages;
  std::vector<FusedOutputLifetimeStage> fused_lifetime_stages;
  fused_stages.reserve(stage_count);
  fused_lifetime_stages.reserve(stage_count);

  for (size_t i = 0; i < stage_count; ++i) {
    const size_t idx = node_indices[i];
    if (idx >= ordered_ops.size()) {
      return std::nullopt;
    }
    const auto &stage_node = ordered_ops[idx];
    auto stage = create_stage(stage_node);
    if (!stage) {
      return std::nullopt;
    }

    const auto &inner_plan = plan.fused_inner_stages[i];
    FusedStageInfo info;
    info.stage = std::move(stage);
    info.output_indices = inner_plan.output_indices;
    info.inputs.reserve(inner_plan.inputs.size());
    for (const auto &input : inner_plan.inputs) {
      info.inputs.push_back(
          {to_runtime_fused_input_kind(input.kind), input.index});
    }

    FusedOutputLifetimeStage lifetime_stage;
    lifetime_stage.output_indices = info.output_indices;
    lifetime_stage.descriptor = descriptor_for(stage_node);
    lifetime_stage.inputs.reserve(inner_plan.inputs.size());
    for (const auto &input : inner_plan.inputs) {
      FusedOutputLifetimeInputRef lifetime_input;
      lifetime_input.kind = to_lifetime_fused_input_kind(input.kind);
      lifetime_input.index = input.index;
      lifetime_stage.inputs.push_back(lifetime_input);
    }
    fused_lifetime_stages.push_back(std::move(lifetime_stage));
    fused_stages.emplace_back(std::move(info));
  }

  if (fused_stages.size() != stage_count) {
    return std::nullopt;
  }

  MaterializedFusedSequenceStage result;
  result.output_lifetimes = build_fused_output_lifetime_plan(
      fused_lifetime_stages, m_runtime_descriptor.memory_plan,
      plan.io_plan.outputs.size());
  result.stage = std::make_unique<FusedSequenceStage>(
      std::move(fused_stages),
      plan.io_plan.node ? plan.io_plan.node->get_friendly_name()
                        : std::string("fused_attention"),
      "FusedAttention");
  result.materialized_stage_count = stage_count;
  return result;
}

std::shared_ptr<const RuntimeStageExecutableDescriptor>
PipelineStageMaterializer::create_materialized_descriptor(
    const PipelineStageMaterializationPlan &plan) const {
  const auto *base_descriptor = descriptor_for(plan.io_plan.node);
  OPENVINO_ASSERT(base_descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for materialized pipeline stage ",
                  plan.io_plan.node
                      ? plan.io_plan.node->get_friendly_name()
                      : std::string("<null>"));

  OPENVINO_ASSERT(
      plan.materialized_descriptor_valid,
      "GFX: compiler did not freeze a materialized runtime descriptor for ",
      materialized_stage_name(plan.io_plan));
  RuntimeStageExecutableDescriptor descriptor = plan.materialized_descriptor;
  OPENVINO_ASSERT(
      descriptor.stage_record_key == base_descriptor->stage_record_key,
      "GFX: compiler-owned materialized descriptor stage key drift for ",
      materialized_stage_name(plan.io_plan));
  OPENVINO_ASSERT(
      descriptor.stage_index == base_descriptor->stage_index,
      "GFX: compiler-owned materialized descriptor stage index drift for ",
      materialized_stage_name(plan.io_plan));
  assert_materialized_descriptor_complete(descriptor, plan);

  return std::make_shared<RuntimeStageExecutableDescriptor>(
      std::move(descriptor));
}

void PipelineStageMaterializer::configure_stage(
    const std::unique_ptr<GpuStage> &stage) const {
  if (stage) {
    stage->set_runtime_options(m_runtime_options);
  }
}

std::vector<PipelineStageDesc> materialize_pipeline_stage_descriptors(
    const PipelineStageRuntimeMaterializationRequest &request) {
  OPENVINO_ASSERT(request.stage_factory,
                  "GFX: pipeline materializer requires backend stage factory");
  OPENVINO_ASSERT(request.runtime_descriptor,
                  "GFX: pipeline materializer requires runtime descriptor");
  OPENVINO_ASSERT(request.runtime_plan,
                  "GFX: pipeline materializer requires runtime stage plan");

  PipelineStageMaterializer materializer(
      *request.stage_factory, request.runtime_plan->ordered_ops,
      *request.runtime_descriptor, request.runtime_options);

  std::vector<PipelineStageDesc> pipeline;
  pipeline.reserve(request.runtime_plan->stage_plans.size());
  for (const auto &stage_plan : request.runtime_plan->stage_plans) {
    PipelineStageDesc stage_desc;
    apply_pipeline_stage_io_plan(stage_desc, stage_plan.io_plan);
    stage_desc.runtime_descriptor =
        materializer.create_materialized_descriptor(stage_plan);

    uint64_t materialized_stage_count = 1;
    switch (stage_plan.kind) {
    case PipelineStageMaterializationKind::SingleStage:
      stage_desc.stage = materializer.create_stage(stage_plan.io_plan.node);
      OPENVINO_ASSERT(stage_desc.stage,
                      "GFX: unsupported op in MLIR pipeline: ",
                      stage_plan.io_plan.node
                          ? stage_plan.io_plan.node->get_friendly_name()
                          : std::string("<null>"),
                      " (",
                      stage_plan.io_plan.node
                          ? stage_plan.io_plan.node->get_type_name()
                          : std::string("<null>"),
                      ")");
      apply_single_stage_fusion_plan(*stage_desc.stage, stage_plan);
      break;
    case PipelineStageMaterializationKind::VendorAttention:
      stage_desc.stage = materializer.create_vendor_attention_stage(
          stage_plan.vendor_attention, stage_plan.io_plan.node,
          stage_desc.runtime_descriptor.get());
      break;
    case PipelineStageMaterializationKind::FusedAttentionSequence: {
      auto materialized =
          materializer.create_attention_sequence_stage(
              stage_plan, request.runtime_plan->ordered_ops);
      OPENVINO_ASSERT(materialized,
                      "GFX: failed to materialize compiler-owned fused "
                      "attention sequence plan");
      stage_desc.stage = std::move(materialized->stage);
      stage_desc.output_lifetimes =
          std::move(materialized->output_lifetimes);
      materialized_stage_count = materialized->materialized_stage_count;
      break;
    }
    }

    OPENVINO_ASSERT(stage_desc.stage,
                    "GFX: compiler-owned pipeline stage plan materialized to "
                    "a null runtime stage");
    if (request.compile_trace) {
      request.compile_trace->increment_counter("stage_create_count",
                                               materialized_stage_count);
    }
    pipeline.emplace_back(std::move(stage_desc));
  }
  return pipeline;
}

} // namespace gfx_plugin
} // namespace ov
