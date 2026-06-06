// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/pipeline_stage_materializer.hpp"

#include <string>
#include <string_view>
#include <utility>

#include "common/gpu_backend.hpp"
#include "openvino/core/except.hpp"
#include "runtime/fused_output_lifetime_plan.hpp"
#include "runtime/fused_sequence_stage.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void apply_pipeline_stage_io_plan(PipelineStageDesc &stage_desc,
                                  PipelineStageIoPlan stage_plan) {
  static_cast<PipelineStageIoPlan &>(stage_desc) = std::move(stage_plan);
}

std::string materialized_stage_name(const PipelineStageIoPlan &io_plan);
std::string materialized_stage_family(const PipelineStageIoPlan &io_plan);

void apply_descriptor_output_contracts(
    PipelineStageDesc &stage_desc,
    const RuntimeStageExecutableDescriptor &descriptor) {
  OPENVINO_ASSERT(
      descriptor.output_bindings.size() == stage_desc.outputs.size(),
      "GFX: compiler-owned output binding count drift for materialized "
      "pipeline stage ",
      materialized_stage_name(stage_desc),
      ": descriptor=", descriptor.output_bindings.size(),
      " plan=", stage_desc.outputs.size());

  for (size_t output_idx = 0; output_idx < stage_desc.outputs.size();
       ++output_idx) {
    auto &output = stage_desc.outputs[output_idx];
    const auto &binding = descriptor.output_bindings[output_idx];

    const auto descriptor_type =
        element_type_from_contract(binding.element_type);
    if (descriptor_type != ov::element::dynamic) {
      OPENVINO_ASSERT(
          output.type == ov::element::dynamic || output.type == descriptor_type,
          "GFX: compiler-owned output type contract drift for materialized "
          "pipeline stage ",
          materialized_stage_name(stage_desc), " output ", output_idx,
          ": descriptor=", descriptor_type, " plan=", output.type);
      output.type = descriptor_type;
    }

    ov::Shape descriptor_shape;
    if (parse_static_shape_contract(binding.partial_shape, descriptor_shape)) {
      OPENVINO_ASSERT(
          output.shape.empty() || output.shape == descriptor_shape,
          "GFX: compiler-owned output shape contract drift for materialized "
          "pipeline stage ",
          materialized_stage_name(stage_desc), " output ", output_idx);
      output.shape = std::move(descriptor_shape);
    }
  }
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
                    materialized_stage_name(plan.io_plan));
  }

  if (plan.post_ops.input_activation) {
    const auto &input_activation = *plan.post_ops.input_activation;
    OPENVINO_ASSERT(
        stage.fuse_input_activation(input_activation.input_idx,
                                    input_activation.kind,
                                    input_activation.alpha),
        "GFX: runtime stage rejected compiler-owned input activation fusion "
        "plan for ",
        materialized_stage_name(plan.io_plan));
  }
  if (plan.post_ops.batchnorm) {
    OPENVINO_ASSERT(stage.fuse_batchnorm(*plan.post_ops.batchnorm),
                    "GFX: runtime stage rejected compiler-owned BatchNorm "
                    "fusion plan for ",
                    materialized_stage_name(plan.io_plan));
  }
  if (plan.post_ops.bias) {
    OPENVINO_ASSERT(stage.fuse_bias(*plan.post_ops.bias),
                    "GFX: runtime stage rejected compiler-owned bias fusion "
                    "plan for ",
                    materialized_stage_name(plan.io_plan));
  }
  if (plan.post_ops.activation) {
    OPENVINO_ASSERT(stage.fuse_activation(*plan.post_ops.activation,
                                          plan.post_ops.activation_alpha),
                    "GFX: runtime stage rejected compiler-owned activation "
                    "fusion plan for ",
                    materialized_stage_name(plan.io_plan));
  }
}

FusedInputKind to_runtime_fused_input_kind(PipelineFusedInputPlan::Kind kind) {
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

FusedOutputLifetimeInputRef::Kind
to_lifetime_fused_input_kind(PipelineFusedInputPlan::Kind kind) {
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

size_t count_launch_plan_outputs(const KernelLaunchPlanDescriptor &plan) {
  size_t count = 0;
  for (const auto &role : plan.buffer_roles) {
    if (role == "tensor_output") {
      ++count;
    }
  }
  return count;
}

std::string materialized_stage_name(const PipelineStageIoPlan &io_plan) {
  return io_plan.stage_name.empty() ? std::string("<unnamed>")
                                    : io_plan.stage_name;
}

std::string materialized_stage_family(const PipelineStageIoPlan &io_plan) {
  return io_plan.op_family.empty() ? std::string("<unknown>")
                                   : io_plan.op_family;
}

void assert_complete_materialized_binding(
    const RuntimeTensorBindingContract &binding, std::string_view direction,
    const PipelineStageIoPlan &io_plan, size_t binding_idx) {
  OPENVINO_ASSERT(
      binding_complete(binding), "GFX: compiler-owned materialized ", direction,
      " binding contract is incomplete for pipeline stage ",
      materialized_stage_name(io_plan), " at index ", binding_idx,
      "; runtime must not synthesize ABI, memory regions or tensor layout");
}

void assert_materialized_descriptor_complete(
    const RuntimeStageExecutableDescriptor &descriptor,
    const PipelineStageMaterializationPlan &plan) {
  OPENVINO_ASSERT(
      !descriptor.manifest_ref.empty() && !descriptor.abi_fingerprint.empty() &&
          !descriptor.artifact_key.empty() &&
          !descriptor.backend_domain.empty() && !descriptor.kernel_id.empty() &&
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
        materialized_stage_name(plan.io_plan),
        ": descriptor=", descriptor.input_bindings.size(),
        " plan=", plan.io_plan.inputs.size());
    OPENVINO_ASSERT(
        descriptor.output_bindings.size() == plan.io_plan.outputs.size(),
        "GFX: compiler-owned materialized output binding count drift for ",
        materialized_stage_name(plan.io_plan),
        ": descriptor=", descriptor.output_bindings.size(),
        " plan=", plan.io_plan.outputs.size());
  }

  if (descriptor.launch_plan.valid) {
    OPENVINO_ASSERT(
        descriptor.launch_plan.buffer_roles.size() == descriptor.abi_arg_count,
        "GFX: compiler-owned materialized launch-plan ABI count drift for ",
        materialized_stage_name(plan.io_plan));
    OPENVINO_ASSERT(
        count_launch_plan_outputs(descriptor.launch_plan) ==
            descriptor.abi_output_arg_count,
        "GFX: compiler-owned materialized launch-plan output count drift for ",
        materialized_stage_name(plan.io_plan));
  } else {
    OPENVINO_ASSERT(
        descriptor.abi_arg_count >= descriptor.input_bindings.size(),
        "GFX: compiler-owned materialized tensor ABI count drift for ",
        materialized_stage_name(plan.io_plan));
    OPENVINO_ASSERT(
        descriptor.abi_output_arg_count == descriptor.output_bindings.size(),
        "GFX: compiler-owned materialized output ABI count drift for ",
        materialized_stage_name(plan.io_plan));
  }

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
    const RuntimeExecutableDescriptor &runtime_descriptor,
    GpuStageRuntimeOptions runtime_options)
    : m_stage_factory(stage_factory), m_runtime_descriptor(runtime_descriptor),
      m_runtime_options(std::move(runtime_options)) {
  m_descriptors_by_stage_index.assign(runtime_descriptor.stages.size(),
                                      nullptr);
  for (size_t i = 0; i < runtime_descriptor.stages.size(); ++i) {
    const auto &descriptor = runtime_descriptor.stages[i];
    OPENVINO_ASSERT(
        descriptor.stage_index != PipelineStageIoPlan::npos &&
            descriptor.stage_index < runtime_descriptor.stages.size(),
        "GFX: runtime executable descriptor stage index out of range at ", i);
    OPENVINO_ASSERT(!m_descriptors_by_stage_index[descriptor.stage_index],
                    "GFX: duplicate runtime executable descriptor stage index ",
                    descriptor.stage_index);
    m_descriptors_by_stage_index[descriptor.stage_index] = &descriptor;
  }
}

const RuntimeStageExecutableDescriptor *
PipelineStageMaterializer::descriptor_for_stage_index(
    size_t stage_index) const noexcept {
  if (stage_index == PipelineStageIoPlan::npos ||
      stage_index >= m_descriptors_by_stage_index.size()) {
    return nullptr;
  }
  return m_descriptors_by_stage_index[stage_index];
}

std::unique_ptr<GpuStage> PipelineStageMaterializer::create_stage(
    const RuntimeStageExecutableDescriptor &descriptor) const {
  auto stage = m_stage_factory.create_stage(
      RuntimeStageMaterializationContext{descriptor});
  configure_stage(stage);
  return stage;
}

std::unique_ptr<GpuStage>
PipelineStageMaterializer::create_vendor_attention_stage(
    const PipelineVendorAttentionStagePlan &plan,
    const RuntimeStageExecutableDescriptor *descriptor) const {
  const auto *base_descriptor = descriptor_for_stage_index(
      descriptor ? descriptor->stage_index : PipelineStageIoPlan::npos);
  OPENVINO_ASSERT(base_descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for vendor attention stage ",
                  plan.name);
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
      RuntimeStageMaterializationContext{*descriptor});
  configure_stage(stage);
  return stage;
}

std::optional<MaterializedFusedSequenceStage>
PipelineStageMaterializer::create_attention_sequence_stage(
    const PipelineStageMaterializationPlan &plan) const {
  const size_t stage_count = plan.fused_descriptor_stage_indices.size();
  if (stage_count < 3 || plan.fused_inner_stages.size() != stage_count ||
      plan.fused_descriptor_stage_indices.size() != stage_count) {
    return std::nullopt;
  }

  std::vector<FusedStageInfo> fused_stages;
  std::vector<FusedOutputLifetimeStage> fused_lifetime_stages;
  fused_stages.reserve(stage_count);
  fused_lifetime_stages.reserve(stage_count);

  for (size_t i = 0; i < stage_count; ++i) {
    const auto *stage_descriptor =
        descriptor_for_stage_index(plan.fused_descriptor_stage_indices[i]);
    if (!stage_descriptor) {
      return std::nullopt;
    }
    auto stage = create_stage(*stage_descriptor);
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
    lifetime_stage.descriptor = stage_descriptor;
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
      materialized_stage_name(plan.io_plan),
      "FusedAttention");
  result.materialized_stage_count = stage_count;
  return result;
}

std::shared_ptr<const RuntimeStageExecutableDescriptor>
PipelineStageMaterializer::create_materialized_descriptor(
    const PipelineStageMaterializationPlan &plan) const {
  const auto *base_descriptor =
      descriptor_for_stage_index(plan.descriptor_stage_index);
  OPENVINO_ASSERT(base_descriptor,
                  "GFX: missing compiler-owned runtime executable descriptor "
                  "for materialized pipeline stage ",
                  materialized_stage_name(plan.io_plan));

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
  OPENVINO_ASSERT(request.runtime_descriptor->pipeline_plan,
                  "GFX: pipeline materializer requires descriptor-owned "
                  "compiler materialization plan");
  const auto &runtime_plan = *request.runtime_descriptor->pipeline_plan;

  PipelineStageMaterializer materializer(*request.stage_factory,
                                         *request.runtime_descriptor,
                                         request.runtime_options);

  std::vector<PipelineStageDesc> pipeline;
  pipeline.reserve(runtime_plan.stage_plans.size());
  for (const auto &stage_plan : runtime_plan.stage_plans) {
    PipelineStageDesc stage_desc;
    apply_pipeline_stage_io_plan(stage_desc, stage_plan.io_plan);
    stage_desc.runtime_descriptor =
        materializer.create_materialized_descriptor(stage_plan);
    apply_descriptor_output_contracts(stage_desc,
                                      *stage_desc.runtime_descriptor);

    uint64_t materialized_stage_count = 1;
    switch (stage_plan.kind) {
    case PipelineStageMaterializationKind::SingleStage:
      stage_desc.stage = materializer.create_stage(
          *stage_desc.runtime_descriptor);
      OPENVINO_ASSERT(
          stage_desc.stage, "GFX: unsupported op in MLIR pipeline: ",
          materialized_stage_name(stage_plan.io_plan), " (",
          materialized_stage_family(stage_plan.io_plan), ")");
      apply_single_stage_fusion_plan(*stage_desc.stage, stage_plan);
      break;
    case PipelineStageMaterializationKind::VendorAttention:
      stage_desc.stage = materializer.create_vendor_attention_stage(
          stage_plan.vendor_attention, stage_desc.runtime_descriptor.get());
      break;
    case PipelineStageMaterializationKind::FusedAttentionSequence: {
      auto materialized = materializer.create_attention_sequence_stage(
          stage_plan);
      OPENVINO_ASSERT(materialized,
                      "GFX: failed to materialize compiler-owned fused "
                      "attention sequence plan");
      stage_desc.stage = std::move(materialized->stage);
      stage_desc.output_lifetimes = std::move(materialized->output_lifetimes);
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
