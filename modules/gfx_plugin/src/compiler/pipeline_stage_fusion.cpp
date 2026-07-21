// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/pipeline_stage_fusion.hpp"

#include <algorithm>
#include <utility>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softmax.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool is_msl_source_stage(const PipelineStageFusionContract &contract) {
  return contract.payload_kind == KernelArtifactPayloadKind::MslSource;
}

bool is_supported_input_activation_kind(ActivationKind kind) {
  switch (kind) {
  case ActivationKind::Relu:
  case ActivationKind::Sigmoid:
  case ActivationKind::Tanh:
  case ActivationKind::Gelu:
  case ActivationKind::Swish:
  case ActivationKind::HSwish:
  case ActivationKind::HSigmoid:
    return true;
  default:
    return false;
  }
}

bool read_const_f32_values(
    const std::shared_ptr<const ov::op::v0::Constant> &constant,
    std::vector<float> &values) {
  if (!constant) {
    return false;
  }
  const auto count = ov::shape_size(constant->get_shape());
  if (count == 0) {
    return false;
  }
  values.resize(count);
  if (constant->get_element_type() == ov::element::f32) {
    const auto *src = constant->get_data_ptr<float>();
    std::copy(src, src + count, values.begin());
    return true;
  }
  if (constant->get_element_type() == ov::element::f16) {
    const auto *src = constant->get_data_ptr<ov::float16>();
    for (size_t i = 0; i < count; ++i) {
      values[i] = static_cast<float>(src[i]);
    }
    return true;
  }
  return false;
}

bool read_uniform_scale_from_multiply(
    const std::shared_ptr<const ov::op::v1::Multiply> &multiply,
    const std::shared_ptr<const ov::Node> &producer, float &scale) {
  if (!multiply || !producer) {
    return false;
  }
  ov::Output<const ov::Node> scale_value;
  if (multiply->input_value(0).get_node_shared_ptr() == producer) {
    scale_value = multiply->input_value(1);
  } else if (multiply->input_value(1).get_node_shared_ptr() == producer) {
    scale_value = multiply->input_value(0);
  } else {
    return false;
  }
  auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(
      scale_value.get_node_shared_ptr());
  std::vector<float> values;
  if (!read_const_f32_values(constant, values)) {
    return false;
  }
  scale = values.front();
  return std::all_of(values.begin(), values.end(),
                     [&](float value) { return value == scale; });
}

bool extract_scaled_tensor_input(
    const std::shared_ptr<const ov::op::v1::Multiply> &multiply,
    ov::Output<const ov::Node> &tensor, float &scale) {
  if (!multiply || multiply->get_input_size() != 2) {
    return false;
  }
  auto const0 = ov::util::get_constant_from_source(multiply->input_value(0));
  auto const1 = ov::util::get_constant_from_source(multiply->input_value(1));
  ov::Output<const ov::Node> tensor_candidate;
  std::shared_ptr<const ov::op::v0::Constant> scale_const;
  if (const0 && !const1) {
    tensor_candidate = multiply->input_value(1);
    scale_const = const0;
  } else if (const1 && !const0) {
    tensor_candidate = multiply->input_value(0);
    scale_const = const1;
  } else {
    return false;
  }

  std::vector<float> values;
  if (!read_const_f32_values(scale_const, values)) {
    return false;
  }
  scale = values.front();
  if (!std::all_of(values.begin(), values.end(),
                   [&](float value) { return value == scale; })) {
    return false;
  }
  tensor = tensor_candidate;
  return true;
}

bool is_supported_softmax_node(const std::shared_ptr<const ov::Node> &node) {
  return static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Softmax>(node)) ||
         static_cast<bool>(ov::as_type_ptr<const ov::op::v8::Softmax>(node));
}

bool shape_matches_without_broadcast(const ov::PartialShape &input,
                                     const ov::PartialShape &output) {
  if (input.rank().is_dynamic() || output.rank().is_dynamic() ||
      input.rank().get_length() != output.rank().get_length()) {
    return false;
  }
  const auto rank = static_cast<size_t>(input.rank().get_length());
  for (size_t i = 0; i < rank; ++i) {
    if (input[i].is_static() && output[i].is_static() &&
        input[i].get_length() != output[i].get_length()) {
      return false;
    }
  }
  return true;
}

bool compiler_allows_precision_sensitive_arithmetic_fusion_group(
    const StageCompilerPolicy &stage_compiler_policy, const FusionGroup &group,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops) {
  if (group.node_indices.empty()) {
    return false;
  }
  const auto primary_idx = group.node_indices.front();
  if (primary_idx >= ordered_ops.size() || !ordered_ops[primary_idx]) {
    return false;
  }
  const auto &primary = ordered_ops[primary_idx];
  PrecisionSensitiveFusionQuery query{};
  query.group_kind = group.kind;
  query.stage_type = primary->get_type_name();
  query.primary_node = primary;
  query.element_type = primary->get_output_element_type(0);
  query.has_bias = group.bias.has_value();
  query.activation = group.activation;
  query.has_input_activation = group.input_activation.has_value();
  query.has_batchnorm = group.batchnorm.has_value();
  return allow_precision_sensitive_arithmetic_fusion(stage_compiler_policy,
                                                     query);
}

} // namespace

bool allow_stage_input_activation_fusion(
    const PipelineStageFusionContract &contract, size_t input_idx,
    ActivationKind kind) {
  if (!is_msl_source_stage(contract) || contract.op_family != "Multiply" ||
      input_idx >= 2 || !is_supported_input_activation_kind(kind)) {
    return false;
  }
  return !contract.element_type.is_integral_number() &&
         contract.element_type != ov::element::boolean;
}

bool allow_stage_residual_add_fusion(
    const PipelineStageFusionContract &contract) {
  return is_msl_source_stage(contract) && contract.op_family == "RMS";
}

bool pipeline_fusion_group_is_attention(const FusionGroup &group) noexcept {
  return group.kind == "Attention" || group.kind == "AttentionScale" ||
         group.kind == "AttentionScaleMask" || group.kind == "NativeSDPA" ||
         group.kind == "VendorAttention";
}

size_t pipeline_fusion_primary_index(const FusionGroup &group) noexcept {
  if (group.node_indices.empty()) {
    return PipelineStageIoPlan::npos;
  }
  return pipeline_fusion_group_is_attention(group) ? group.node_indices.back()
                                                   : group.node_indices.front();
}

bool pipeline_fusion_group_has_fp32_precision(
    const FusionGroup &group,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops) {
  for (const auto node_idx : group.node_indices) {
    if (node_idx < ordered_ops.size() &&
        ov::fp16_compression_is_disabled(ordered_ops[node_idx])) {
      return true;
    }
  }
  return false;
}

bool pipeline_fusion_group_is_precision_sensitive_arithmetic(
    const FusionGroup &group) noexcept {
  return group.kind == "ConvActivation" || group.kind == "ConvBiasActivation" ||
         group.kind == "ConvBatchNormAct" || group.kind == "ConvBias" ||
         group.kind == "ConvBatchNorm" || group.kind == "ConvScale" ||
         group.kind == "ConvScaleActivation" ||
         group.kind == "MatMulActivation" ||
         group.kind == "MatMulBiasActivation" || group.kind == "MatMulBias" ||
         group.kind == "EltwiseActivation" ||
         group.kind == "EltwiseBiasActivation" || group.kind == "EltwiseBias" ||
         group.kind == "EltwiseInputActivation";
}

bool pipeline_fusion_requires_bias_payload(const FusionGroup &group) noexcept {
  return group.kind == "ConvBias" || group.kind == "ConvBiasActivation" ||
         group.kind == "EltwiseBias" || group.kind == "EltwiseBiasActivation" ||
         group.kind == "MatMulBias" || group.kind == "MatMulBiasActivation";
}

bool pipeline_fusion_requires_batchnorm_payload(
    const FusionGroup &group) noexcept {
  return group.kind == "ConvBatchNorm" || group.kind == "ConvBatchNormAct" ||
         group.kind == "ConvScale" || group.kind == "ConvScaleActivation";
}

bool pipeline_input_activation_has_exclusive_consumer(
    const FusionGroup &group, size_t primary_idx,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const ModelOutputPorts &model_outputs) {
  if (group.kind != "EltwiseInputActivation" || group.node_indices.size() < 2 ||
      primary_idx >= ordered_ops.size()) {
    return false;
  }
  const size_t act_idx = group.node_indices[1];
  if (act_idx >= ordered_ops.size()) {
    return false;
  }
  const auto &act_node = ordered_ops[act_idx];
  if (!act_node || act_node->get_output_size() != 1 ||
      model_outputs.count(act_node.get()) != 0) {
    return false;
  }
  const auto &targets = act_node->output(0).get_target_inputs();
  if (targets.size() != 1) {
    return false;
  }
  const auto &target = *targets.begin();
  return target.get_node() == ordered_ops[primary_idx].get() &&
         target.get_index() == group.input_activation_input;
}

std::optional<PipelineVendorAttentionPlan> plan_vendor_attention_subgraph(
    const FusionGroup &group,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops) {
  if (group.node_indices.size() != 4) {
    return std::nullopt;
  }
  for (auto idx : group.node_indices) {
    if (idx >= ordered_ops.size()) {
      return std::nullopt;
    }
  }
  const auto first = ordered_ops[group.node_indices[0]];
  const auto second = ordered_ops[group.node_indices[1]];
  auto softmax = ordered_ops[group.node_indices[2]];
  auto matmul2 = ov::as_type_ptr<const ov::op::v0::MatMul>(
      ordered_ops[group.node_indices[3]]);
  auto matmul1 = ov::as_type_ptr<const ov::op::v0::MatMul>(first);
  auto scale = ov::as_type_ptr<const ov::op::v1::Multiply>(second);
  bool pre_scaled_key = false;
  if (!matmul1 || !scale) {
    scale = ov::as_type_ptr<const ov::op::v1::Multiply>(first);
    matmul1 = ov::as_type_ptr<const ov::op::v0::MatMul>(second);
    pre_scaled_key = static_cast<bool>(scale && matmul1);
  }
  if (!matmul1 || !scale || !is_supported_softmax_node(softmax) || !matmul2 ||
      !matmul1->get_transpose_a() || matmul1->get_transpose_b() ||
      matmul2->get_transpose_a() || !matmul2->get_transpose_b()) {
    return std::nullopt;
  }

  const auto q = matmul1->input_value(0);
  ov::Output<const ov::Node> k;
  ov::Output<const ov::Node> value;
  if (matmul2->input_value(0).get_node() == softmax.get()) {
    value = matmul2->input_value(1);
  } else if (matmul2->input_value(1).get_node() == softmax.get()) {
    value = matmul2->input_value(0);
  } else {
    return std::nullopt;
  }
  float attention_scale = 1.0f;
  if (pre_scaled_key) {
    if (softmax->input_value(0).get_node() != matmul1.get() ||
        matmul1->input_value(1).get_node() != scale.get() ||
        !extract_scaled_tensor_input(scale, k, attention_scale)) {
      return std::nullopt;
    }
  } else {
    if (softmax->input_value(0).get_node() != scale.get() ||
        !read_uniform_scale_from_multiply(scale, matmul1->shared_from_this(),
                                          attention_scale)) {
      return std::nullopt;
    }
    k = matmul1->input_value(1);
  }

  if (!q.get_partial_shape().is_static() ||
      !k.get_partial_shape().is_static() ||
      !value.get_partial_shape().is_static() ||
      !matmul2->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }

  PipelineVendorAttentionPlan plan;
  plan.name = matmul2->get_friendly_name();
  plan.element_type = q.get_element_type();
  plan.query_shape = q.get_shape();
  plan.key_shape = k.get_shape();
  plan.value_shape = value.get_shape();
  plan.output_shape = matmul2->get_output_shape(0);
  plan.scale = attention_scale;
  if (plan.element_type != ov::element::f32 &&
      plan.element_type != ov::element::f16) {
    return std::nullopt;
  }
  if (k.get_element_type() != plan.element_type ||
      value.get_element_type() != plan.element_type ||
      matmul2->get_output_element_type(0) != plan.element_type) {
    return std::nullopt;
  }
  if (plan.query_shape.size() != 4 || plan.key_shape.size() != 4 ||
      plan.value_shape.size() != 4 || plan.output_shape.size() != 4) {
    return std::nullopt;
  }
  if (plan.query_shape[0] != plan.key_shape[0] ||
      plan.query_shape[0] != plan.value_shape[0] ||
      plan.query_shape[1] != plan.key_shape[1] ||
      plan.query_shape[1] != plan.value_shape[1] ||
      plan.query_shape[2] != plan.key_shape[2] ||
      plan.key_shape[3] != plan.value_shape[3] ||
      plan.output_shape[0] != plan.query_shape[0] ||
      plan.output_shape[1] != plan.query_shape[1] ||
      plan.output_shape[2] != plan.value_shape[2] ||
      plan.output_shape[3] != plan.query_shape[3]) {
    return std::nullopt;
  }
  plan.query = PipelineStageInputLink{q.get_node_shared_ptr(), q.get_index()};
  plan.key = PipelineStageInputLink{k.get_node_shared_ptr(), k.get_index()};
  plan.value =
      PipelineStageInputLink{value.get_node_shared_ptr(), value.get_index()};
  return plan;
}

std::shared_ptr<const ov::Node>
find_rms_residual_add(const std::shared_ptr<const ov::Node> &rms,
                      const ModelOutputPorts &model_outputs,
                      const std::unordered_set<const ov::Node *> &fused_nodes) {
  if (!rms || rms->get_type_name() != std::string("RMS") ||
      rms->get_input_size() != 2 || rms->get_output_size() != 1) {
    return nullptr;
  }
  auto add = ov::as_type_ptr<const ov::op::v1::Add>(
      rms->input_value(0).get_node_shared_ptr());
  if (!add || add->get_output_size() != 1 || add->get_input_size() != 2 ||
      add->output(0).get_target_inputs().size() != 1 ||
      is_model_output_port(model_outputs, add.get(), 0) ||
      fused_nodes.count(add.get()) != 0) {
    return nullptr;
  }
  const auto out_shape = add->get_output_partial_shape(0);
  if (!shape_matches_without_broadcast(add->get_input_partial_shape(0),
                                       out_shape) ||
      !shape_matches_without_broadcast(add->get_input_partial_shape(1),
                                       out_shape)) {
    return nullptr;
  }
  if (!shape_matches_without_broadcast(out_shape,
                                       rms->get_input_partial_shape(0))) {
    return nullptr;
  }
  return add;
}

PipelineFusionSelectionPlan
plan_pipeline_fusions(const FusionPlan &fusion_plan,
                      const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
                      const ModelOutputPorts &model_outputs,
                      const PipelineFusionSelectionOptions &options) {
  PipelineFusionSelectionPlan selection;
  if (!options.enable_fusion || fusion_plan.groups.empty()) {
    return selection;
  }
  selection.fusion_plan = fusion_plan;

  selection.primary_group_indices.reserve(selection.fusion_plan.groups.size());
  for (size_t group_idx = 0; group_idx < selection.fusion_plan.groups.size();
       ++group_idx) {
    const auto &group = selection.fusion_plan.groups[group_idx];
    if (!options.fusion_capabilities
             .enable_precision_sensitive_arithmetic_fusion &&
        pipeline_fusion_group_is_precision_sensitive_arithmetic(group) &&
        pipeline_fusion_group_has_fp32_precision(group, ordered_ops)) {
      const bool allowed_by_compiler =
          options.stage_compiler_policy &&
          compiler_allows_precision_sensitive_arithmetic_fusion_group(
              *options.stage_compiler_policy, group, ordered_ops);
      if (allowed_by_compiler) {
        ++selection.precision_sensitive_vendor_allow_count;
      } else {
        ++selection.precision_sensitive_arithmetic_skip_count;
        continue;
      }
    }
    if (group.node_indices.size() < 2) {
      continue;
    }
    const bool attention_group = pipeline_fusion_group_is_attention(group);
    if (group.kind == "VendorAttention") {
      if (!plan_vendor_attention_subgraph(group, ordered_ops)) {
        continue;
      }
    }
    const size_t primary_idx = pipeline_fusion_primary_index(group);
    if (primary_idx == PipelineStageIoPlan::npos) {
      continue;
    }
    selection.primary_group_indices[primary_idx] = group_idx;

    const bool pre_fusion_supported = [&]() {
      if (group.kind != "EltwiseInputActivation" ||
          !group.input_activation.has_value() ||
          primary_idx >= ordered_ops.size() ||
          !pipeline_input_activation_has_exclusive_consumer(
              group, primary_idx, ordered_ops, model_outputs) ||
          !options.fusion_contract_for) {
        return false;
      }
      return allow_stage_input_activation_fusion(
          options.fusion_contract_for(ordered_ops[primary_idx]),
          group.input_activation_input, *group.input_activation);
    }();

    for (const auto node_idx : group.node_indices) {
      if (node_idx < ordered_ops.size()) {
        selection.fused_nodes.insert(ordered_ops[node_idx].get());
        if ((attention_group || pre_fusion_supported) &&
            node_idx != primary_idx) {
          selection.planned_fused_indices.insert(node_idx);
          selection.planned_fused_nodes.insert(ordered_ops[node_idx].get());
        }
      }
    }
  }
  return selection;
}

const FusionGroup *
primary_fusion_group_for(const PipelineFusionSelectionPlan &plan,
                         size_t primary_idx) noexcept {
  const auto it = plan.primary_group_indices.find(primary_idx);
  if (it == plan.primary_group_indices.end() ||
      it->second >= plan.fusion_plan.groups.size()) {
    return nullptr;
  }
  return &plan.fusion_plan.groups[it->second];
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
