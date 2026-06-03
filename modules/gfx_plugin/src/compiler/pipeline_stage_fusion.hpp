// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/gfx_activation.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/operation_support.hpp"
#include "compiler/pipeline_stage_plan.hpp"
#include "compiler/stage_compiler_policy.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transforms/fusion_pass.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct PipelineStageFusionContract {
  std::string op_family;
  KernelArtifactOrigin origin = KernelArtifactOrigin::Unknown;
  KernelArtifactPayloadKind payload_kind = KernelArtifactPayloadKind::None;
  ov::element::Type element_type = ov::element::dynamic;
};

bool allow_stage_input_activation_fusion(
    const PipelineStageFusionContract &contract, size_t input_idx,
    ActivationKind kind);

bool allow_stage_residual_add_fusion(
    const PipelineStageFusionContract &contract);

struct PipelineVendorAttentionPlan {
  std::string name;
  ov::element::Type element_type = ov::element::dynamic;
  ov::Shape query_shape;
  ov::Shape key_shape;
  ov::Shape value_shape;
  ov::Shape output_shape;
  float scale = 1.0f;
  PipelineStageInputLink query;
  PipelineStageInputLink key;
  PipelineStageInputLink value;
};

struct PipelineFusionSelectionOptions {
  bool enable_fusion = true;
  bool debug_dump_ir = false;
  FusionCapabilities fusion_capabilities{};
  const StageCompilerPolicy *stage_compiler_policy = nullptr;
  std::function<PipelineStageFusionContract(
      const std::shared_ptr<const ov::Node> &)>
      fusion_contract_for;
};

struct PipelineFusionSelectionPlan {
  FusionPlan fusion_plan;
  std::unordered_map<size_t, size_t> primary_group_indices;
  std::unordered_set<size_t> planned_fused_indices;
  std::unordered_set<const ov::Node *> planned_fused_nodes;
  std::unordered_set<const ov::Node *> fused_nodes;
  size_t precision_sensitive_vendor_allow_count = 0;
  size_t precision_sensitive_arithmetic_skip_count = 0;
};

bool pipeline_fusion_group_is_attention(const FusionGroup &group) noexcept;
size_t pipeline_fusion_primary_index(const FusionGroup &group) noexcept;
bool pipeline_fusion_group_has_fp32_precision(
    const FusionGroup &group,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops);
bool pipeline_fusion_group_is_precision_sensitive_arithmetic(
    const FusionGroup &group) noexcept;
bool pipeline_fusion_requires_bias_payload(const FusionGroup &group) noexcept;
bool pipeline_fusion_requires_batchnorm_payload(
    const FusionGroup &group) noexcept;
bool pipeline_input_activation_has_exclusive_consumer(
    const FusionGroup &group, size_t primary_idx,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const ModelOutputPorts &model_outputs);

std::optional<PipelineVendorAttentionPlan> plan_vendor_attention_subgraph(
    const FusionGroup &group,
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops);

std::shared_ptr<const ov::Node>
find_rms_residual_add(const std::shared_ptr<const ov::Node> &rms,
                      const ModelOutputPorts &model_outputs,
                      const std::unordered_set<const ov::Node *> &fused_nodes);

PipelineFusionSelectionPlan
plan_pipeline_fusions(const std::shared_ptr<const ov::Model> &model,
                      const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
                      const ModelOutputPorts &model_outputs,
                      const PipelineFusionSelectionOptions &options);

const FusionGroup *
primary_fusion_group_for(const PipelineFusionSelectionPlan &plan,
                         size_t primary_idx) noexcept;

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
