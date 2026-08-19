// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "compiler/pipeline_stage_fusion.hpp"
#include "compiler/pipeline_stage_plan.hpp"
#include "runtime/pipeline_stage_plan.hpp"

namespace ov {
namespace gfx_plugin {

namespace compiler {

struct PipelineStageInputTransformBinding {
  size_t input_idx = 0;
  PipelineInputTransformPlan transform;
};

struct PipelineStageResidualAddFusionPlan {
  std::shared_ptr<const ov::Node> add_node;
};

struct PipelineStageInputActivationFusionPlan {
  size_t input_idx = 0;
  ActivationKind kind = ActivationKind::Relu;
  float alpha = 0.0f;
};

struct PipelineStagePostOpFusionPlan {
  std::optional<PipelineStageInputActivationFusionPlan> input_activation;
  std::optional<BatchNormParams> batchnorm;
  std::optional<BiasParams> bias;
  std::optional<ActivationKind> activation;
  float activation_alpha = 0.0f;

  bool empty() const noexcept {
    return !input_activation && !batchnorm && !bias && !activation;
  }
};

enum class PipelineStageMaterializationKind {
  SingleStage,
  VendorAttention,
  FusedAttentionSequence,
};

struct PipelineFusedInputPlan {
  enum class Kind {
    None,
    Output,
    External,
  };

  Kind kind = Kind::None;
  size_t index = 0;
};

struct PipelineFusedInnerStagePlan {
  std::vector<size_t> output_indices;
  std::vector<PipelineFusedInputPlan> inputs;
};

struct PipelineStageMaterializationPlan {
  PipelineStageMaterializationKind kind =
      PipelineStageMaterializationKind::SingleStage;
  PipelineStageIoPlan io_plan;
  PipelineVendorAttentionPlan vendor_attention;
  PipelineVendorAttentionArtifact vendor_attention_artifact;
  FusionGroup fusion_group;
  std::vector<PipelineFusedInnerStagePlan> fused_inner_stages;
  std::vector<PipelineStageInputTransformBinding> input_transforms;
  std::optional<PipelineStageResidualAddFusionPlan> residual_add;
  PipelineStagePostOpFusionPlan post_ops;
};

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
