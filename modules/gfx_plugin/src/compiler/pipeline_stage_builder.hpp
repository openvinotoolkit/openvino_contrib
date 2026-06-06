// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "compiler/backend_target.hpp"
#include "compiler/pipeline_stage_fusion.hpp"
#include "compiler/pipeline_stage_plan.hpp"
#include "runtime/pipeline_stage_plan.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfilingTrace;
struct RuntimeExecutableDescriptor;

namespace compiler {

class BackendRegistry;

struct PipelineStagePublicOutputSource {
  std::shared_ptr<const ov::Node> node;
  size_t port = 0;
  ov::Shape shape;
  ov::element::Type type = ov::element::dynamic;
};

struct PipelineStageGraphSnapshot {
  std::vector<std::shared_ptr<ov::Node>> ordered_ops;
  ModelOutputPorts model_outputs;
  std::unordered_map<const ov::Node *, size_t> param_index;
  std::vector<PipelineStagePublicOutputSource> public_outputs;
  FusionPlan fusion_plan;
  size_t graph_op_count = 0;

  bool valid() const noexcept { return !ordered_ops.empty(); }
};

struct PipelineStageBuildRequest {
  PipelineStageGraphSnapshot graph;
  const RuntimeExecutableDescriptor *runtime_descriptor = nullptr;
  const BackendRegistry *backend_registry = nullptr;
  BackendTarget target;
  std::string backend_name;
  bool enable_fusion = true;
  GfxProfilingTrace *compile_trace = nullptr;
};

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

PipelineStageGraphSnapshot
make_pipeline_stage_graph_snapshot(const std::shared_ptr<const ov::Model> &model,
                                   const FusionConfig &fusion_config);

FusionConfig make_pipeline_stage_fusion_config(
    const FusionCapabilities &fusion_capabilities, bool enable_fusion,
    bool debug_dump_ir);

std::shared_ptr<const PipelineStageRuntimePlan>
build_pipeline_stage_runtime_plan(const PipelineStageBuildRequest &request);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
