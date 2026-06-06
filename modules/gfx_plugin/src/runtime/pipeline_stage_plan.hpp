// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "common/gfx_activation.hpp"
#include "common/gfx_bias.hpp"
#include "common/gpu_parallelism_profile.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gfx_batchnorm.hpp"

namespace ov {
namespace gfx_plugin {

enum class PipelineStageTensorRefKind {
  None,
  Parameter,
  StageOutput,
};

struct PipelineStageTensorRef {
  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  PipelineStageTensorRefKind kind = PipelineStageTensorRefKind::None;
  size_t index = npos;
  size_t port = npos;

  bool valid() const noexcept {
    return kind != PipelineStageTensorRefKind::None && index != npos &&
           port != npos;
  }
};

struct PipelineStageInputLink {
  size_t port = 0;
  PipelineStageTensorRef source_ref;
};

struct PipelineStageOutputAlias {
  size_t source_port = 0;
  size_t output_port = 0;
  PipelineStageTensorRef source_ref;
};

struct PipelineStageOutputDesc {
  ov::Shape shape;
  ov::element::Type type = ov::element::dynamic;
  bool is_model_output = false;
  size_t source_port = 0;
  std::string direct_stateful_assign_variable_id;
  PipelineStageTensorRef source_ref;
};

struct PipelineStagePublicOutputDesc {
  PipelineStageTensorRef source_ref;
  ov::Shape shape;
  ov::element::Type type = ov::element::dynamic;
};

struct PipelineStageIoPlan {
  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  std::string stage_name;
  std::string op_family;
  size_t runtime_stage_index = npos;
  std::vector<PipelineStageOutputDesc> outputs;
  std::vector<PipelineStageInputLink> inputs;
  std::vector<PipelineStageOutputAlias> output_aliases;
};

struct PipelineStageInputTransformPlan {
  ov::Shape source_shape;
  std::vector<int64_t> transpose_permutation;

  bool has_transpose() const {
    return !source_shape.empty() && !transpose_permutation.empty();
  }
};

struct PipelineStageInputTransformBinding {
  size_t input_idx = 0;
  PipelineStageInputTransformPlan transform;
};

struct PipelineStageResidualAddFusionPlan {};

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

struct PipelineVendorAttentionStagePlan {
  std::string name;
  RuntimeStageExecutableDescriptor descriptor;

  bool valid() const noexcept {
    return !name.empty() && descriptor.payload && descriptor.payload->valid() &&
           !descriptor.artifact_key.empty();
  }
};

struct PipelineStageMaterializationPlan {
  PipelineStageMaterializationKind kind =
      PipelineStageMaterializationKind::SingleStage;
  PipelineStageIoPlan io_plan;
  size_t descriptor_stage_index = PipelineStageIoPlan::npos;
  RuntimeStageExecutableDescriptor materialized_descriptor;
  bool materialized_descriptor_valid = false;
  PipelineVendorAttentionStagePlan vendor_attention;
  std::vector<size_t> fused_node_indices;
  std::vector<size_t> fused_descriptor_stage_indices;
  std::vector<PipelineFusedInnerStagePlan> fused_inner_stages;
  std::vector<PipelineStageInputTransformBinding> input_transforms;
  std::optional<PipelineStageResidualAddFusionPlan> residual_add;
  PipelineStagePostOpFusionPlan post_ops;
};

struct PipelineStageRuntimeOptionsPlan {
  bool custom_kernel_dispatch_enabled = false;
  GpuParallelismProfile custom_kernel_dispatch_profile{};
};

struct PipelineStageRuntimePlan {
  std::vector<PipelineStageMaterializationPlan> stage_plans;
  std::vector<PipelineStagePublicOutputDesc> public_outputs;
  PipelineStageRuntimeOptionsPlan runtime_options;
};

} // namespace gfx_plugin
} // namespace ov
