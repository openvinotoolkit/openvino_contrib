// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/artifact_payload.hpp"
#include "common/gfx_activation.hpp"
#include "common/gfx_bias.hpp"
#include "common/gpu_parallelism_profile.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gfx_batchnorm.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeTensorBindingContract {
  std::string logical_name;
  std::string memory_region_id;
  std::string role;
  std::string element_type;
  std::string partial_shape;
  std::string layout = "logical";
  std::string storage_kind = "device_buffer";
  std::string lifetime_class;
  std::string alias_group;
  std::string stateful_prebind_variable_id;
  std::string stateful_prebind_shape_rule = "none";
  int64_t stateful_prebind_shape_axis = -1;
  bool external_binding = false;
  bool host_visible = false;
};

struct RuntimeStageExecutableDescriptor {
  size_t stage_index = 0;
  uint64_t stage_record_key = 0;
  size_t artifact_descriptor_index = 0;
  std::string manifest_ref;
  std::string abi_fingerprint;
  std::string artifact_key;
  std::string backend_domain;
  std::string kernel_id;
  std::string op_family;
  std::string stage_name;
  KernelArtifactOrigin origin = KernelArtifactOrigin::Unknown;
  KernelArtifactPayloadKind payload_kind = KernelArtifactPayloadKind::None;
  std::string entry_point;
  std::string compile_options_key;
  uint32_t abi_arg_count = 0;
  uint32_t abi_output_arg_count = 0;
  std::string dispatch_contract;
  std::string layout_contract = "logical";
  std::string runtime_shape_rule = "static_or_descriptor";
  std::vector<int64_t> runtime_shape_i64_metadata;
  bool requires_runtime_shape_args = false;
  bool tensor_view_only = false;
  uint32_t submission_stage_weight = 1;
  uint64_t submission_macs_estimate = 0;
  bool submission_dependency_boundary = false;
  std::string stateful_effect = "none";
  std::string stateful_variable_id;
  std::vector<std::string> tensor_roles;
  std::vector<std::string> scalar_roles;
  uint32_t runtime_param_buffer_count = 0;
  std::vector<int64_t> runtime_param_i64_metadata;
  bool runtime_param_reduce_keep_dims = false;
  bool runtime_param_reduce_keep_dims_valid = false;
  KernelLaunchPlanDescriptor launch_plan;
  std::string exception_ticket;
  std::string exception_reason;
  std::string exception_removal_condition;
  bool optional_cache_payload_allowed = true;
  std::vector<KernelArtifactConstTensor> const_tensors;
  std::shared_ptr<const KernelArtifactPayload> payload;
  std::vector<RuntimeTensorBindingContract> input_bindings;
  std::vector<RuntimeTensorBindingContract> output_bindings;
};

enum class RuntimePublicOutputSourceKind {
  None,
  Parameter,
  StageOutput,
};

struct RuntimePublicOutputDescriptor {
  RuntimePublicOutputSourceKind kind = RuntimePublicOutputSourceKind::None;
  size_t index = static_cast<size_t>(-1);
  size_t port = 0;
  ov::Shape static_shape;
  ov::element::Type static_type = ov::element::dynamic;
};

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

struct RuntimeMemoryRegionDescriptor {
  std::string region_id;
  std::string logical_tensor_name;
  std::string kind;
  std::string element_type;
  std::string partial_shape;
  std::string layout = "logical";
  std::string storage_kind = "device_buffer";
  std::string alias_group;
  size_t first_stage = 0;
  size_t last_stage = 0;
  bool external_binding = false;
  bool host_visible = false;
};

struct RuntimeMemoryAliasGroupDescriptor {
  std::string group_id;
  std::vector<std::string> region_ids;
  bool output_aliasing = false;
};

struct RuntimeTransientArenaDescriptor {
  std::string arena_id;
  std::string storage_kind = "device_buffer";
  std::vector<std::string> region_ids;
};

struct RuntimeMemoryPlanDescriptor {
  uint32_t schema_version = 1;
  std::string fingerprint;
  std::vector<RuntimeMemoryRegionDescriptor> regions;
  std::vector<RuntimeMemoryAliasGroupDescriptor> alias_groups;
  std::vector<RuntimeTransientArenaDescriptor> transient_arenas;
  bool hidden_host_copies_allowed = false;

  bool has_region(std::string_view region_id) const;
  bool has_alias_group(std::string_view group_id) const;
};

struct RuntimeExecutableDescriptorVerificationResult {
  std::vector<std::string> diagnostics;

  bool valid() const noexcept { return diagnostics.empty(); }
};

std::vector<GfxKernelBufferRole> materialize_descriptor_launch_roles(
    const KernelLaunchPlanDescriptor &plan, std::string_view stage_name);

struct RuntimeExecutableDescriptor {
  uint32_t schema_version = 1;
  std::string target_fingerprint;
  RuntimeMemoryPlanDescriptor memory_plan;
  std::vector<RuntimeStageExecutableDescriptor> stages;
  bool materialization_finalized = false;
  std::vector<PipelineStageMaterializationPlan> materialization_stages;
  std::vector<RuntimePublicOutputDescriptor> public_outputs;
  PipelineStageRuntimeOptionsPlan runtime_options;
};

bool runtime_descriptor_source_payload_kind(
    KernelArtifactPayloadKind kind) noexcept;

bool runtime_descriptor_payload_kind_requires_payload(
    KernelArtifactPayloadKind kind) noexcept;

bool runtime_stage_descriptor_is_materializable(
    const RuntimeStageExecutableDescriptor &descriptor) noexcept;

} // namespace gfx_plugin
} // namespace ov
