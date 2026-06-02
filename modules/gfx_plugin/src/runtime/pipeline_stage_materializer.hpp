// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "compiler/pipeline_stage_fusion.hpp"
#include "openvino/core/node.hpp"
#include "runtime/backend_stage_factory.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/output_lifetime.hpp"

namespace ov {
namespace gfx_plugin {

struct MaterializedFusedSequenceStage {
  compiler::PipelineStageIoPlan io_plan;
  std::unique_ptr<GpuStage> stage;
  std::vector<RuntimeOutputLifetime> output_lifetimes;
  size_t materialized_stage_count = 0;
};

class PipelineStageMaterializer final {
public:
  PipelineStageMaterializer(
      const BackendStageFactory &stage_factory,
      const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
      const RuntimeExecutableDescriptor &runtime_descriptor,
      GpuStageRuntimeOptions runtime_options,
      compiler::PipelineVendorAttentionArtifactResolver
          vendor_attention_artifact_resolver = {});

  const RuntimeStageExecutableDescriptor *
  descriptor_for(const std::shared_ptr<const ov::Node> &node) const noexcept;

  size_t stage_index_for(const std::shared_ptr<const ov::Node> &node) const;

  compiler::PipelineStageFusionContract
  fusion_contract_for(const std::shared_ptr<const ov::Node> &node) const;

  std::unique_ptr<GpuStage>
  create_stage(const std::shared_ptr<const ov::Node> &node) const;

  std::unique_ptr<GpuStage> create_vendor_attention_stage(
      const compiler::PipelineVendorAttentionPlan &plan,
      const std::shared_ptr<const ov::Node> &final_node) const;

  std::optional<MaterializedFusedSequenceStage> create_attention_sequence_stage(
      const FusionGroup &group,
      const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
      const compiler::PipelineStagePlanBuilder &stage_plan_builder,
      const compiler::PipelineOutputAliasMap &output_aliases) const;

  void configure_stage(const std::unique_ptr<GpuStage> &stage) const;

private:
  const BackendStageFactory &m_stage_factory;
  const RuntimeExecutableDescriptor &m_runtime_descriptor;
  GpuStageRuntimeOptions m_runtime_options;
  compiler::PipelineVendorAttentionArtifactResolver
      m_vendor_attention_artifact_resolver;
  std::unordered_map<const ov::Node *, const RuntimeStageExecutableDescriptor *>
      m_descriptors_by_node;
};

} // namespace gfx_plugin
} // namespace ov
