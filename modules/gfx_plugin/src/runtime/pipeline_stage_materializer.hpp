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

#include "compiler/pipeline_stage_builder.hpp"
#include "openvino/core/node.hpp"
#include "runtime/backend_stage_factory.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/output_lifetime.hpp"
#include "runtime/pipeline_stage_desc.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfilingTrace;

struct MaterializedFusedSequenceStage {
  std::unique_ptr<GpuStage> stage;
  std::vector<RuntimeOutputLifetime> output_lifetimes;
  size_t materialized_stage_count = 0;
};

struct PipelineStageRuntimeMaterializationRequest {
  const BackendStageFactory *stage_factory = nullptr;
  const RuntimeExecutableDescriptor *runtime_descriptor = nullptr;
  const compiler::PipelineStageBuildResult *build_result = nullptr;
  GpuStageRuntimeOptions runtime_options;
  GfxProfilingTrace *compile_trace = nullptr;
};

class PipelineStageMaterializer final {
public:
  PipelineStageMaterializer(
      const BackendStageFactory &stage_factory,
      const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
      const RuntimeExecutableDescriptor &runtime_descriptor,
      GpuStageRuntimeOptions runtime_options);

  const RuntimeStageExecutableDescriptor *
  descriptor_for(const std::shared_ptr<const ov::Node> &node) const noexcept;

  size_t stage_index_for(const std::shared_ptr<const ov::Node> &node) const;

  compiler::PipelineStageFusionContract
  fusion_contract_for(const std::shared_ptr<const ov::Node> &node) const;

  std::unique_ptr<GpuStage>
  create_stage(const std::shared_ptr<const ov::Node> &node) const;

  std::unique_ptr<GpuStage> create_vendor_attention_stage(
      const compiler::PipelineVendorAttentionPlan &plan,
      const compiler::PipelineVendorAttentionArtifact &artifact,
      const std::shared_ptr<const ov::Node> &final_node) const;

  std::optional<MaterializedFusedSequenceStage> create_attention_sequence_stage(
      const compiler::PipelineStageMaterializationPlan &plan,
      const std::vector<std::shared_ptr<ov::Node>> &ordered_ops) const;

  void configure_stage(const std::unique_ptr<GpuStage> &stage) const;

private:
  const BackendStageFactory &m_stage_factory;
  const RuntimeExecutableDescriptor &m_runtime_descriptor;
  GpuStageRuntimeOptions m_runtime_options;
  std::unordered_map<const ov::Node *, const RuntimeStageExecutableDescriptor *>
      m_descriptors_by_node;
};

std::vector<PipelineStageDesc> materialize_pipeline_stage_descriptors(
    const PipelineStageRuntimeMaterializationRequest &request);

} // namespace gfx_plugin
} // namespace ov
