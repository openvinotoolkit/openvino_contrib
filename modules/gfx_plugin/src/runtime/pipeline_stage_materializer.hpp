// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "runtime/backend_stage_factory.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/output_lifetime.hpp"
#include "runtime/pipeline_stage_desc.hpp"
#include "runtime/pipeline_stage_plan.hpp"

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
  GpuStageRuntimeOptions runtime_options;
  GfxProfilingTrace *compile_trace = nullptr;
};

class PipelineStageMaterializer final {
public:
  PipelineStageMaterializer(
      const BackendStageFactory &stage_factory,
      const RuntimeExecutableDescriptor &runtime_descriptor,
      GpuStageRuntimeOptions runtime_options);

  const RuntimeStageExecutableDescriptor *
  descriptor_for_stage_index(size_t stage_index) const noexcept;

  std::unique_ptr<GpuStage>
  create_stage(const RuntimeStageExecutableDescriptor &descriptor) const;

  std::unique_ptr<GpuStage> create_vendor_attention_stage(
      const PipelineVendorAttentionStagePlan &plan,
      const RuntimeStageExecutableDescriptor *descriptor) const;

  std::optional<MaterializedFusedSequenceStage> create_attention_sequence_stage(
      const PipelineStageMaterializationPlan &plan) const;

  std::shared_ptr<const RuntimeStageExecutableDescriptor>
  create_materialized_descriptor(
      const PipelineStageMaterializationPlan &plan) const;

  void configure_stage(const std::unique_ptr<GpuStage> &stage) const;

private:
  const BackendStageFactory &m_stage_factory;
  const RuntimeExecutableDescriptor &m_runtime_descriptor;
  GpuStageRuntimeOptions m_runtime_options;
  std::vector<const RuntimeStageExecutableDescriptor *>
      m_descriptors_by_stage_index;
};

std::vector<PipelineStageDesc> materialize_pipeline_stage_descriptors(
    const PipelineStageRuntimeMaterializationRequest &request);

} // namespace gfx_plugin
} // namespace ov
