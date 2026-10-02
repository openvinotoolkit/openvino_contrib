// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/runtime_execution_plan.hpp"

#include <utility>

#include "openvino/core/except.hpp"
#include "runtime/pipeline_stage_materializer.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void verify_materialized_stage_contract(
    const PipelineStageDesc &stage, size_t materialized_index,
    const RuntimeExecutableDescriptor &descriptor) {
  OPENVINO_ASSERT(stage.stage,
                  "GFX: runtime execution plan stage ", materialized_index,
                  " materialized to a null backend stage");
  OPENVINO_ASSERT(stage.runtime_descriptor,
                  "GFX: runtime execution plan stage ", materialized_index,
                  " is missing a compiler-owned runtime descriptor");
  OPENVINO_ASSERT(stage.runtime_stage_index != PipelineStageDesc::npos,
                  "GFX: runtime execution plan stage ", materialized_index,
                  " has no descriptor stage index");
  OPENVINO_ASSERT(stage.runtime_stage_index < descriptor.stages.size(),
                  "GFX: runtime execution plan stage ", materialized_index,
                  " references descriptor stage ", stage.runtime_stage_index,
                  " outside descriptor stage table");

  const auto &stage_descriptor = *stage.runtime_descriptor;
  const auto &frozen_descriptor = descriptor.stages[stage.runtime_stage_index];
  OPENVINO_ASSERT(stage_descriptor.stage_index == frozen_descriptor.stage_index &&
                      stage_descriptor.stage_record_key ==
                          frozen_descriptor.stage_record_key &&
                      stage_descriptor.kernel_id == frozen_descriptor.kernel_id &&
                      stage_descriptor.abi_fingerprint ==
                          frozen_descriptor.abi_fingerprint,
                  "GFX: runtime execution plan stage ", materialized_index,
                  " descriptor drifted from compiler-owned executable "
                  "descriptor");
}

} // namespace

RuntimeExecutionPlan::RuntimeExecutionPlan(
    std::shared_ptr<const RuntimeExecutableDescriptor> descriptor,
    std::vector<PipelineStageDesc> stages)
    : m_descriptor(std::move(descriptor)), m_stages(std::move(stages)) {
  OPENVINO_ASSERT(m_descriptor,
                  "GFX: runtime execution plan descriptor is null");
  OPENVINO_ASSERT(m_descriptor->materialization_finalized,
                  "GFX: runtime execution plan requires finalized compiler "
                  "materialization contract");
  OPENVINO_ASSERT(
      m_stages.size() == m_descriptor->materialization_stages.size(),
      "GFX: runtime execution plan materialized stage count drift: plan=",
      m_stages.size(), " descriptor=", m_descriptor->materialization_stages.size());
  for (size_t i = 0; i < m_stages.size(); ++i) {
    verify_materialized_stage_contract(m_stages[i], i, *m_descriptor);
  }
}

std::shared_ptr<const RuntimeExecutionPlan>
RuntimeExecutionPlan::build(RuntimeExecutionPlanBuildRequest request) {
  OPENVINO_ASSERT(request.stage_factory,
                  "GFX: runtime execution plan requires backend stage factory");
  OPENVINO_ASSERT(request.runtime_descriptor,
                  "GFX: runtime execution plan requires runtime descriptor");

  PipelineStageRuntimeMaterializationRequest materialization_request;
  materialization_request.stage_factory = request.stage_factory;
  materialization_request.runtime_descriptor = request.runtime_descriptor.get();
  materialization_request.runtime_options = request.runtime_options;
  materialization_request.compile_trace = request.compile_trace;

  auto stages = materialize_pipeline_stage_descriptors(materialization_request);
  return std::shared_ptr<const RuntimeExecutionPlan>(
      new RuntimeExecutionPlan(std::move(request.runtime_descriptor),
                               std::move(stages)));
}

} // namespace gfx_plugin
} // namespace ov
