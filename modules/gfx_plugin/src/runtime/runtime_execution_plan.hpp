// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/pipeline_stage_desc.hpp"

namespace ov {
namespace gfx_plugin {

class BackendStageFactory;
class GfxProfilingTrace;

struct RuntimeExecutionPlanBuildRequest {
  const BackendStageFactory *stage_factory = nullptr;
  std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor;
  GpuStageRuntimeOptions runtime_options;
  GfxProfilingTrace *compile_trace = nullptr;
};

class RuntimeExecutionPlan final {
public:
  static std::shared_ptr<const RuntimeExecutionPlan>
  build(RuntimeExecutionPlanBuildRequest request);

  const RuntimeExecutableDescriptor &descriptor() const noexcept {
    return *m_descriptor;
  }

  std::shared_ptr<const RuntimeExecutableDescriptor> descriptor_ptr()
      const noexcept {
    return m_descriptor;
  }

  const std::vector<PipelineStageDesc> &stages() const noexcept {
    return m_stages;
  }

  size_t stage_count() const noexcept { return m_stages.size(); }
  bool empty() const noexcept { return m_stages.empty(); }

private:
  RuntimeExecutionPlan(
      std::shared_ptr<const RuntimeExecutableDescriptor> descriptor,
      std::vector<PipelineStageDesc> stages);

  std::shared_ptr<const RuntimeExecutableDescriptor> m_descriptor;
  std::vector<PipelineStageDesc> m_stages;
};

} // namespace gfx_plugin
} // namespace ov
