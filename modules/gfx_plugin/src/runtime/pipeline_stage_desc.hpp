// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/output_lifetime.hpp"
#include "runtime/pipeline_stage_plan.hpp"

namespace ov {
namespace gfx_plugin {

using OutputDesc = PipelineStageOutputDesc;

struct PipelineStageDesc : PipelineStageIoPlan {
  static constexpr size_t npos = PipelineStageIoPlan::npos;

  std::unique_ptr<GpuStage> stage; // runtime prototype; prepared per request
  using InputLink = PipelineStageInputLink;
  using OutputAlias = PipelineStageOutputAlias;
  using OutputLifetime = RuntimeOutputLifetime;
  std::shared_ptr<const RuntimeStageExecutableDescriptor> runtime_descriptor;
  std::vector<OutputLifetime> output_lifetimes;
};

} // namespace gfx_plugin
} // namespace ov
