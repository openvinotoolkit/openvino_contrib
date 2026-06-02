// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "compiler/pipeline_stage_plan.hpp"
#include "runtime/gpu_stage.hpp"
#include "runtime/output_lifetime.hpp"

namespace ov {
namespace gfx_plugin {

using OutputDesc = compiler::PipelineStageOutputDesc;

struct PipelineStageDesc : compiler::PipelineStageIoPlan {
  static constexpr size_t npos = compiler::PipelineStageIoPlan::npos;

  std::unique_ptr<GpuStage> stage; // runtime prototype; prepared per request
  using InputLink = compiler::PipelineStageInputLink;
  using OutputAlias = compiler::PipelineStageOutputAlias;
  using OutputLifetime = RuntimeOutputLifetime;
  std::vector<OutputLifetime> output_lifetimes;
};

} // namespace gfx_plugin
} // namespace ov
