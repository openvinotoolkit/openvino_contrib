// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "compiler/backend_target.hpp"
#include "compiler/pipeline_stage_graph_snapshot.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfilingTrace;
struct RuntimeExecutableDescriptor;

namespace compiler {

class BackendRegistry;

namespace detail {

struct PipelineStageBuildRequest {
  PipelineStageGraphSnapshot graph;
  const RuntimeExecutableDescriptor *runtime_descriptor = nullptr;
  const BackendRegistry *backend_registry = nullptr;
  BackendTarget target;
  std::string backend_name;
  GfxProfilingTrace *compile_trace = nullptr;
};

RuntimeExecutableDescriptor build_pipeline_stage_runtime_descriptor(
    const PipelineStageBuildRequest &request);

} // namespace detail
} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
