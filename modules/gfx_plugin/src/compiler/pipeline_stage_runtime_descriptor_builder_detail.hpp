// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "compiler/backend_target.hpp"
#include "compiler/pipeline_stage_builder.hpp"
#include "compiler/pipeline_stage_fusion.hpp"
#include "compiler/pipeline_stage_plan.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfilingTrace;
struct RuntimeExecutableDescriptor;

namespace compiler {

class BackendRegistry;

namespace detail {

struct PipelineStagePublicOutputSource {
  std::shared_ptr<const ov::Node> node;
  size_t port = 0;
  ov::Shape shape;
  ov::element::Type type = ov::element::dynamic;
};

struct PipelineStageGraphSnapshot {
  std::vector<std::shared_ptr<ov::Node>> ordered_ops;
  ModelOutputPorts model_outputs;
  std::unordered_map<const ov::Node *, size_t> param_index;
  std::vector<PipelineStagePublicOutputSource> public_outputs;
  FusionPlan fusion_plan;
  size_t graph_op_count = 0;

  bool valid() const noexcept { return !ordered_ops.empty(); }
};

struct PipelineStageBuildRequest {
  PipelineStageGraphSnapshot graph;
  const RuntimeExecutableDescriptor *runtime_descriptor = nullptr;
  const BackendRegistry *backend_registry = nullptr;
  BackendTarget target;
  std::string backend_name;
  bool enable_fusion = true;
  GfxProfilingTrace *compile_trace = nullptr;
};

PipelineStageGraphSnapshot
make_pipeline_stage_graph_snapshot(const std::shared_ptr<const ov::Model> &model,
                                   const FusionConfig &fusion_config);

FusionConfig make_pipeline_stage_fusion_config(
    const FusionCapabilities &fusion_capabilities, bool enable_fusion,
    bool debug_dump_ir);

RuntimeExecutableDescriptor
build_pipeline_stage_runtime_descriptor(
    const PipelineStageBuildRequest &request);

} // namespace detail
} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
