// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "compiler/operation_support.hpp"
#include "compiler/pipeline_stage_fusion.hpp"
#include "compiler/pipeline_stage_plan.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
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
  bool fusion_enabled = false;
  size_t graph_op_count = 0;

  bool valid() const noexcept { return !ordered_ops.empty(); }
};

PipelineStageGraphSnapshot make_pipeline_stage_graph_snapshot(
    const std::shared_ptr<const ov::Model> &model,
    const FusionConfig &fusion_config);

FusionConfig
make_pipeline_stage_fusion_config(const FusionCapabilities &fusion_capabilities,
                                  bool enable_fusion, bool debug_dump_ir);

} // namespace detail
} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
