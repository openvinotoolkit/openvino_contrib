// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/pipeline_stage_graph_snapshot.hpp"

#include <utility>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace detail {

PipelineStageGraphSnapshot make_pipeline_stage_graph_snapshot(
    const std::shared_ptr<const ov::Model> &model,
    const FusionConfig &fusion_config) {
  OPENVINO_ASSERT(model,
                  "GFX: pipeline stage graph snapshot requires compiler model");
  PipelineStageGraphSnapshot snapshot;
  snapshot.ordered_ops = model->get_ordered_ops();
  snapshot.graph_op_count = model->get_ops().size();
  snapshot.model_outputs = collect_model_output_ports(*model);
  snapshot.fusion_enabled = fusion_config.enable_fusion;
  snapshot.param_index.reserve(model->inputs().size());
  for (size_t i = 0; i < model->inputs().size(); ++i) {
    snapshot.param_index[model->inputs()[i].get_node()] = i;
  }
  const auto &model_results = model->get_results();
  snapshot.public_outputs.reserve(model_results.size());
  for (const auto &model_result : model_results) {
    const auto source = model_result->input_value(0);
    PipelineStagePublicOutputSource public_output;
    public_output.node = source.get_node_shared_ptr();
    public_output.port = source.get_index();
    if (source.get_partial_shape().is_static()) {
      public_output.shape = source.get_shape();
    }
    public_output.type = source.get_element_type();
    snapshot.public_outputs.push_back(std::move(public_output));
  }
  snapshot.fusion_plan = build_fusion_plan(model, fusion_config);
  return snapshot;
}

FusionConfig
make_pipeline_stage_fusion_config(const FusionCapabilities &fusion_capabilities,
                                  bool enable_fusion, bool debug_dump_ir) {
  FusionConfig fusion_config;
  fusion_config.enable_fusion = enable_fusion;
  fusion_config.debug_dump_ir = debug_dump_ir;
  fusion_config.enable_attention_fusion =
      fusion_capabilities.enable_generic_attention_fusion;
  fusion_config.enable_vendor_attention_fusion =
      fusion_capabilities.supports_vendor_attention_stage;
  fusion_config.enable_conv_activation_fusion =
      fusion_capabilities.enable_conv_activation_fusion;
  fusion_config.enable_conv_swish_fusion = true;
  return fusion_config;
}

} // namespace detail
} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
