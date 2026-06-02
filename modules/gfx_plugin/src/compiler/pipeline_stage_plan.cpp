// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/pipeline_stage_plan.hpp"

#include <algorithm>
#include <functional>

namespace ov {
namespace gfx_plugin {
namespace compiler {

size_t PipelineOutputPortKeyHash::operator()(
    const PipelineOutputPortKey &key) const {
  size_t h1 = std::hash<const ov::Node *>()(key.node);
  size_t h2 = std::hash<size_t>()(key.port);
  return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
}

ModelOutputPorts collect_model_output_ports(const ov::Model &model) {
  ModelOutputPorts model_outputs;
  for (const auto &result : model.get_results()) {
    auto src = result->input_value(0).get_node_shared_ptr();
    const size_t port = result->input_value(0).get_index();
    auto &flags = model_outputs[src.get()];
    if (flags.empty()) {
      flags.resize(src->get_output_size(), false);
    }
    if (port < flags.size()) {
      flags[port] = true;
    }
  }
  return model_outputs;
}

bool is_model_output_port(const ModelOutputPorts &model_outputs,
                          const ov::Node *node, size_t port) {
  auto it = model_outputs.find(node);
  if (it == model_outputs.end() || port >= it->second.size()) {
    return false;
  }
  return it->second[port];
}

PipelineStagePlanBuilder::PipelineStagePlanBuilder(
    const ModelOutputPorts &model_outputs)
    : m_model_outputs(model_outputs) {}

PipelineStageIoPlan PipelineStagePlanBuilder::make_stage_plan(
    const std::shared_ptr<const ov::Node> &node,
    size_t runtime_stage_index) const {
  PipelineStageIoPlan stage_plan;
  stage_plan.node = node;
  stage_plan.runtime_stage_index = runtime_stage_index;
  if (!node) {
    return stage_plan;
  }

  const size_t out_count = node->get_output_size();
  stage_plan.outputs.reserve(out_count);
  for (size_t oi = 0; oi < out_count; ++oi) {
    describe_output(stage_plan, oi, node, oi);
  }
  return stage_plan;
}

PipelineStageIoPlan PipelineStagePlanBuilder::make_fused_stage_plan(
    const std::shared_ptr<const ov::Node> &final_node, size_t output_count,
    size_t runtime_stage_index) const {
  PipelineStageIoPlan stage_plan;
  stage_plan.node = final_node;
  stage_plan.runtime_stage_index = runtime_stage_index;
  stage_plan.outputs.resize(output_count);
  return stage_plan;
}

PipelineStageInputLink PipelineStagePlanBuilder::remap_input_link(
    const PipelineOutputAliasMap &aliases,
    std::shared_ptr<const ov::Node> linked_node, size_t linked_port) const {
  if (linked_node) {
    auto it = aliases.find({linked_node.get(), linked_port});
    if (it != aliases.end()) {
      linked_port = it->second;
    }
  }
  return PipelineStageInputLink{std::move(linked_node), linked_port};
}

void PipelineStagePlanBuilder::merge_model_outputs(
    PipelineStageIoPlan &stage_plan, const ov::Node *node) const {
  auto it = m_model_outputs.find(node);
  if (it == m_model_outputs.end()) {
    return;
  }
  const auto &flags = it->second;
  for (size_t oi = 0; oi < stage_plan.outputs.size() && oi < flags.size();
       ++oi) {
    stage_plan.outputs[oi].is_model_output =
        stage_plan.outputs[oi].is_model_output || flags[oi];
  }
}

void PipelineStagePlanBuilder::append_output_alias(
    PipelineStageIoPlan &stage_plan,
    const std::shared_ptr<const ov::Node> &source_node, size_t source_port,
    size_t output_port) const {
  if (!source_node || output_port >= stage_plan.outputs.size()) {
    return;
  }
  const auto duplicate = std::any_of(
      stage_plan.output_aliases.begin(), stage_plan.output_aliases.end(),
      [&](const PipelineStageOutputAlias &alias) {
        return alias.node.get() == source_node.get() &&
               alias.source_port == source_port &&
               alias.output_port == output_port;
      });
  if (!duplicate) {
    stage_plan.output_aliases.push_back(
        {source_node, source_port, output_port});
  }
}

void PipelineStagePlanBuilder::describe_output(
    PipelineStageIoPlan &stage_plan, size_t output_slot,
    const std::shared_ptr<const ov::Node> &source_node,
    size_t source_port) const {
  if (output_slot >= stage_plan.outputs.size()) {
    stage_plan.outputs.resize(output_slot + 1);
  }
  auto &out_desc = stage_plan.outputs[output_slot];
  if (!source_node || source_port >= source_node->get_output_size()) {
    return;
  }
  if (source_node->get_output_partial_shape(source_port).is_static()) {
    out_desc.shape = source_node->get_output_shape(source_port);
  }
  out_desc.type = source_node->get_output_element_type(source_port);
  out_desc.is_model_output =
      is_model_output_port(m_model_outputs, source_node.get(), source_port);
  out_desc.source_node = source_node;
  out_desc.source_port = source_port;
}

void PipelineStagePlanBuilder::record_output_alias(
    PipelineOutputAliasMap &aliases, const ov::Node *source_node,
    size_t source_port, size_t output_port) const {
  if (!source_node) {
    return;
  }
  aliases[{source_node, source_port}] = output_port;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
