// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

using ModelOutputPorts =
    std::unordered_map<const ov::Node *, std::vector<bool>>;

struct PipelineOutputPortKey {
  const ov::Node *node = nullptr;
  size_t port = 0;

  bool operator==(const PipelineOutputPortKey &other) const {
    return node == other.node && port == other.port;
  }
};

struct PipelineOutputPortKeyHash {
  size_t operator()(const PipelineOutputPortKey &key) const;
};

using PipelineOutputAliasMap =
    std::unordered_map<PipelineOutputPortKey, size_t,
                       PipelineOutputPortKeyHash>;

struct PipelineStageInputLink {
  std::shared_ptr<const ov::Node> node;
  size_t port = 0;
};

struct PipelineStageOutputAlias {
  std::shared_ptr<const ov::Node> node;
  size_t source_port = 0;
  size_t output_port = 0;
};

struct PipelineStageOutputDesc {
  ov::Shape shape;
  ov::element::Type type = ov::element::dynamic;
  bool is_model_output = false;
  std::shared_ptr<const ov::Node> source_node;
  size_t source_port = 0;
  std::string direct_stateful_assign_variable_id;
};

struct PipelineStageIoPlan {
  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  std::shared_ptr<const ov::Node> node;
  size_t runtime_stage_index = npos;
  std::vector<PipelineStageOutputDesc> outputs;
  std::vector<PipelineStageInputLink> inputs;
  std::vector<PipelineStageOutputAlias> output_aliases;
};

struct PipelineInputTransformPlan {
  ov::Shape source_shape;
  std::vector<int64_t> transpose_permutation;

  bool has_transpose() const {
    return !source_shape.empty() && !transpose_permutation.empty();
  }
};

using PipelineInputTransformMap =
    std::unordered_map<const ov::Node *,
                       std::unordered_map<size_t, PipelineInputTransformPlan>>;

struct PipelineAbsorbedInputTransformPlan {
  PipelineInputTransformMap input_transforms;
  std::unordered_set<const ov::Node *> absorbed_nodes;
};

ModelOutputPorts collect_model_output_ports(const ov::Model &model);

bool is_model_output_port(const ModelOutputPorts &model_outputs,
                          const ov::Node *node, size_t port);

bool is_executable_stage_node(const std::shared_ptr<ov::Node> &node);

std::string direct_stateful_assign_variable_id(
    const std::shared_ptr<const ov::Node> &node, size_t output_idx);

PipelineAbsorbedInputTransformPlan plan_absorbed_input_transforms(
    const std::vector<std::shared_ptr<ov::Node>> &ordered_ops,
    const ModelOutputPorts &model_outputs,
    const std::unordered_set<const ov::Node *> &fused_nodes);

class PipelineStagePlanBuilder {
public:
  explicit PipelineStagePlanBuilder(const ModelOutputPorts &model_outputs);

  PipelineStageIoPlan
  make_stage_plan(const std::shared_ptr<const ov::Node> &node,
                  size_t runtime_stage_index = PipelineStageIoPlan::npos) const;

  PipelineStageIoPlan make_fused_stage_plan(
      const std::shared_ptr<const ov::Node> &final_node, size_t output_count,
      size_t runtime_stage_index = PipelineStageIoPlan::npos) const;

  PipelineStageInputLink
  remap_input_link(const PipelineOutputAliasMap &aliases,
                   std::shared_ptr<const ov::Node> linked_node,
                   size_t linked_port) const;

  void merge_model_outputs(PipelineStageIoPlan &stage_plan,
                           const ov::Node *node) const;

  void append_output_alias(PipelineStageIoPlan &stage_plan,
                           const std::shared_ptr<const ov::Node> &source_node,
                           size_t source_port, size_t output_port) const;

  void describe_output(PipelineStageIoPlan &stage_plan, size_t output_slot,
                       const std::shared_ptr<const ov::Node> &source_node,
                       size_t source_port) const;

  void record_output_alias(PipelineOutputAliasMap &aliases,
                           const ov::Node *source_node, size_t source_port,
                           size_t output_port) const;

private:
  const ModelOutputPorts &m_model_outputs;
};

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
