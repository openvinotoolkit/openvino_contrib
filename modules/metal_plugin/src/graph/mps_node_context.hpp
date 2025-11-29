// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "graph/metal_node.hpp"

namespace ov {
namespace metal_plugin {

class NodeContext {
public:
    NodeContext(MPSGraph* graph,
                std::vector<std::unique_ptr<MetalNode>>& nodes,
                std::unordered_map<const ov::Node*, MetalNode*>& node_map,
                Layout default_layout = Layout::NCHW);

    MPSGraph* graph() const { return m_graph; }

    MetalNode* create_node(MetalOpType type,
                           const std::shared_ptr<const ov::Node>& ov_node,
                           const std::vector<MetalNode*>& deps);

    MetalNode* get_node(const ov::Node* node) const;

    Value get_input_value(const ov::Output<ov::Node>& out) const;
    Value get_input_value(const ov::Node& node, size_t idx) const;

    // Map an OpenVINO node to an existing MetalNode (used for fused patterns).
    void map_node(const std::shared_ptr<const ov::Node>& ov_node, MetalNode* target);

    void require_rank(const Value& v, size_t rank, const std::string& what) const;
    void require_same_shape(const Value& a, const Value& b, const std::string& what) const;

    Value to_nhwc(const Value& v) const;
    Value to_nchw(const Value& v) const;

    static MPSShape* to_mps_shape(const ov::Shape& shape);

private:
    MPSGraph* m_graph;
    std::vector<std::unique_ptr<MetalNode>>& m_nodes;
    std::unordered_map<const ov::Node*, MetalNode*>& m_node_map;
    Layout m_default_layout;
};

}  // namespace metal_plugin
}  // namespace ov
