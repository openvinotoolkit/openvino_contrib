// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>

#include "graph/mps_node_context.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/node_output.hpp"

namespace ov {
namespace metal_plugin {

NodeContext::NodeContext(MPSGraph* graph,
                         std::vector<std::unique_ptr<MetalNode>>& nodes,
                         std::unordered_map<const ov::Node*, MetalNode*>& node_map,
                         Layout default_layout)
    : m_graph(graph),
      m_nodes(nodes),
      m_node_map(node_map),
      m_default_layout(default_layout) {}

MetalNode* NodeContext::create_node(MetalOpType type,
                                    const std::shared_ptr<const ov::Node>& ov_node,
                                    const std::vector<MetalNode*>& deps) {
    auto node = std::make_unique<MetalNode>();
    node->index = m_nodes.size();
    node->op.type = type;
    node->op.friendly_name = ov_node->get_friendly_name();
    node->op.ov_node = ov_node;
    node->deps = deps;
    for (auto* d : node->deps) {
        d->users.push_back(node.get());
        node->op.input_indices.push_back(d->index);
    }
    MetalNode* raw = node.get();
    m_node_map[ov_node.get()] = raw;
    m_nodes.emplace_back(std::move(node));
    return raw;
}

MetalNode* NodeContext::get_node(const ov::Node* node) const {
    auto it = m_node_map.find(node);
    OPENVINO_ASSERT(it != m_node_map.end(), "NodeContext: node not found");
    return it->second;
}

Value NodeContext::get_input_value(const ov::Output<ov::Node>& out) const {
    auto* n = get_node(out.get_node());
    return Value{n->output_desc, n->mps_tensor};
}

void NodeContext::map_node(const std::shared_ptr<const ov::Node>& ov_node, MetalNode* target) {
    OPENVINO_ASSERT(ov_node && target, "map_node: invalid arguments");
    m_node_map[ov_node.get()] = target;
}

Value NodeContext::get_input_value(const ov::Node& node, size_t idx) const {
    return get_input_value(node.input(idx).get_source_output());
}

void NodeContext::require_rank(const Value& v, size_t rank, const std::string& what) const {
    if (v.desc.shape.size() != rank) {
        OPENVINO_THROW(what, ": expected rank ", rank, ", got ", v.desc.shape.size());
    }
}

void NodeContext::require_same_shape(const Value& a, const Value& b, const std::string& what) const {
    if (a.desc.shape != b.desc.shape) {
        OPENVINO_THROW(what, ": shape mismatch (", a.desc.shape, " vs ", b.desc.shape, ")");
    }
}

Value NodeContext::to_nhwc(const Value& v) const {
    if (v.desc.layout == Layout::NHWC) {
        return v;
    }
    require_rank(v, 4, "to_nhwc");
    NSArray<NSNumber*>* perm = @[ @0, @2, @3, @1 ];  // NCHW -> NHWC
    MPSGraphTensor* t = [m_graph transposeTensor:v.tensor permutation:perm name:nil];
    ov::Shape nhwc_shape{v.desc.shape[0], v.desc.shape[2], v.desc.shape[3], v.desc.shape[1]};
    Value out;
    out.desc.shape = nhwc_shape;
    out.desc.element_type = v.desc.element_type;
    out.desc.layout = Layout::NHWC;
    out.tensor = t;
    return out;
}

Value NodeContext::to_nchw(const Value& v) const {
    if (v.desc.layout == Layout::NCHW) {
        return v;
    }
    require_rank(v, 4, "to_nchw");
    NSArray<NSNumber*>* perm = @[ @0, @3, @1, @2 ];  // NHWC -> NCHW
    MPSGraphTensor* t = [m_graph transposeTensor:v.tensor permutation:perm name:nil];
    ov::Shape nchw_shape{v.desc.shape[0], v.desc.shape[3], v.desc.shape[1], v.desc.shape[2]};
    Value out;
    out.desc.shape = nchw_shape;
    out.desc.element_type = v.desc.element_type;
    out.desc.layout = Layout::NCHW;
    out.tensor = t;
    return out;
}

MPSShape* NodeContext::to_mps_shape(const ov::Shape& shape) {
    NSMutableArray<NSNumber*>* arr = [NSMutableArray arrayWithCapacity:shape.size()];
    for (auto dim : shape) {
        [arr addObject:@(dim)];
    }
    return [NSArray arrayWithArray:arr];
}

}  // namespace metal_plugin
}  // namespace ov
