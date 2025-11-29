// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "ops/elementwise.hpp"

namespace ov::metal_plugin::ops {

MetalNode* build_add(NodeContext& ctx, const ov::op::v1::Add& node) {
    auto* left_node = ctx.get_node(node.input_value(0).get_node());
    auto* right_node = ctx.get_node(node.input_value(1).get_node());
    std::vector<MetalNode*> deps{left_node, right_node};

    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::Add, ov_ptr, deps);

    Value a = ctx.get_input_value(node, 0);
    Value b = ctx.get_input_value(node, 1);
    // Basic numpy-style broadcast
    const size_t max_rank = std::max(a.desc.shape.size(), b.desc.shape.size());
    if (max_rank > 4) {
        OPENVINO_THROW("Add: unsupported rank for broadcasting");
    }
    ov::Shape shape_a = a.desc.shape;
    ov::Shape shape_b = b.desc.shape;
    ov::Shape shape_a_pad(max_rank, 1), shape_b_pad(max_rank, 1);
    // pad leading ones
    std::copy(shape_a.begin(), shape_a.end(), shape_a_pad.begin() + (max_rank - shape_a.size()));
    std::copy(shape_b.begin(), shape_b.end(), shape_b_pad.begin() + (max_rank - shape_b.size()));

    ov::Shape out_shape(max_rank);
    for (size_t i = 0; i < max_rank; ++i) {
        if (shape_a_pad[i] == shape_b_pad[i]) {
            out_shape[i] = shape_a_pad[i];
        } else if (shape_a_pad[i] == 1) {
            out_shape[i] = shape_b_pad[i];
        } else if (shape_b_pad[i] == 1) {
            out_shape[i] = shape_a_pad[i];
        } else {
            OPENVINO_THROW("Add: unsupported broadcast pattern (", shape_a, " vs ", shape_b, ")");
        }
    }

    auto maybe_broadcast = [&](const Value& v, const ov::Shape& target) -> MPSGraphTensor* {
        if (v.desc.shape == target) {
            return v.tensor;
        }
        MPSShape* tgt = NodeContext::to_mps_shape(target);
        return [ctx.graph() broadcastTensor:v.tensor toShape:tgt name:nil];
    };

    MPSGraphTensor* ta = maybe_broadcast(a, out_shape);
    MPSGraphTensor* tb = maybe_broadcast(b, out_shape);

    MPSGraphTensor* res =
        [ctx.graph() additionWithPrimaryTensor:ta secondaryTensor:tb name:nil];

    out->output_desc.shape = out_shape;
    out->output_desc.element_type = a.desc.element_type;
    out->output_desc.layout = a.desc.layout;
    out->mps_tensor = res;
    return out;
}

MetalNode* build_relu(NodeContext& ctx, const ov::op::v0::Relu& node) {
    auto* in_node = ctx.get_node(node.input_value(0).get_node());
    std::vector<MetalNode*> deps{in_node};

    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::Relu, ov_ptr, deps);

    Value v = ctx.get_input_value(node, 0);

    MPSGraphTensor* res = [ctx.graph() reLUWithTensor:v.tensor name:nil];

    out->output_desc = v.desc;
    out->mps_tensor = res;
    return out;
}

}  // namespace ov::metal_plugin::ops
