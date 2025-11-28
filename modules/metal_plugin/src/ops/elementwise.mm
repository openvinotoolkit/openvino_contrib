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
    ctx.require_same_shape(a, b, "Add");

    MPSGraphTensor* res =
        [ctx.graph() additionWithPrimaryTensor:a.tensor secondaryTensor:b.tensor name:nil];

    out->output_desc = a.desc;
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
