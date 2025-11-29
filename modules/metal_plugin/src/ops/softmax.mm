// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <MetalPerformanceShadersGraph/MPSGraphActivationOps.h>

#include "ops/softmax.hpp"

namespace ov::metal_plugin::ops {

namespace {

MetalNode* build_softmax_impl(NodeContext& ctx,
                              const std::shared_ptr<const ov::Node>& softmax_node,
                              int64_t axis_raw) {
    auto input = softmax_node->input_value(0);
    auto* in_node = ctx.get_node(input.get_node());
    std::vector<MetalNode*> deps{in_node};
    MetalNode* out = ctx.create_node(MetalOpType::Softmax, softmax_node, deps);

    Value v = ctx.get_input_value(*softmax_node, 0);
    auto rank = v.desc.shape.size();
    int64_t axis = axis_raw;
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
        OPENVINO_THROW("Softmax: axis out of range");
    }

    MPSGraphTensor* res = [ctx.graph() softMaxWithTensor:v.tensor axis:(NSInteger)axis name:nil];

    out->output_desc = v.desc;
    out->mps_tensor = res;
    return out;
}

}  // namespace

MetalNode* build_softmax(NodeContext& ctx, const ov::op::v1::Softmax& node) {
    return build_softmax_impl(ctx, node.shared_from_this(), static_cast<int64_t>(node.get_axis()));
}

MetalNode* build_softmax(NodeContext& ctx, const ov::op::v8::Softmax& node) {
    return build_softmax_impl(ctx, node.shared_from_this(), node.get_axis());
}

}  // namespace ov::metal_plugin::ops
