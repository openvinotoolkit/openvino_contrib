// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ops/layernorm.hpp"

#if OV_LAYER_NORM_AVAILABLE

#import <MetalPerformanceShadersGraph/MPSGraphNormalizationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphArithmeticOps.h>

#include "ops/common.hpp"

namespace ov::metal_plugin::ops {
namespace {
NSArray<NSNumber*>* make_axes(const std::vector<size_t>& axes) {
    NSMutableArray<NSNumber*>* arr = [NSMutableArray arrayWithCapacity:axes.size()];
    for (size_t a : axes) {
        [arr addObject:@(static_cast<NSInteger>(a))];
    }
    return arr;
}
}  // namespace

MetalNode* build_layer_norm(NodeContext& ctx, const ov::op::v12::LayerNorm& node) {
    auto* in_node = ctx.get_node(node.input_value(0).get_node());
    auto* gamma_node = ctx.get_node(node.input_value(1).get_node());
    auto* beta_node = ctx.get_node(node.input_value(2).get_node());
    std::vector<MetalNode*> deps{in_node, gamma_node, beta_node};
    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::LayerNorm, ov_ptr, deps);

    Value x = ctx.get_input_value(node, 0);
    Value gamma = ctx.get_input_value(node, 1);
    Value beta = ctx.get_input_value(node, 2);

    const auto rank = x.desc.shape.size();
    int64_t axis = node.get_axis();
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
        OPENVINO_THROW("LayerNorm: axis out of range");
    }

    // normalize over [axis, axis+1, ..., rank-1]
    std::vector<size_t> norm_axes;
    for (size_t i = static_cast<size_t>(axis); i < rank; ++i) {
        norm_axes.push_back(i);
    }
    auto axes_arr = make_axes(norm_axes);

    MPSGraphTensor* mean = [ctx.graph() meanOfTensor:x.tensor axes:axes_arr name:nil];
    MPSGraphTensor* var = [ctx.graph() varianceOfTensor:x.tensor meanTensor:mean axes:axes_arr name:nil];
    float eps = static_cast<float>(node.get_eps());
    MPSGraphTensor* y = [ctx.graph() normalizationWithTensor:x.tensor
                                                   meanTensor:mean
                                               varianceTensor:var
                                                  gammaTensor:gamma.tensor
                                                   betaTensor:beta.tensor
                                                      epsilon:eps
                                                         name:nil];

    out->output_desc = x.desc;
    out->mps_tensor = y;
    return out;
}

}  // namespace ov::metal_plugin::ops

#endif  // OV_LAYER_NORM_AVAILABLE
