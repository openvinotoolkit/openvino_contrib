// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <MetalPerformanceShadersGraph/MPSGraphActivationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphArithmeticOps.h>

#include "ops/gelu.hpp"
#include "ops/common.hpp"

namespace ov::metal_plugin::ops {
namespace {

MPSGraphTensor* make_scalar(NodeContext& ctx, double v, const ov::element::Type& et) {
    return [ctx.graph() constantWithScalar:v dataType:to_mps_type(et)];
}

MPSGraphTensor* tanh_via_sigmoid(NodeContext& ctx, MPSGraphTensor* x, const ov::element::Type& et) {
    auto two = make_scalar(ctx, 2.0, et);
    auto one = make_scalar(ctx, 1.0, et);
    auto two_x = [ctx.graph() multiplicationWithPrimaryTensor:x secondaryTensor:two name:nil];
    auto sig = [ctx.graph() sigmoidWithTensor:two_x name:nil];
    auto two_sig = [ctx.graph() multiplicationWithPrimaryTensor:two secondaryTensor:sig name:nil];
    return [ctx.graph() subtractionWithPrimaryTensor:two_sig secondaryTensor:one name:nil];
}

MPSGraphTensor* gelu_tanh(NodeContext& ctx, const Value& v) {
    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
    const double kBeta = 0.044715;
    const double kSqrt2OverPi = 0.7978845608028654;  // sqrt(2/pi)
    auto beta_c = make_scalar(ctx, kBeta, v.desc.element_type);
    auto sqrt_c = make_scalar(ctx, kSqrt2OverPi, v.desc.element_type);
    auto half = make_scalar(ctx, 0.5, v.desc.element_type);
    auto one = make_scalar(ctx, 1.0, v.desc.element_type);

    auto x2 = [ctx.graph() multiplicationWithPrimaryTensor:v.tensor secondaryTensor:v.tensor name:nil];
    auto x3 = [ctx.graph() multiplicationWithPrimaryTensor:x2 secondaryTensor:v.tensor name:nil];
    auto beta_x3 = [ctx.graph() multiplicationWithPrimaryTensor:beta_c secondaryTensor:x3 name:nil];
    auto inner = [ctx.graph() additionWithPrimaryTensor:v.tensor secondaryTensor:beta_x3 name:nil];
    auto scaled = [ctx.graph() multiplicationWithPrimaryTensor:sqrt_c secondaryTensor:inner name:nil];
    auto t = tanh_via_sigmoid(ctx, scaled, v.desc.element_type);
    auto one_plus_t = [ctx.graph() additionWithPrimaryTensor:one secondaryTensor:t name:nil];
    auto x_term = [ctx.graph() multiplicationWithPrimaryTensor:v.tensor secondaryTensor:one_plus_t name:nil];
    return [ctx.graph() multiplicationWithPrimaryTensor:half secondaryTensor:x_term name:nil];
}

}  // namespace

MetalNode* build_gelu(NodeContext& ctx, const ov::op::v7::Gelu& node) {
    auto* in_node = ctx.get_node(node.input_value(0).get_node());
    std::vector<MetalNode*> deps{in_node};
    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::Gelu, ov_ptr, deps);

    Value v = ctx.get_input_value(node, 0);
    // Both ERF and TANH approximations fallback to tanh-based formula for MPSGraph
    MPSGraphTensor* res = gelu_tanh(ctx, v);

    out->output_desc = v.desc;
    out->mps_tensor = res;
    return out;
}

MetalNode* build_gelu(NodeContext& ctx, const ov::op::v0::Gelu& node) {
    auto* in_node = ctx.get_node(node.input_value(0).get_node());
    std::vector<MetalNode*> deps{in_node};
    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::Gelu, ov_ptr, deps);

    Value v = ctx.get_input_value(node, 0);
    MPSGraphTensor* res = gelu_tanh(ctx, v);

    out->output_desc = v.desc;
    out->mps_tensor = res;
    return out;
}

}  // namespace ov::metal_plugin::ops
