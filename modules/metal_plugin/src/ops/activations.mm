// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <MetalPerformanceShadersGraph/MPSGraphActivationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphArithmeticOps.h>

#include "ops/activations.hpp"
#include "ops/common.hpp"

namespace ov::metal_plugin::ops {
namespace {

MPSGraphTensor* make_scalar(NodeContext& ctx, double v, const ov::element::Type& et) {
    return [ctx.graph() constantWithScalar:v dataType:to_mps_type(et)];
}

// Fallback tanh using exp if native op is unavailable in headers.
MPSGraphTensor* tanh_via_sigmoid(NodeContext& ctx, MPSGraphTensor* x, const ov::element::Type& et) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    auto two = make_scalar(ctx, 2.0, et);
    auto one = make_scalar(ctx, 1.0, et);
    auto two_x = [ctx.graph() multiplicationWithPrimaryTensor:x secondaryTensor:two name:nil];
    auto sig = [ctx.graph() sigmoidWithTensor:two_x name:nil];
    auto two_sig = [ctx.graph() multiplicationWithPrimaryTensor:two secondaryTensor:sig name:nil];
    return [ctx.graph() subtractionWithPrimaryTensor:two_sig secondaryTensor:one name:nil];
}

}  // namespace

MetalNode* build_tanh(NodeContext& ctx, const ov::op::v0::Tanh& node) {
    auto* in_node = ctx.get_node(node.input_value(0).get_node());
    std::vector<MetalNode*> deps{in_node};
    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::Tanh, ov_ptr, deps);

    Value v = ctx.get_input_value(node, 0);
    // No direct tanh API in headers found; build via sigmoid for numerical stability
    MPSGraphTensor* res = tanh_via_sigmoid(ctx, v.tensor, v.desc.element_type);

    out->output_desc = v.desc;
    out->mps_tensor = res;
    return out;
}

MetalNode* build_sigmoid(NodeContext& ctx, const ov::op::v0::Sigmoid& node) {
    auto* in_node = ctx.get_node(node.input_value(0).get_node());
    std::vector<MetalNode*> deps{in_node};
    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::Sigmoid, ov_ptr, deps);

    Value v = ctx.get_input_value(node, 0);
    MPSGraphTensor* res = [ctx.graph() sigmoidWithTensor:v.tensor name:nil];

    out->output_desc = v.desc;
    out->mps_tensor = res;
    return out;
}

MetalNode* build_elu(NodeContext& ctx, const ov::op::v0::Elu& node) {
    auto* in_node = ctx.get_node(node.input_value(0).get_node());
    std::vector<MetalNode*> deps{in_node};
    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::Elu, ov_ptr, deps);

    Value v = ctx.get_input_value(node, 0);
    const double alpha = node.get_alpha();
    auto zero = make_scalar(ctx, 0.0, v.desc.element_type);
    auto one = make_scalar(ctx, 1.0, v.desc.element_type);
    auto alpha_c = make_scalar(ctx, alpha, v.desc.element_type);

    // pos = max(x, 0)
    auto pos = [ctx.graph() maximumWithPrimaryTensor:v.tensor secondaryTensor:zero name:nil];
    // neg = min(x, 0)
    auto neg = [ctx.graph() minimumWithPrimaryTensor:v.tensor secondaryTensor:zero name:nil];
    auto exp_neg = [ctx.graph() exponentWithTensor:neg name:nil];
    auto elu_neg = [ctx.graph() subtractionWithPrimaryTensor:exp_neg secondaryTensor:one name:nil];
    auto scaled_neg = [ctx.graph() multiplicationWithPrimaryTensor:alpha_c secondaryTensor:elu_neg name:nil];
    auto res = [ctx.graph() additionWithPrimaryTensor:pos secondaryTensor:scaled_neg name:nil];

    out->output_desc = v.desc;
    out->mps_tensor = res;
    return out;
}

MetalNode* build_leaky_relu(NodeContext& ctx, const ov::op::v0::PRelu& node) {
    auto* in_node = ctx.get_node(node.input_value(0).get_node());
    auto* slope_node = ctx.get_node(node.input_value(1).get_node());
    std::vector<MetalNode*> deps{in_node, slope_node};
    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::LeakyRelu, ov_ptr, deps);

    Value x = ctx.get_input_value(node, 0);
    Value slope = ctx.get_input_value(node, 1);

    // Support scalar slope only for now
    if (!(slope.desc.shape.empty() || (slope.desc.shape.size() == 1 && slope.desc.shape[0] == 1))) {
        OPENVINO_THROW("LeakyReLU (PRelu): only scalar slope is supported");
    }

    MPSGraphTensor* slope_tensor = slope.tensor;
    if (!slope.desc.shape.empty()) {
        slope_tensor = [ctx.graph() reshapeTensor:slope.tensor
                                       withShape:NodeContext::to_mps_shape({1})
                                            name:nil];
    }

    MPSGraphTensor* res = [ctx.graph() leakyReLUWithTensor:x.tensor alphaTensor:slope_tensor name:nil];

    out->output_desc = x.desc;
    out->mps_tensor = res;
    return out;
}

}  // namespace ov::metal_plugin::ops
