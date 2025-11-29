// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <MetalPerformanceShadersGraph/MPSGraphNormalizationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>

#include "ops/batchnorm.hpp"
#include "ops/common.hpp"

#include "openvino/core/except.hpp"

namespace ov::metal_plugin::ops {
namespace {

template <typename TBatchNorm>
MetalNode* build_batch_norm_impl(NodeContext& ctx, const TBatchNorm& node) {
    // Inputs: data, gamma, beta, mean, variance
    auto* data_node = ctx.get_node(node.input_value(0).get_node());
    auto* gamma_node = ctx.get_node(node.input_value(1).get_node());
    auto* beta_node = ctx.get_node(node.input_value(2).get_node());
    auto* mean_node = ctx.get_node(node.input_value(3).get_node());
    auto* var_node = ctx.get_node(node.input_value(4).get_node());
    std::vector<MetalNode*> deps{data_node, gamma_node, beta_node, mean_node, var_node};

    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::BatchNorm, ov_ptr, deps);

    Value x = ctx.get_input_value(node, 0);
    Value gamma = ctx.get_input_value(node, 1);
    Value beta = ctx.get_input_value(node, 2);
    Value mean = ctx.get_input_value(node, 3);
    Value var = ctx.get_input_value(node, 4);

    ctx.require_rank(x, 4, "BatchNormInference");
    const size_t C = x.desc.shape[1];
    auto require_vec_c = [&](const Value& v, const char* name) {
        if (v.desc.shape.size() != 1 || v.desc.shape[0] != C) {
            OPENVINO_THROW("BatchNormInference: expected ", name, " shape [", C, "], got ", v.desc.shape);
        }
    };
    require_vec_c(gamma, "gamma");
    require_vec_c(beta, "beta");
    require_vec_c(mean, "mean");
    require_vec_c(var, "variance");

    // Convert input to NHWC for MPSGraph convenience
    Value x_nhwc = ctx.to_nhwc(x);
    const ov::Shape nhwc_shape = x_nhwc.desc.shape;  // [N, H, W, C]

    MPSShape* param_shape = NodeContext::to_mps_shape({1, 1, 1, C});
    auto reshape_param = [&](const Value& v) -> MPSGraphTensor* {
        return [ctx.graph() reshapeTensor:v.tensor withShape:param_shape name:nil];
    };

    MPSGraphTensor* gamma_r = reshape_param(gamma);
    MPSGraphTensor* beta_r = reshape_param(beta);
    MPSGraphTensor* mean_r = reshape_param(mean);
    MPSGraphTensor* var_r = reshape_param(var);

    float eps = static_cast<float>(node.get_eps_value());
    MPSGraphTensor* y_nhwc = [ctx.graph() normalizationWithTensor:x_nhwc.tensor
                                                       meanTensor:mean_r
                                                   varianceTensor:var_r
                                                      gammaTensor:gamma_r
                                                       betaTensor:beta_r
                                                          epsilon:eps
                                                             name:nil];

    // Convert back to NCHW to match external contract
    Value y_value;
    y_value.desc.shape = nhwc_shape;  // current tensor is NHWC
    y_value.desc.element_type = x.desc.element_type;
    y_value.desc.layout = Layout::NHWC;
    y_value.tensor = y_nhwc;
    Value y_nchw = ctx.to_nchw(y_value);

    out->output_desc = y_nchw.desc;
    out->mps_tensor = y_nchw.tensor;
    return out;
}

}  // namespace

MetalNode* build_batch_norm(NodeContext& ctx, const ov::op::v5::BatchNormInference& node) {
    return build_batch_norm_impl(ctx, node);
}

MetalNode* build_batch_norm(NodeContext& ctx, const ov::op::v0::BatchNormInference& node) {
    return build_batch_norm_impl(ctx, node);
}

}  // namespace ov::metal_plugin::ops
