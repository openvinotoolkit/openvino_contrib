// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>

#include "ops/matmul.hpp"

namespace ov::metal_plugin::ops {

MetalNode* build_matmul(NodeContext& ctx, const ov::op::v0::MatMul& node) {
    auto* left_node = ctx.get_node(node.input_value(0).get_node());
    auto* right_node = ctx.get_node(node.input_value(1).get_node());
    std::vector<MetalNode*> deps{left_node, right_node};

    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::MatMul, ov_ptr, deps);

    Value a = ctx.get_input_value(node, 0);
    Value b = ctx.get_input_value(node, 1);
    auto rank_a = a.desc.shape.size();
    auto rank_b = b.desc.shape.size();
    if (!((rank_a == 2 && rank_b == 2) || (rank_a == 3 && rank_b == 3))) {
        OPENVINO_THROW("MatMul: only 2D or 3D batched matmul is supported");
    }
    if (node.get_transpose_a() || node.get_transpose_b()) {
        OPENVINO_THROW("MatMul: transpose flags not supported in current MPSGraph path");
    }

    MPSGraphTensor* a_t = a.tensor;
    MPSGraphTensor* b_t = b.tensor;
    ov::Shape out_shape;

    if (rank_a == 2) {
        // simple 2D
        out_shape = node.get_output_shape(0);
    } else {
        size_t batch_a = a.desc.shape[0];
        size_t batch_b = b.desc.shape[0];
        size_t M = a.desc.shape[1];
        size_t K = a.desc.shape[2];
        size_t N = b.desc.shape[2];

        size_t batch_out = 0;
        if (batch_a == batch_b) {
            batch_out = batch_a;
        } else if (batch_a == 1) {
            batch_out = batch_b;
            MPSShape* shape_a = NodeContext::to_mps_shape({batch_out, M, K});
            a_t = [ctx.graph() broadcastTensor:a_t toShape:shape_a name:nil];
        } else if (batch_b == 1) {
            batch_out = batch_a;
            MPSShape* shape_b = NodeContext::to_mps_shape({batch_out, K, N});
            b_t = [ctx.graph() broadcastTensor:b_t toShape:shape_b name:nil];
        } else {
            OPENVINO_THROW("MatMul: unsupported batch broadcasting (", batch_a, " vs ", batch_b, ")");
        }
        out_shape = {batch_out, M, N};
    }

    MPSGraphTensor* res =
        [ctx.graph() matrixMultiplicationWithPrimaryTensor:a_t secondaryTensor:b_t name:nil];

    out->output_desc.shape = out_shape;
    out->output_desc.element_type = node.get_output_element_type(0);
    out->output_desc.layout = Layout::NCHW;
    out->mps_tensor = res;
    return out;
}

}  // namespace ov::metal_plugin::ops
