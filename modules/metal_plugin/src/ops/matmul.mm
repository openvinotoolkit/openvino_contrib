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
    auto rank_a = node.input_value(0).get_partial_shape().rank();
    auto rank_b = node.input_value(1).get_partial_shape().rank();
    if (!(rank_a.is_static() && rank_b.is_static())) {
        OPENVINO_THROW("MatMul: dynamic ranks are not supported yet");
    }
    auto ra = rank_a.get_length();
    auto rb = rank_b.get_length();
    if (!((ra == 2 && rb == 2) || (ra == 3 && rb == 3))) {
        OPENVINO_THROW("MatMul: only 2D or 3D batched matmul is supported");
    }
    if (node.get_transpose_a() || node.get_transpose_b()) {
        OPENVINO_THROW("MatMul: transpose flags not supported in current MPSGraph path");
    }

    MPSGraphTensor* res =
        [ctx.graph() matrixMultiplicationWithPrimaryTensor:a.tensor secondaryTensor:b.tensor name:nil];

    out->output_desc.shape = node.get_output_shape(0);
    out->output_desc.element_type = node.get_output_element_type(0);
    out->output_desc.layout = Layout::NCHW;
    out->mps_tensor = res;
    return out;
}

}  // namespace ov::metal_plugin::ops
