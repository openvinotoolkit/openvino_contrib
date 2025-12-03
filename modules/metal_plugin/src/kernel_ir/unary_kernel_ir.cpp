// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/unary_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/tanh.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_unary(const std::shared_ptr<const ov::Node>& node,
                                        ActivationKind kind,
                                        float alpha) {
    MetalKernelIR ir;

    OPENVINO_ASSERT(node->get_input_size() >= 1, "Unary builder: expected at least one input");
    auto shape = node->get_input_shape(0);
    OPENVINO_ASSERT(node->get_input_element_type(0) == ov::element::f32, "Unary builder supports only f32");

    KernelTensor in{"in", {shape.begin(), shape.end()}};
    KernelTensor out{"out", {shape.begin(), shape.end()}};
    ir.tensors.push_back(in);
    ir.tensors.push_back(out);

    KernelOp op;
    op.kind = KernelOpKind::Unary;
    op.activation = kind;
    op.alpha = alpha;
    op.input0 = &ir.tensors[0];
    op.output = &ir.tensors[1];
    ir.ops.push_back(op);

    return ir;
}

}  // namespace metal_plugin
}  // namespace ov

