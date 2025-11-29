// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_codegen/kernel_ir.hpp"

#include "graph/metal_node.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace metal_plugin {

// Build a trivial IR for a graph that contains exactly one Add with static shapes.
MetalKernelIR build_kernel_ir_for_add(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;

    size_t add_count = 0;
    std::shared_ptr<const ov::op::v1::Add> add_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto add = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
            add_count++;
            add_node = add;
        }
    }
    OPENVINO_ASSERT(add_count == 1, "CustomKernelBackend currently supports exactly one Add node");
    OPENVINO_ASSERT(add_node->get_input_size() == 2, "Add must have two inputs");
    OPENVINO_ASSERT(add_node->get_output_size() == 1, "Add must have one output");
    OPENVINO_ASSERT(add_node->get_input_element_type(0) == ov::element::f32,
                    "CustomKernelBackend: only f32 Add is supported");
    OPENVINO_ASSERT(add_node->get_input_element_type(1) == ov::element::f32,
                    "CustomKernelBackend: only f32 Add is supported");

    const auto& shape0 = add_node->get_input_shape(0);
    const auto& shape1 = add_node->get_input_shape(1);
    OPENVINO_ASSERT(shape0 == shape1, "CustomKernelBackend: Add requires equal input shapes");

    KernelTensor in0{"in0", {shape0.begin(), shape0.end()}};
    KernelTensor in1{"in1", {shape1.begin(), shape1.end()}};
    KernelTensor out{"out", {add_node->get_output_shape(0).begin(), add_node->get_output_shape(0).end()}};

    ir.tensors.push_back(in0);
    ir.tensors.push_back(in1);
    ir.tensors.push_back(out);

    KernelOp op;
    op.kind = KernelOpKind::ElementwiseAdd;
    op.input0 = &ir.tensors[0];
    op.input1 = &ir.tensors[1];
    op.output = &ir.tensors[2];
    ir.ops.push_back(op);

    return ir;
}

MetalKernelIR build_kernel_ir_for_matmul(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;
    size_t mm_count = 0;
    std::shared_ptr<const ov::op::v0::MatMul> matmul;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
            mm_count++;
            matmul = mm;
        }
    }
    OPENVINO_ASSERT(mm_count == 1, "CustomKernelBackend/MLIR path supports exactly one MatMul node");
    OPENVINO_ASSERT(!matmul->get_transpose_a() && !matmul->get_transpose_b(),
                    "MatMul: transposed inputs are not supported in custom backend");
    auto shape_a = matmul->get_input_shape(0);
    auto shape_b = matmul->get_input_shape(1);
    OPENVINO_ASSERT(shape_a.size() == 2 && shape_b.size() == 2, "MatMul: only 2D inputs supported");
    const int64_t M = static_cast<int64_t>(shape_a[0]);
    const int64_t K = static_cast<int64_t>(shape_a[1]);
    OPENVINO_ASSERT(shape_b[0] == static_cast<size_t>(K), "MatMul: K mismatch");
    const int64_t N = static_cast<int64_t>(shape_b[1]);

    OPENVINO_ASSERT(matmul->get_output_element_type(0) == ov::element::f32, "MatMul only supports f32");

    KernelTensor a{"a", {shape_a.begin(), shape_a.end()}};
    KernelTensor b{"b", {shape_b.begin(), shape_b.end()}};
    KernelTensor c{"c", {M, N}};

    ir.tensors.push_back(a);
    ir.tensors.push_back(b);
    ir.tensors.push_back(c);

    KernelOp op;
    op.kind = KernelOpKind::MatMul;
    op.input0 = &ir.tensors[0];
    op.input1 = &ir.tensors[1];
    op.output = &ir.tensors[2];
    op.M = M;
    op.N = N;
    op.K = K;
    ir.ops.push_back(op);
    return ir;
}

}  // namespace metal_plugin
}  // namespace ov
