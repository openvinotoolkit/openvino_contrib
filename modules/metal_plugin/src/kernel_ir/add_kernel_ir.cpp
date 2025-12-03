// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/add_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_add(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;

    std::shared_ptr<const ov::op::v1::Add> add_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto add = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
            add_node = add;
            break;
        }
    }
    OPENVINO_ASSERT(add_node, "Add builder: no Add node found in model");
    OPENVINO_ASSERT(add_node->get_input_size() == 2, "Add must have two inputs");
    OPENVINO_ASSERT(add_node->get_output_size() == 1, "Add must have one output");
    OPENVINO_ASSERT(add_node->get_input_element_type(0) == ov::element::f32,
                    "Add builder: only f32 Add is supported");
    OPENVINO_ASSERT(add_node->get_input_element_type(1) == ov::element::f32,
                    "Add builder: only f32 Add is supported");

    const auto& shape0 = add_node->get_input_shape(0);
    const auto& shape1 = add_node->get_input_shape(1);

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

MetalKernelIR build_kernel_ir_for_broadcast_add(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;

    std::shared_ptr<const ov::op::v1::Add> add_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto add = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
            add_node = add;
            break;
        }
    }
    OPENVINO_ASSERT(add_node, "Broadcast Add builder: no Add node found in model");
    OPENVINO_ASSERT(add_node->get_input_size() == 2, "Add must have two inputs");
    OPENVINO_ASSERT(add_node->get_output_size() == 1, "Add must have one output");
    OPENVINO_ASSERT(add_node->get_input_element_type(0) == ov::element::f32,
                    "Add builder: only f32 Add is supported");
    OPENVINO_ASSERT(add_node->get_input_element_type(1) == ov::element::f32,
                    "Add builder: only f32 Add is supported");

    auto shape0 = add_node->get_input_shape(0);
    auto shape1 = add_node->get_input_shape(1);
    auto out_shape = add_node->get_output_shape(0);

    auto normalize_rank = [](const ov::Shape& s, size_t rank) {
        std::vector<int64_t> r(rank, 1);
        size_t offset = rank - s.size();
        for (size_t i = 0; i < s.size(); ++i) r[offset + i] = static_cast<int64_t>(s[i]);
        return r;
    };

    size_t rank = out_shape.size();
    auto a = normalize_rank(shape0, rank);
    auto b = normalize_rank(shape1, rank);
    std::vector<int64_t> strides_a(rank, 0), strides_b(rank, 0);

    auto compute_strides = [&](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    auto strides_norm_a = compute_strides(a);
    auto strides_norm_b = compute_strides(b);

    for (size_t i = 0; i < rank; ++i) {
        strides_a[i] = (a[i] == 1 ? 0 : strides_norm_a[i]);
        strides_b[i] = (b[i] == 1 ? 0 : strides_norm_b[i]);
    }

    KernelTensor in0{"in0", {shape0.begin(), shape0.end()}};
    KernelTensor in1{"in1", {shape1.begin(), shape1.end()}};
    KernelTensor out{"out", {out_shape.begin(), out_shape.end()}};

    ir.tensors.push_back(in0);
    ir.tensors.push_back(in1);
    ir.tensors.push_back(out);

    KernelOp op;
    op.kind = KernelOpKind::ElementwiseAdd;
    op.is_broadcast = true;
    op.out_shape.assign(out_shape.begin(), out_shape.end());
    op.stride0 = std::move(strides_a);
    op.stride1 = std::move(strides_b);
    op.input0 = &ir.tensors[0];
    op.input1 = &ir.tensors[1];
    op.output = &ir.tensors[2];
    ir.ops.push_back(op);

    return ir;
}

}  // namespace metal_plugin
}  // namespace ov

