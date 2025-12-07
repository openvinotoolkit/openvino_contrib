// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sub_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace metal_plugin {

namespace {
std::shared_ptr<const ov::op::v1::Subtract> find_sub(const std::shared_ptr<const ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (auto s = ov::as_type_ptr<const ov::op::v1::Subtract>(node)) {
            return s;
        }
    }
    return {};
}
}  // namespace

MetalKernelIR build_kernel_ir_for_subtract(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;

    auto sub_node = find_sub(model);
    OPENVINO_ASSERT(sub_node, "Subtract builder: no Subtract node found in model");
    OPENVINO_ASSERT(sub_node->get_input_size() == 2, "Subtract must have two inputs");
    OPENVINO_ASSERT(sub_node->get_output_size() == 1, "Subtract must have one output");
    auto et = sub_node->get_output_element_type(0);
    OPENVINO_ASSERT(et == ov::element::f32 || et == ov::element::f16 || et == ov::element::i32,
                    "Subtract builder: only f32/f16/i32 supported");

    const auto& shape0 = sub_node->get_input_shape(0);
    const auto& shape1 = sub_node->get_input_shape(1);

    KernelTensor in0{"in0", {shape0.begin(), shape0.end()}, resolve_metal_dtype(et)};
    KernelTensor in1{"in1", {shape1.begin(), shape1.end()}, resolve_metal_dtype(et)};
    KernelTensor out{"out", {sub_node->get_output_shape(0).begin(), sub_node->get_output_shape(0).end()},
                     resolve_metal_dtype(et)};

    ir.tensors.push_back(in0);
    ir.tensors.push_back(in1);
    ir.tensors.push_back(out);

    KernelOp op;
    op.kind = KernelOpKind::ElementwiseSub;
    op.input0 = &ir.tensors[0];
    op.input1 = &ir.tensors[1];
    op.output = &ir.tensors[2];
    op.dtype = resolve_metal_dtype(et);
    op.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(et));
    ir.ops.push_back(op);

    return ir;
}

MetalKernelIR build_kernel_ir_for_broadcast_subtract(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;

    auto sub_node = find_sub(model);
    OPENVINO_ASSERT(sub_node, "Broadcast Subtract builder: no Subtract node found in model");
    OPENVINO_ASSERT(sub_node->get_input_size() == 2, "Subtract must have two inputs");
    OPENVINO_ASSERT(sub_node->get_output_size() == 1, "Subtract must have one output");
    OPENVINO_ASSERT(sub_node->get_input_element_type(0) == ov::element::f32,
                    "Subtract builder: only f32 Subtract is supported");
    OPENVINO_ASSERT(sub_node->get_input_element_type(1) == ov::element::f32,
                    "Subtract builder: only f32 Subtract is supported");

    auto shape0 = sub_node->get_input_shape(0);
    auto shape1 = sub_node->get_input_shape(1);
    auto out_shape = sub_node->get_output_shape(0);

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
    op.kind = KernelOpKind::ElementwiseSub;
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
