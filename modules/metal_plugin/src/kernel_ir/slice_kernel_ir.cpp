// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_slice(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;

    std::shared_ptr<const ov::op::v8::Slice> slice_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto s = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
            slice_node = s;
            break;
        }
    }
    OPENVINO_ASSERT(slice_node, "Slice builder: no Slice node found in model");

    OPENVINO_ASSERT(slice_node->get_input_size() == 4 || slice_node->get_input_size() == 5,
                    "Slice must have 4 or 5 inputs");
    OPENVINO_ASSERT(slice_node->get_output_size() == 1, "Slice must have one output");

    const auto& in_shape = slice_node->get_input_shape(0);
    const auto& out_shape = slice_node->get_output_shape(0);
    const auto dtype = resolve_metal_dtype(slice_node->get_input_element_type(0));

    KernelTensor in0{"in0", {in_shape.begin(), in_shape.end()}};
    KernelTensor out{"out", {out_shape.begin(), out_shape.end()}};
    in0.dtype = dtype;
    out.dtype = resolve_metal_dtype(slice_node->get_output_element_type(0));

    ir.tensors.push_back(in0);
    ir.tensors.push_back(out);

    KernelOp op;
    op.kind = KernelOpKind::Unary;  // placeholder; Slice handled specially in backend
    op.input0 = &ir.tensors[0];
    op.output = &ir.tensors[1];
    op.dtype = dtype;
    ir.ops.push_back(op);

    return ir;
}

KernelOp make_slice_op(const std::vector<int64_t>& in_shape,
                       const std::vector<int64_t>& starts,
                       const std::vector<int64_t>& steps,
                       const std::vector<int64_t>& out_shape,
                       ov::element::Type et,
                       KernelTensor& in_tensor,
                       KernelTensor& out_tensor) {
    KernelOp op;
    op.kind = KernelOpKind::Slice;
    op.input0 = &in_tensor;
    op.output = &out_tensor;
    op.dtype = resolve_metal_dtype(et);
    op.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(et));
    op.slice.dtype = op.dtype;
    op.slice.starts = starts;
    op.slice.steps = steps;
    op.slice.in_shape = in_shape;
    op.slice.out_shape = out_shape;
    // compute default axes and strides
    op.slice.axes.resize(in_shape.size());
    std::iota(op.slice.axes.begin(), op.slice.axes.end(), 0);
    // strides for flattening
    op.slice.in_strides.resize(in_shape.size());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(in_shape.size()) - 1; i >= 0; --i) {
        op.slice.in_strides[i] = stride;
        stride *= in_shape[i];
    }
    return op;
}

}  // namespace metal_plugin
}  // namespace ov
