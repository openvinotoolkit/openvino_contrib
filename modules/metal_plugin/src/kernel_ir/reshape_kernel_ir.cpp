// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_reshape(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;

    std::shared_ptr<const ov::Node> reshape_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v1::Reshape>(node.get())) {
            reshape_node = node;
            break;
        }
    }
    OPENVINO_ASSERT(reshape_node, "Reshape builder: reshape op not found");
    OPENVINO_ASSERT(reshape_node->get_input_size() >= 1, "Reshape must have input");
    OPENVINO_ASSERT(reshape_node->get_output_size() == 1, "Reshape must have one output");

    const auto& in_shape = reshape_node->get_input_shape(0);
    const auto& out_shape = reshape_node->get_output_shape(0);
    const auto et = reshape_node->get_output_element_type(0);

    KernelTensor in{"in", {in_shape.begin(), in_shape.end()}};
    KernelTensor out{"out", {out_shape.begin(), out_shape.end()}};
    in.dtype = resolve_metal_dtype(et);
    out.dtype = resolve_metal_dtype(et);
    ir.tensors.push_back(in);
    ir.tensors.push_back(out);

    KernelOp op;
    op.kind = KernelOpKind::Reshape;
    op.input0 = &ir.tensors[0];
    op.output = &ir.tensors[1];
    op.dtype = resolve_metal_dtype(et);
    op.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(et));
    op.reshape.in_shape.assign(in_shape.begin(), in_shape.end());
    op.reshape.out_shape.assign(out_shape.begin(), out_shape.end());
    ir.ops.push_back(op);

    return ir;
}

}  // namespace metal_plugin
}  // namespace ov
