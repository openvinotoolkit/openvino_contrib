// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/softmax_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/softmax.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_softmax(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;
    std::shared_ptr<const ov::Node> sm_node;
    int64_t axis = -1;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto s = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
            sm_node = s;
            axis = s->get_axis();
            break;
        } else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
            sm_node = s8;
            axis = s8->get_axis();
            break;
        }
    }
    OPENVINO_ASSERT(sm_node, "Softmax builder: Softmax op not found");
    auto pshape = sm_node->get_input_partial_shape(0);
    OPENVINO_ASSERT(pshape.rank().is_static(), "Softmax: input rank must be static");
    // Shape may be partially dynamic; rows/cols/inner will be recomputed at runtime when needed.
    std::vector<int64_t> shape;
    if (pshape.is_static()) {
        auto s = pshape.to_shape();
        shape.assign(s.begin(), s.end());
    } else {
        const auto r = pshape.rank().get_length();
        shape.assign(static_cast<size_t>(r), -1);
    }
    const auto et = sm_node->get_input_element_type(0);
    OPENVINO_ASSERT(et == ov::element::f32 || et == ov::element::f16, "Softmax supports only f16/f32");

    if (axis < 0) axis += static_cast<int64_t>(shape.size());
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(shape.size()), "Softmax: invalid axis");

    int64_t rows = 0, cols = 0, inner = 1;
    if (pshape.is_static()) {
        cols = static_cast<int64_t>(shape[static_cast<size_t>(axis)]);
        int64_t outer = 1;
        for (int64_t i = 0; i < axis; ++i) outer *= static_cast<int64_t>(shape[static_cast<size_t>(i)]);
        for (size_t i = axis + 1; i < shape.size(); ++i) inner *= static_cast<int64_t>(shape[i]);
        rows = outer * inner;
    }

    KernelTensor in{"in", {shape.begin(), shape.end()}};
    KernelTensor out{"out", {shape.begin(), shape.end()}};
    ir.tensors.push_back(in);
    ir.tensors.push_back(out);

    KernelOp op;
    op.kind = KernelOpKind::Softmax;
    op.rows = rows;
    op.cols = cols;
    op.inner = inner;
    op.softmax_axis = axis;
    op.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(et));
    op.input0 = &ir.tensors[0];
    op.output = &ir.tensors[1];
    ir.ops.push_back(op);

    return ir;
}

}  // namespace metal_plugin
}  // namespace ov
