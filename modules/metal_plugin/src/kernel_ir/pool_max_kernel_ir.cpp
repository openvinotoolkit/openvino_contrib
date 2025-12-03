// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/pool_max_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/max_pool.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_maxpool(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;
    std::shared_ptr<const ov::op::v1::MaxPool> pool_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto p = ov::as_type_ptr<const ov::op::v1::MaxPool>(node)) {
            pool_node = p;
            break;
        }
    }
    OPENVINO_ASSERT(pool_node, "MaxPool builder: MaxPool op not found");
    auto in_shape = pool_node->get_input_shape(0);
    auto out_shape = pool_node->get_output_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4, "MaxPool2D expects rank-4 NCHW input");
    OPENVINO_ASSERT(pool_node->get_input_element_type(0) == ov::element::f32, "MaxPool only supports f32");

    KernelTensor in{"in", {in_shape.begin(), in_shape.end()}};
    KernelTensor out{"out", {out_shape.begin(), out_shape.end()}};
    ir.tensors.push_back(in);
    ir.tensors.push_back(out);

    const auto& k = pool_node->get_kernel();
    const auto& s = pool_node->get_strides();
    const auto& pb = pool_node->get_pads_begin();

    KernelOp op;
    op.kind = KernelOpKind::MaxPool2D;
    op.input0 = &ir.tensors[0];
    op.output = &ir.tensors[1];
    op.pool.N = static_cast<uint32_t>(in_shape[0]);
    op.pool.C = static_cast<uint32_t>(in_shape[1]);
    op.pool.H = static_cast<uint32_t>(in_shape[2]);
    op.pool.W = static_cast<uint32_t>(in_shape[3]);
    op.pool.outH = static_cast<uint32_t>(out_shape[2]);
    op.pool.outW = static_cast<uint32_t>(out_shape[3]);
    op.pool.kernelH = static_cast<uint32_t>(k[0]);
    op.pool.kernelW = static_cast<uint32_t>(k[1]);
    op.pool.strideH = static_cast<uint32_t>(s[0]);
    op.pool.strideW = static_cast<uint32_t>(s[1]);
    op.pool.padTop = static_cast<uint32_t>(pb[0]);
    op.pool.padLeft = static_cast<uint32_t>(pb[1]);
    ir.ops.push_back(op);
    return ir;
}

}  // namespace metal_plugin
}  // namespace ov

