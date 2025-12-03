// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/batchnorm_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_batchnorm(const std::shared_ptr<const ov::Model>& model, bool& has_const_params) {
    MetalKernelIR ir;
    has_const_params = false;

    std::shared_ptr<const ov::Node> bn_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v5::BatchNormInference>(node.get()) ||
            ov::is_type<ov::op::v0::BatchNormInference>(node.get())) {
            bn_node = node;
            break;
        }
    }
    OPENVINO_ASSERT(bn_node, "BatchNorm builder: node not found");
    const auto bn_v5 = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(bn_node);
    const auto bn_v0 = ov::as_type_ptr<const ov::op::v0::BatchNormInference>(bn_node);
    const bool use_v5 = static_cast<bool>(bn_v5);

    const auto& in_shape = use_v5 ? bn_v5->get_input_shape(0) : bn_v0->get_input_shape(0);  // NCHW
    OPENVINO_ASSERT(in_shape.size() == 4, "BatchNorm expects rank-4 NCHW input");
    OPENVINO_ASSERT(bn_node->get_input_element_type(0) == ov::element::f32, "BatchNorm only supports f32");

    KernelTensor in{"in", {in_shape.begin(), in_shape.end()}};
    KernelTensor out{"out", {bn_node->get_output_shape(0).begin(), bn_node->get_output_shape(0).end()}};
    ir.tensors.push_back(in);
    ir.tensors.push_back(out);

    KernelOp op;
    op.kind = KernelOpKind::BatchNorm2D;
    op.input0 = &ir.tensors[0];
    op.output = &ir.tensors[1];
    op.batchnorm.N = static_cast<uint32_t>(in_shape[0]);
    op.batchnorm.C = static_cast<uint32_t>(in_shape[1]);
    op.batchnorm.H = static_cast<uint32_t>(in_shape[2]);
    op.batchnorm.W = static_cast<uint32_t>(in_shape[3]);
    op.batchnorm.eps = static_cast<float>(use_v5 ? bn_v5->get_eps_value() : bn_v0->get_eps_value());

    auto gamma = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
        use_v5 ? bn_v5->get_input_node_shared_ptr(1) : bn_v0->get_input_node_shared_ptr(1));
    auto beta = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
        use_v5 ? bn_v5->get_input_node_shared_ptr(2) : bn_v0->get_input_node_shared_ptr(2));
    auto mean = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
        use_v5 ? bn_v5->get_input_node_shared_ptr(3) : bn_v0->get_input_node_shared_ptr(3));
    auto var = std::dynamic_pointer_cast<const ov::op::v0::Constant>(
        use_v5 ? bn_v5->get_input_node_shared_ptr(4) : bn_v0->get_input_node_shared_ptr(4));
    if (gamma && beta && mean && var) {
        has_const_params = true;
        const size_t C = in_shape[1];
        op.bn_params.resize(4 * C + 1, 0.f);
        auto copy_vec = [&](const std::shared_ptr<const ov::op::v0::Constant>& cst, size_t offset) {
            auto vec = cst->cast_vector<float>();
            OPENVINO_ASSERT(vec.size() == C, "BatchNorm param size mismatch");
            std::copy(vec.begin(), vec.end(), op.bn_params.begin() + offset);
        };
        copy_vec(gamma, 0 * C);
        copy_vec(beta, 1 * C);
        copy_vec(mean, 2 * C);
        copy_vec(var, 3 * C);
        op.bn_params[4 * C] = op.batchnorm.eps;
    }

    ir.ops.push_back(op);
    return ir;
}

}  // namespace metal_plugin
}  // namespace ov
