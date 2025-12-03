// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/conv_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_conv2d(const std::shared_ptr<const ov::Model>& model, bool& has_const_weights) {
    MetalKernelIR ir;
    has_const_weights = false;

    std::shared_ptr<const ov::Node> conv_node_base;
    bool is_group = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            conv_node_base = c;
            break;
        }
        if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
            conv_node_base = g;
            is_group = true;
            break;
        }
    }
    OPENVINO_ASSERT(conv_node_base, "Conv2D builder: Convolution/GroupConvolution op not found");

    auto get_shape = [&](size_t idx) { return conv_node_base->get_input_shape(idx); };
    const auto& in_shape = get_shape(0);   // NCHW
    const auto& w_shape = get_shape(1);    // OIHW or [G, O/G, C/G, kH, kW]
    const auto& out_shape = conv_node_base->get_output_shape(0); // NCHW
    OPENVINO_ASSERT(in_shape.size() == 4, "Conv2D expects rank-4 NCHW input");
    OPENVINO_ASSERT(w_shape.size() == (is_group ? 5 : 4), "Conv2D expects rank-4 or rank-5 weights for group conv");
    const auto in_et = conv_node_base->get_input_element_type(0);
    const auto w_et = conv_node_base->get_input_element_type(1);
    OPENVINO_ASSERT(in_et == ov::element::f32 || in_et == ov::element::f16, "Conv2D supports f32/f16");
    OPENVINO_ASSERT(w_et == ov::element::f32 || w_et == ov::element::f16, "Conv2D weights must be f32/f16");
    const auto& dil = is_group
        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(conv_node_base)->get_dilations()
        : ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node_base)->get_dilations();
    OPENVINO_ASSERT(dil.size() == 2, "Conv2D: dilations rank must be 2");

    KernelTensor in{"in", {in_shape.begin(), in_shape.end()}};
    KernelTensor w{"w", {w_shape.begin(), w_shape.end()}};
    KernelTensor out{"out", {out_shape.begin(), out_shape.end()}};
    ir.tensors.push_back(in);
    ir.tensors.push_back(w);
    ir.tensors.push_back(out);

    const auto& s = is_group
        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(conv_node_base)->get_strides()
        : ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node_base)->get_strides();
    const auto& pb = is_group
        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(conv_node_base)->get_pads_begin()
        : ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node_base)->get_pads_begin();
    const auto& pe = is_group
        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(conv_node_base)->get_pads_end()
        : ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node_base)->get_pads_end();

    KernelOp op;
    op.kind = KernelOpKind::Conv2D;
    op.input0 = &ir.tensors[0];
    op.input1 = &ir.tensors[1];
    op.output = &ir.tensors[2];
    op.conv2d.N = static_cast<uint32_t>(in_shape[0]);
    if (!is_group) {
        op.conv2d.C_in = static_cast<uint32_t>(in_shape[1]);
        op.conv2d.groups = 1;
        op.conv2d.C_in_per_group = op.conv2d.C_in;
        op.conv2d.C_out = static_cast<uint32_t>(w_shape[0]);
        op.conv2d.C_out_per_group = op.conv2d.C_out;
        op.conv2d.kernelH = static_cast<uint32_t>(w_shape[2]);
        op.conv2d.kernelW = static_cast<uint32_t>(w_shape[3]);
    } else {
        uint32_t groups = static_cast<uint32_t>(w_shape[0]);
        uint32_t c_out_pg = static_cast<uint32_t>(w_shape[1]);
        uint32_t c_in_pg = static_cast<uint32_t>(w_shape[2]);
        op.conv2d.groups = groups;
        op.conv2d.C_in = static_cast<uint32_t>(in_shape[1]);
        op.conv2d.C_in_per_group = c_in_pg;
        op.conv2d.C_out = groups * c_out_pg;
        op.conv2d.C_out_per_group = c_out_pg;
        op.conv2d.kernelH = static_cast<uint32_t>(w_shape[3]);
        op.conv2d.kernelW = static_cast<uint32_t>(w_shape[4]);
        OPENVINO_ASSERT(op.conv2d.C_in == groups * c_in_pg, "GroupConv: input channels mismatch with groups");
    }
    op.conv2d.H = static_cast<uint32_t>(in_shape[2]);
    op.conv2d.W = static_cast<uint32_t>(in_shape[3]);
    op.conv2d.strideH = static_cast<uint32_t>(s[0]);
    op.conv2d.strideW = static_cast<uint32_t>(s[1]);
    op.conv2d.padTop = static_cast<uint32_t>(pb[0]);
    op.conv2d.padLeft = static_cast<uint32_t>(pb[1]);
    op.conv2d.padBottom = static_cast<uint32_t>(pe[0]);
    op.conv2d.padRight = static_cast<uint32_t>(pe[1]);
    op.conv2d.dilationH = static_cast<uint32_t>(dil[0]);
    op.conv2d.dilationW = static_cast<uint32_t>(dil[1]);
    if (out_shape.size() == 4) {
        op.conv2d.outH = static_cast<uint32_t>(out_shape[2]);
        op.conv2d.outW = static_cast<uint32_t>(out_shape[3]);
    }

    // pad type (0 - EXPLICIT, 1 - SAME_UPPER, 2 - SAME_LOWER, 3 - VALID)
    if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node_base)) {
        op.conv2d.padType = static_cast<uint32_t>(conv->get_auto_pad());
    } else if (auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(conv_node_base)) {
        op.conv2d.padType = static_cast<uint32_t>(gconv->get_auto_pad());
    }
    op.conv2d.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(in_et));

    // Detect constant weights
    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv_node_base->get_input_node_shared_ptr(1))) {
        has_const_weights = true;
    }

    ir.ops.push_back(op);
    return ir;
}

}  // namespace metal_plugin
}  // namespace ov
