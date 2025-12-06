// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/conv3d_kernel_ir.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_conv3d(const std::shared_ptr<const ov::Model>& model, bool& has_const_weights) {
    MetalKernelIR ir;
    has_const_weights = false;

    std::shared_ptr<const ov::op::v1::Convolution> conv;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            if (c->get_input_shape(0).size() == 5) {
                conv = c;
                break;
            }
        }
    }
    OPENVINO_ASSERT(conv, "Conv3D builder: rank-5 Convolution not found");

    const auto& in_shape = conv->get_input_shape(0);   // NCDHW
    const auto& w_shape = conv->get_input_shape(1);    // OIDHW
    const auto& out_shape = conv->get_output_shape(0); // NCDHW
    OPENVINO_ASSERT(in_shape.size() == 5 && w_shape.size() == 5, "Conv3D expects rank-5 inputs/weights");

    const auto in_et = conv->get_input_element_type(0);
    const auto w_et = conv->get_input_element_type(1);
    OPENVINO_ASSERT(in_et == ov::element::f32 || in_et == ov::element::f16, "Conv3D supports f32/f16");
    OPENVINO_ASSERT(w_et == ov::element::f32 || w_et == ov::element::f16, "Conv3D weights must be f32/f16");

    KernelTensor in{"in", {in_shape.begin(), in_shape.end()}};
    KernelTensor w{"w", {w_shape.begin(), w_shape.end()}};
    KernelTensor out{"out", {out_shape.begin(), out_shape.end()}};
    in.dtype = resolve_metal_dtype(in_et);
    w.dtype = resolve_metal_dtype(w_et);
    out.dtype = resolve_metal_dtype(conv->get_output_element_type(0));
    ir.tensors = {in, w, out};

    const auto& s = conv->get_strides();       // {sD,sH,sW}
    const auto& pb = conv->get_pads_begin();   // {pF,pT,pL}
    const auto& pe = conv->get_pads_end();     // {pB,pBot,pR}
    const auto& dil = conv->get_dilations();   // {dD,dH,dW}

    KernelOp op;
    op.kind = KernelOpKind::Conv3D;
    op.input0 = &ir.tensors[0];
    op.input1 = &ir.tensors[1];
    op.output = &ir.tensors[2];
    op.conv3d.N = static_cast<uint32_t>(in_shape[0]);
    op.conv3d.C_in = static_cast<uint32_t>(in_shape[1]);
    op.conv3d.D = static_cast<uint32_t>(in_shape[2]);
    op.conv3d.H = static_cast<uint32_t>(in_shape[3]);
    op.conv3d.W = static_cast<uint32_t>(in_shape[4]);
    op.conv3d.C_out = static_cast<uint32_t>(w_shape[0]);
    op.conv3d.kernelD = static_cast<uint32_t>(w_shape[2]);
    op.conv3d.kernelH = static_cast<uint32_t>(w_shape[3]);
    op.conv3d.kernelW = static_cast<uint32_t>(w_shape[4]);
    op.conv3d.strideD = static_cast<uint32_t>(s[0]);
    op.conv3d.strideH = static_cast<uint32_t>(s[1]);
    op.conv3d.strideW = static_cast<uint32_t>(s[2]);
    op.conv3d.padFront = static_cast<uint32_t>(pb[0]);
    op.conv3d.padTop = static_cast<uint32_t>(pb[1]);
    op.conv3d.padLeft = static_cast<uint32_t>(pb[2]);
    op.conv3d.padBack = static_cast<uint32_t>(pe[0]);
    op.conv3d.padBottom = static_cast<uint32_t>(pe[1]);
    op.conv3d.padRight = static_cast<uint32_t>(pe[2]);
    op.conv3d.dilationD = static_cast<uint32_t>(dil[0]);
    op.conv3d.dilationH = static_cast<uint32_t>(dil[1]);
    op.conv3d.dilationW = static_cast<uint32_t>(dil[2]);
    if (out_shape.size() == 5) {
        op.conv3d.outD = static_cast<uint32_t>(out_shape[2]);
        op.conv3d.outH = static_cast<uint32_t>(out_shape[3]);
        op.conv3d.outW = static_cast<uint32_t>(out_shape[4]);
    }
    op.dtype = resolve_metal_dtype(in_et);
    op.conv3d.dtype = op.dtype;
    op.conv3d.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(in_et));

    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv->get_input_node_shared_ptr(1))) {
        has_const_weights = true;
    }

    ir.ops.push_back(op);
    return ir;
}

}  // namespace metal_plugin
}  // namespace ov
