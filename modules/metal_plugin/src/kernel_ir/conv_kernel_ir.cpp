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

    const auto in_pshape = conv_node_base->get_input_partial_shape(0);   // NCHW
    const auto w_pshape  = conv_node_base->get_input_partial_shape(1);   // OIHW or [G, O/G, C/G, kH, kW]
    const auto out_pshape = conv_node_base->get_output_partial_shape(0); // NCHW

    OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4,
                    "Conv2D expects rank-4 NCHW input");
    OPENVINO_ASSERT(w_pshape.rank().is_static() && w_pshape.rank().get_length() == (is_group ? 5 : 4),
                    "Conv2D expects rank-4 or rank-5 weights for group conv");
    OPENVINO_ASSERT(out_pshape.rank().is_static() && out_pshape.rank().get_length() == 4,
                    "Conv2D expects rank-4 output");

    auto to_shape = [](const ov::PartialShape& ps) {
        std::vector<int64_t> shp;
        shp.reserve(ps.rank().get_length());
        for (const auto& d : ps) {
            shp.push_back(d.is_dynamic() ? -1 : static_cast<int64_t>(d.get_length()));
        }
        return shp;
    };

    const auto in_shape  = to_shape(in_pshape);
    const auto w_shape   = to_shape(w_pshape);
    const auto out_shape = to_shape(out_pshape);
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
    in.dtype = resolve_metal_dtype(in_et);
    w.dtype = resolve_metal_dtype(w_et);
    out.dtype = resolve_metal_dtype(conv_node_base->get_output_element_type(0));
    ir.tensors.push_back(in);
    ir.tensors.push_back(w);
    ir.tensors.push_back(out);

    auto dim_or_zero = [](int64_t v) -> uint32_t {
        return v < 0 ? 0u : static_cast<uint32_t>(v);
    };
    auto require_static = [&](int64_t v, const char* what) -> uint32_t {
        OPENVINO_ASSERT(v >= 0, what);
        return static_cast<uint32_t>(v);
    };

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
    op.conv2d.N = dim_or_zero(in_shape[0]);
    if (!is_group) {
        op.conv2d.C_in = require_static(in_shape[1], "Conv2D: dynamic C_in is not supported");
        op.conv2d.groups = 1;
        op.conv2d.C_in_per_group = op.conv2d.C_in;
        op.conv2d.C_out = require_static(w_shape[0], "Conv2D: dynamic C_out is not supported");
        op.conv2d.C_out_per_group = op.conv2d.C_out;
        op.conv2d.kernelH = require_static(w_shape[2], "Conv2D: dynamic kernelH is not supported");
        op.conv2d.kernelW = require_static(w_shape[3], "Conv2D: dynamic kernelW is not supported");
    } else {
        uint32_t groups = require_static(w_shape[0], "GroupConv: dynamic groups not supported");
        uint32_t c_out_pg = require_static(w_shape[1], "GroupConv: dynamic C_out/group not supported");
        uint32_t c_in_pg = require_static(w_shape[2], "GroupConv: dynamic C_in/group not supported");
        op.conv2d.groups = groups;
        op.conv2d.C_in = require_static(in_shape[1], "GroupConv: dynamic C_in not supported");
        op.conv2d.C_in_per_group = c_in_pg;
        op.conv2d.C_out = groups * c_out_pg;
        op.conv2d.C_out_per_group = c_out_pg;
        op.conv2d.kernelH = require_static(w_shape[3], "GroupConv: dynamic kernelH not supported");
        op.conv2d.kernelW = require_static(w_shape[4], "GroupConv: dynamic kernelW not supported");
        OPENVINO_ASSERT(op.conv2d.C_in == groups * c_in_pg, "GroupConv: input channels mismatch with groups");
    }
    op.conv2d.H = dim_or_zero(in_shape[2]);
    op.conv2d.W = dim_or_zero(in_shape[3]);
    op.conv2d.strideH = static_cast<uint32_t>(s[0]);
    op.conv2d.strideW = static_cast<uint32_t>(s[1]);
    op.conv2d.padTop = static_cast<uint32_t>(pb[0]);
    op.conv2d.padLeft = static_cast<uint32_t>(pb[1]);
    op.conv2d.padBottom = static_cast<uint32_t>(pe[0]);
    op.conv2d.padRight = static_cast<uint32_t>(pe[1]);
    op.conv2d.dilationH = static_cast<uint32_t>(dil[0]);
    op.conv2d.dilationW = static_cast<uint32_t>(dil[1]);
    if (out_shape.size() == 4) {
        op.conv2d.outH = dim_or_zero(out_shape[2]);  // mirrored; runtime/MLIR compute authoritative value
        op.conv2d.outW = dim_or_zero(out_shape[3]);
    }

    // pad type (0 - EXPLICIT, 1 - SAME_UPPER, 2 - SAME_LOWER, 3 - VALID)
    if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(conv_node_base)) {
        op.conv2d.padType = static_cast<uint32_t>(conv->get_auto_pad());
    } else if (auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(conv_node_base)) {
        op.conv2d.padType = static_cast<uint32_t>(gconv->get_auto_pad());
    }
    op.dtype = resolve_metal_dtype(in_et);
    op.conv2d.dtype = op.dtype;
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
