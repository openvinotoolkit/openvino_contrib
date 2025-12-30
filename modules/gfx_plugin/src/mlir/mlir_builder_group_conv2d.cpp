// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace ov {
namespace gfx_plugin {

namespace {
mlir::SmallVector<int64_t> to_tensor_shape(const ov::PartialShape& ps) {
    mlir::SmallVector<int64_t> dims;
    dims.reserve(ps.rank().get_length());
    for (const auto& d : ps) {
        dims.push_back(d.is_dynamic() ? mlir::ShapedType::kDynamic
                                      : static_cast<int64_t>(d.get_length()));
    }
    return dims;
}

mlir::DenseIntElementsAttr make_i64_attr(mlir::OpBuilder& b, const ov::Strides& vals) {
    auto i64 = b.getI64Type();
    auto type = mlir::RankedTensorType::get({static_cast<int64_t>(vals.size())}, i64);
    llvm::SmallVector<int64_t, 4> data;
    data.reserve(vals.size());
    for (auto v : vals) {
        data.push_back(static_cast<int64_t>(v));
    }
    return mlir::DenseIntElementsAttr::get(type, data);
}

mlir::Value pad_input(mlir::OpBuilder& b,
                      mlir::Location loc,
                      mlir::Value input,
                      const ov::CoordinateDiff& pads_begin,
                      const ov::CoordinateDiff& pads_end) {
    if ((pads_begin[0] == 0 && pads_begin[1] == 0) &&
        (pads_end[0] == 0 && pads_end[1] == 0)) {
        return input;
    }

    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto elem_ty = input_type.getElementType();
    auto in_shape = input_type.getShape();
    mlir::SmallVector<int64_t, 4> padded_shape(in_shape.begin(), in_shape.end());
    if (padded_shape.size() >= 4) {
        if (padded_shape[2] != mlir::ShapedType::kDynamic) {
            padded_shape[2] += pads_begin[0] + pads_end[0];
        }
        if (padded_shape[3] != mlir::ShapedType::kDynamic) {
            padded_shape[3] += pads_begin[1] + pads_end[1];
        }
    }
    auto padded_type = mlir::RankedTensorType::get(padded_shape, elem_ty);

    mlir::SmallVector<mlir::OpFoldResult, 4> low;
    mlir::SmallVector<mlir::OpFoldResult, 4> high;
    low.reserve(4);
    high.reserve(4);
    low.push_back(b.getI64IntegerAttr(0));
    low.push_back(b.getI64IntegerAttr(0));
    low.push_back(b.getI64IntegerAttr(pads_begin[0]));
    low.push_back(b.getI64IntegerAttr(pads_begin[1]));
    high.push_back(b.getI64IntegerAttr(0));
    high.push_back(b.getI64IntegerAttr(0));
    high.push_back(b.getI64IntegerAttr(pads_end[0]));
    high.push_back(b.getI64IntegerAttr(pads_end[1]));

    auto pad_value = b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(elem_ty, 0.0));
    auto pad = b.create<mlir::tensor::PadOp>(loc,
                                             padded_type,
                                             input,
                                             llvm::ArrayRef<mlir::OpFoldResult>(low),
                                             llvm::ArrayRef<mlir::OpFoldResult>(high),
                                             pad_value,
                                             /*nofold=*/false,
                                             mlir::ArrayRef<mlir::NamedAttribute>{});
    return pad.getResult();
}
}  // namespace

mlir::ModuleOp build_mlir_group_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                                  mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::op::v1::GroupConvolution> gconv;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
            gconv = c;
            break;
        }
    }
    OPENVINO_ASSERT(gconv, "GroupConv2D builder: GroupConvolution op not found");

    const auto in_pshape  = gconv->get_input_partial_shape(0);
    const auto w_pshape   = gconv->get_input_partial_shape(1);
    const auto out_pshape = gconv->get_output_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() == 4,
                    "GroupConv2D builder expects rank-4 input");
    OPENVINO_ASSERT(w_pshape.rank().is_static() && w_pshape.rank().get_length() == 5,
                    "GroupConv2D builder expects rank-5 weights");
    OPENVINO_ASSERT(out_pshape.rank().is_static() && out_pshape.rank().get_length() == 4,
                    "GroupConv2D builder expects rank-4 output");

    const auto in_shape   = to_tensor_shape(in_pshape);
    const auto w_shape    = to_tensor_shape(w_pshape);
    const auto out_shape  = to_tensor_shape(out_pshape);
    const auto pads_begin = gconv->get_pads_begin();
    const auto pads_end   = gconv->get_pads_end();
    const auto strides    = gconv->get_strides();
    const auto dilations  = gconv->get_dilations();
    OPENVINO_ASSERT(w_shape[0] != mlir::ShapedType::kDynamic,
                    "GroupConv2D builder: group dimension must be static");
    const auto groups = static_cast<size_t>(w_shape[0]);

    auto to_elem_ty = [&](ov::element::Type et) -> mlir::Type {
        switch (et) {
            case ov::element::f16: return mlir::Float16Type::get(&ctx);
            case ov::element::f32: return mlir::Float32Type::get(&ctx);
            default: return mlir::Float32Type::get(&ctx);
        }
    };
    auto elem_ty = to_elem_ty(gconv->get_output_element_type(0));
    auto in_type  = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto w_type   = mlir::RankedTensorType::get(w_shape, elem_ty);
    auto out_type = mlir::RankedTensorType::get(out_shape, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_type, w_type}, {out_type});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "group_conv2d_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);

    auto input  = func.getArgument(0);
    auto weight = func.getArgument(1);

    auto padded = pad_input(b, loc, input, pads_begin, pads_end);

    const int64_t in_c = in_shape[1];
    const int64_t out_c = out_shape[1];
    const int64_t group_dim = w_shape[0];
    OPENVINO_ASSERT(group_dim == static_cast<int64_t>(groups),
                    "GroupConv2D builder: group dimension mismatch");

    if (groups == 1) {
        auto out_init = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);
        auto strides_attr = make_i64_attr(b, strides);
        auto dil_attr = make_i64_attr(b, dilations);
        auto conv_op = b.create<mlir::linalg::Conv2DNchwFchwOp>(loc,
                                                                out_type,
                                                                mlir::ValueRange{padded, weight},
                                                                mlir::ValueRange{out_init},
                                                                strides_attr,
                                                                dil_attr);
        b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{conv_op.getResult(0)});
        return module;
    }

    OPENVINO_ASSERT(in_c != mlir::ShapedType::kDynamic &&
                        out_c != mlir::ShapedType::kDynamic &&
                        in_c == static_cast<int64_t>(groups) &&
                        out_c == static_cast<int64_t>(groups),
                    "GroupConv2D builder: only depthwise (groups == in_channels == out_channels) is supported");

    OPENVINO_ASSERT(w_shape[1] != mlir::ShapedType::kDynamic &&
                        w_shape[2] != mlir::ShapedType::kDynamic &&
                        w_shape[1] == 1 && w_shape[2] == 1,
                    "GroupConv2D builder: depthwise expects weights shape [G,1,1,KH,KW]");

    // Collapse [G,1,1,KH,KW] -> [G,KH,KW] for depthwise conv.
    mlir::SmallVector<mlir::ReassociationIndices, 3> reassoc = {{0}, {1, 2, 3}, {4}};
    auto dw_weight = b.create<mlir::tensor::CollapseShapeOp>(loc, weight, reassoc);

    auto out_init = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);
    auto strides_attr = make_i64_attr(b, strides);
    auto dil_attr = make_i64_attr(b, dilations);
    auto dw_op = b.create<mlir::linalg::DepthwiseConv2DNchwChwOp>(loc,
                                                                  out_type,
                                                                  mlir::ValueRange{padded, dw_weight},
                                                                  mlir::ValueRange{out_init},
                                                                  strides_attr,
                                                                  dil_attr);
    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{dw_op.getResult(0)});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
