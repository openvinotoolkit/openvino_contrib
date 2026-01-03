// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "runtime/gfx_logger.hpp"

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
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

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
    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("MLIR",
                      "GroupConv2D in=" << in_pshape << " w=" << w_pshape << " out=" << out_pshape
                                         << " pads_begin=(" << pads_begin[0] << "," << pads_begin[1] << ")"
                                         << " pads_end=(" << pads_end[0] << "," << pads_end[1] << ")"
                                         << " strides=(" << strides[0] << "," << strides[1] << ")"
                                         << " dilations=(" << dilations[0] << "," << dilations[1] << ")");
    }
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

    const int64_t in_c = in_shape[1];
    const int64_t out_c = out_shape[1];
    const int64_t group_dim = w_shape[0];
    OPENVINO_ASSERT(group_dim == static_cast<int64_t>(groups),
                    "GroupConv2D builder: group dimension mismatch");

    if (groups == 1) {
        auto padded = pad_input(b, loc, input, pads_begin, pads_end);
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

    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("MLIR", "GroupConv2D depthwise path: groups=" << groups);
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

    const int64_t kh = w_shape[3];
    const int64_t kw = w_shape[4];
    auto zero_idx = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto one_idx = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto kh_max = b.create<mlir::arith::ConstantIndexOp>(loc, kh);
    auto kw_max = b.create<mlir::arith::ConstantIndexOp>(loc, kw);
    auto stride_h = b.create<mlir::arith::ConstantIndexOp>(loc, strides[0]);
    auto stride_w = b.create<mlir::arith::ConstantIndexOp>(loc, strides[1]);
    auto dil_h = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[0]);
    auto dil_w = b.create<mlir::arith::ConstantIndexOp>(loc, dilations[1]);
    auto pad_h = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[0]);
    auto pad_w = b.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[1]);
    auto in_h = b.create<mlir::arith::ConstantIndexOp>(loc, in_shape[2]);
    auto in_w = b.create<mlir::arith::ConstantIndexOp>(loc, in_shape[3]);
    auto zero_val = b.create<mlir::arith::ConstantOp>(loc, b.getZeroAttr(elem_ty));

    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("MLIR", "GroupConv2D constants ready: kh=" << kh << " kw=" << kw);
    }

    auto gen = b.create<mlir::tensor::GenerateOp>(loc, out_type, mlir::ValueRange{});
    if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("MLIR", "GroupConv2D created tensor.generate");
    }
    {
        mlir::OpBuilder::InsertionGuard guard(b);
        auto idx_ty = b.getIndexType();
        auto* body = b.createBlock(&gen.getBody(),
                                   gen.getBody().begin(),
                                   {idx_ty, idx_ty, idx_ty, idx_ty},
                                   {loc, loc, loc, loc});
        b.setInsertionPointToStart(body);
        auto n = body->getArgument(0);
        auto c = body->getArgument(1);
        auto oh = body->getArgument(2);
        auto ow = body->getArgument(3);

        auto kh_loop = b.create<mlir::scf::ForOp>(loc,
                                                  zero_idx,
                                                  kh_max,
                                                  one_idx,
                                                  mlir::ValueRange{zero_val});
        {
            mlir::OpBuilder kb(kh_loop.getBody(), kh_loop.getBody()->begin());
            auto kh_iv = kh_loop.getInductionVar();
            auto acc_in = kh_loop.getRegionIterArg(0);
            auto kw_loop = kb.create<mlir::scf::ForOp>(loc,
                                                       zero_idx,
                                                       kw_max,
                                                       one_idx,
                                                       mlir::ValueRange{acc_in});
            {
                mlir::OpBuilder wb(kw_loop.getBody(), kw_loop.getBody()->begin());
                auto kw_iv = kw_loop.getInductionVar();
                auto acc = kw_loop.getRegionIterArg(0);
                auto oh_stride = wb.create<mlir::arith::MulIOp>(loc, oh, stride_h);
                auto kh_dil = wb.create<mlir::arith::MulIOp>(loc, kh_iv, dil_h);
                auto ih = wb.create<mlir::arith::SubIOp>(loc,
                                                         wb.create<mlir::arith::AddIOp>(loc, oh_stride, kh_dil),
                                                         pad_h);
                auto ow_stride = wb.create<mlir::arith::MulIOp>(loc, ow, stride_w);
                auto kw_dil = wb.create<mlir::arith::MulIOp>(loc, kw_iv, dil_w);
                auto iw = wb.create<mlir::arith::SubIOp>(loc,
                                                         wb.create<mlir::arith::AddIOp>(loc, ow_stride, kw_dil),
                                                         pad_w);

                auto ih_ge0 = wb.create<mlir::arith::CmpIOp>(loc,
                                                             mlir::arith::CmpIPredicate::sge,
                                                             ih,
                                                             zero_idx);
                auto ih_lt = wb.create<mlir::arith::CmpIOp>(loc,
                                                            mlir::arith::CmpIPredicate::slt,
                                                            ih,
                                                            in_h);
                auto iw_ge0 = wb.create<mlir::arith::CmpIOp>(loc,
                                                             mlir::arith::CmpIPredicate::sge,
                                                             iw,
                                                             zero_idx);
                auto iw_lt = wb.create<mlir::arith::CmpIOp>(loc,
                                                            mlir::arith::CmpIPredicate::slt,
                                                            iw,
                                                            in_w);
                auto ih_ok = wb.create<mlir::arith::AndIOp>(loc, ih_ge0, ih_lt);
                auto iw_ok = wb.create<mlir::arith::AndIOp>(loc, iw_ge0, iw_lt);
                auto in_bounds = wb.create<mlir::arith::AndIOp>(loc, ih_ok, iw_ok);

                auto if_op = wb.create<mlir::scf::IfOp>(loc, elem_ty, in_bounds, true);
                {
                    auto thenb = if_op.getThenBodyBuilder();
                    auto val = thenb.create<mlir::tensor::ExtractOp>(loc,
                                                                     input,
                                                                     mlir::ValueRange{n, c, ih, iw});
                    thenb.create<mlir::scf::YieldOp>(loc, val.getResult());
                }
                {
                    auto elseb = if_op.getElseBodyBuilder();
                    elseb.create<mlir::scf::YieldOp>(loc, zero_val.getResult());
                }

                auto w_val = wb.create<mlir::tensor::ExtractOp>(loc,
                                                                 weight,
                                                                 mlir::ValueRange{c, zero_idx, zero_idx, kh_iv, kw_iv});
                auto mul = wb.create<mlir::arith::MulFOp>(loc, if_op.getResult(0), w_val.getResult());
                auto acc_next = wb.create<mlir::arith::AddFOp>(loc, acc, mul.getResult());
                wb.create<mlir::scf::YieldOp>(loc, acc_next.getResult());
            }
            kb.create<mlir::scf::YieldOp>(loc, kw_loop.getResult(0));
        }

        b.create<mlir::tensor::YieldOp>(loc, kh_loop.getResult(0));
    }

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{gen.getResult()});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
