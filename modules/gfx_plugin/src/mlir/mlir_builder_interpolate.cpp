// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_interpolate_from_model(const std::shared_ptr<const ov::Model>& model,
                                                 mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::math::MathDialect, mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::Node> interp;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v0::Interpolate>(node) ||
            ov::as_type_ptr<const ov::op::v4::Interpolate>(node) ||
            ov::as_type_ptr<const ov::op::v11::Interpolate>(node)) {
            OPENVINO_ASSERT(!interp, "Interpolate MLIR builder: expected single Interpolate");
            interp = node;
        }
    }
    OPENVINO_ASSERT(interp, "Interpolate MLIR builder: Interpolate op not found");

    auto in_shape = to_shape(interp->get_input_partial_shape(0));
    auto out_shape = to_shape(interp->get_output_partial_shape(0));
    auto elem_ty = to_mlir_type(interp->get_output_element_type(0), ctx);
    auto make_float_attr = [&](double v) {
        if (auto ft = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
            return mlir::FloatAttr::get(ft, v);
        }
        return mlir::FloatAttr::get(mlir::Float32Type::get(&ctx), v);
    };

    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_shape, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "interpolate_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    const auto in_shape_static = interp->get_input_shape(0);
    const auto out_shape_static = interp->get_output_shape(0);
    OPENVINO_ASSERT(in_shape_static.size() == 4 && out_shape_static.size() == 4,
                    "Interpolate MLIR: supports NCHW rank4 only");

    bool nearest = true;
    bool align_corners = false;
    bool use_half_pixel = true;
    if (auto v0 = ov::as_type_ptr<const ov::op::v0::Interpolate>(interp)) {
        const auto& attrs = v0->get_attrs();
        const auto mode = ov::util::to_lower(attrs.mode);
        if (mode == "nearest") {
            nearest = true;
        } else if (mode == "linear") {
            nearest = false;
        } else {
            OPENVINO_THROW("Interpolate MLIR: mode not supported");
        }
        align_corners = attrs.align_corners;
        use_half_pixel = !align_corners;
    } else if (auto v4 = ov::as_type_ptr<const ov::op::v4::Interpolate>(interp)) {
        using Base = ov::op::util::InterpolateBase;
        switch (v4->get_attrs().mode) {
            case Base::InterpolateMode::NEAREST:
                nearest = true;
                break;
            case Base::InterpolateMode::LINEAR:
            case Base::InterpolateMode::LINEAR_ONNX:
            case Base::InterpolateMode::BILINEAR_PILLOW:
                nearest = false;
                break;
            default:
                OPENVINO_THROW("Interpolate MLIR: mode not supported");
        }
        switch (v4->get_attrs().coordinate_transformation_mode) {
            case Base::CoordinateTransformMode::HALF_PIXEL:
                align_corners = false;
                use_half_pixel = true;
                break;
            case Base::CoordinateTransformMode::ALIGN_CORNERS:
                align_corners = true;
                use_half_pixel = true;
                break;
            case Base::CoordinateTransformMode::ASYMMETRIC:
                align_corners = false;
                use_half_pixel = false;
                break;
            default:
                OPENVINO_THROW("Interpolate MLIR: coord transform not supported");
        }
    } else if (auto v11 = ov::as_type_ptr<const ov::op::v11::Interpolate>(interp)) {
        using Base = ov::op::util::InterpolateBase;
        switch (v11->get_attrs().mode) {
            case Base::InterpolateMode::NEAREST:
                nearest = true;
                break;
            case Base::InterpolateMode::LINEAR:
            case Base::InterpolateMode::LINEAR_ONNX:
            case Base::InterpolateMode::BILINEAR_PILLOW:
                nearest = false;
                break;
            default:
                OPENVINO_THROW("Interpolate MLIR: mode not supported");
        }
        switch (v11->get_attrs().coordinate_transformation_mode) {
            case Base::CoordinateTransformMode::HALF_PIXEL:
                align_corners = false;
                use_half_pixel = true;
                break;
            case Base::CoordinateTransformMode::ALIGN_CORNERS:
                align_corners = true;
                use_half_pixel = true;
                break;
            case Base::CoordinateTransformMode::ASYMMETRIC:
                align_corners = false;
                use_half_pixel = false;
                break;
            default:
                OPENVINO_THROW("Interpolate MLIR: coord transform not supported");
        }
    }

    const uint64_t N = in_shape_static[0];
    const uint64_t C = in_shape_static[1];
    const uint64_t H_in = in_shape_static[2];
    const uint64_t W_in = in_shape_static[3];
    const uint64_t H_out = out_shape_static[2];
    const uint64_t W_out = out_shape_static[3];

    mlir::Value in_tensor = func.getArgument(0);
    mlir::Value out_tensor = b.create<mlir::tensor::EmptyOp>(
        loc,
        mlir::ArrayRef<int64_t>({static_cast<int64_t>(N),
                                 static_cast<int64_t>(C),
                                 static_cast<int64_t>(H_out),
                                 static_cast<int64_t>(W_out)}),
        elem_ty);

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto c_total = b.create<mlir::arith::ConstantIndexOp>(loc,
                                                          static_cast<int64_t>(ov::shape_size(out_shape_static)));
    auto c_W_out = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(W_out));
    auto c_H_out = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(H_out));
    auto c_C = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(C));
    auto c_W_in = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(W_in));
    auto c_H_in = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(H_in));

    auto f32 = mlir::Float32Type::get(&ctx);
    auto scale_h = b.create<mlir::arith::ConstantOp>(
                       loc, make_float_attr(static_cast<float>(H_in) / static_cast<float>(H_out)))
                       .getResult();
    auto scale_w = b.create<mlir::arith::ConstantOp>(
                       loc, make_float_attr(static_cast<float>(W_in) / static_cast<float>(W_out)))
                       .getResult();

    auto loop = b.create<mlir::scf::ForOp>(loc, c0, c_total, c1, mlir::ValueRange{out_tensor});
    {
        auto* body = loop.getBody();
        mlir::OpBuilder lb(body, body->begin());
        auto iv = loop.getInductionVar();
        auto acc = loop.getRegionIterArgs()[0];

        auto w = lb.create<mlir::arith::RemSIOp>(loc, iv, c_W_out).getResult();
        auto tmp = lb.create<mlir::arith::DivSIOp>(loc, iv, c_W_out).getResult();
        auto h = lb.create<mlir::arith::RemSIOp>(loc, tmp, c_H_out).getResult();
        tmp = lb.create<mlir::arith::DivSIOp>(loc, tmp, c_H_out).getResult();
        auto c = lb.create<mlir::arith::RemSIOp>(loc, tmp, c_C).getResult();
        auto n = lb.create<mlir::arith::DivSIOp>(loc, tmp, c_C).getResult();

        auto h_i32 = lb.create<mlir::arith::IndexCastOp>(loc, lb.getI32Type(), h).getResult();
        auto w_i32 = lb.create<mlir::arith::IndexCastOp>(loc, lb.getI32Type(), w).getResult();
        mlir::Value fh = lb.create<mlir::arith::SIToFPOp>(loc, f32, h_i32).getResult();
        mlir::Value fw = lb.create<mlir::arith::SIToFPOp>(loc, f32, w_i32).getResult();

        if (align_corners && H_out > 1) {
            auto h_scale = lb.create<mlir::arith::ConstantOp>(
                loc, make_float_attr(static_cast<float>(H_in - 1) / static_cast<float>(H_out - 1))).getResult();
            fh = lb.create<mlir::arith::MulFOp>(loc, fh, h_scale).getResult();
        } else if (use_half_pixel) {
            auto half = lb.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.5f)).getResult();
            auto fh_add = lb.create<mlir::arith::AddFOp>(loc, fh, half).getResult();
            fh = lb.create<mlir::arith::MulFOp>(loc, fh_add, scale_h).getResult();
            fh = lb.create<mlir::arith::SubFOp>(loc, fh, half).getResult();
        } else {
            fh = lb.create<mlir::arith::MulFOp>(loc, fh, scale_h).getResult();
        }
        if (align_corners && W_out > 1) {
            auto w_scale = lb.create<mlir::arith::ConstantOp>(
                loc, make_float_attr(static_cast<float>(W_in - 1) / static_cast<float>(W_out - 1))).getResult();
            fw = lb.create<mlir::arith::MulFOp>(loc, fw, w_scale).getResult();
        } else if (use_half_pixel) {
            auto half = lb.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.5f)).getResult();
            auto fw_add = lb.create<mlir::arith::AddFOp>(loc, fw, half).getResult();
            fw = lb.create<mlir::arith::MulFOp>(loc, fw_add, scale_w).getResult();
            fw = lb.create<mlir::arith::SubFOp>(loc, fw, half).getResult();
        } else {
            fw = lb.create<mlir::arith::MulFOp>(loc, fw, scale_w).getResult();
        }

        auto c1_i32 = lb.create<mlir::arith::ConstantIntOp>(loc, 1, 32).getResult();
        auto h0_i32 = lb.create<mlir::arith::FPToSIOp>(loc, lb.getI32Type(), fh).getResult();
        auto w0_i32 = lb.create<mlir::arith::FPToSIOp>(loc, lb.getI32Type(), fw).getResult();
        auto h0f = lb.create<mlir::arith::SIToFPOp>(loc, f32, h0_i32).getResult();
        auto w0f = lb.create<mlir::arith::SIToFPOp>(loc, f32, w0_i32).getResult();
        auto h_lt = lb.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, fh, h0f).getResult();
        auto w_lt = lb.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, fw, w0f).getResult();
        auto h_adj = lb.create<mlir::arith::SubIOp>(loc, h0_i32, c1_i32).getResult();
        auto w_adj = lb.create<mlir::arith::SubIOp>(loc, w0_i32, c1_i32).getResult();
        h0_i32 = lb.create<mlir::arith::SelectOp>(loc, h_lt, h_adj, h0_i32).getResult();
        w0_i32 = lb.create<mlir::arith::SelectOp>(loc, w_lt, w_adj, w0_i32).getResult();
        h0f = lb.create<mlir::arith::SIToFPOp>(loc, f32, h0_i32).getResult();
        w0f = lb.create<mlir::arith::SIToFPOp>(loc, f32, w0_i32).getResult();
        auto h0 = lb.create<mlir::arith::IndexCastOp>(loc, lb.getIndexType(), h0_i32).getResult();
        auto w0 = lb.create<mlir::arith::IndexCastOp>(loc, lb.getIndexType(), w0_i32).getResult();

        auto clamp_idx = [&](mlir::Value v, mlir::Value maxv) {
            auto zero = lb.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
            auto lt0 = lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, v, zero).getResult();
            auto gt = lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, v, maxv).getResult();
            auto v0 = lb.create<mlir::arith::SelectOp>(loc, lt0, zero, v).getResult();
            return lb.create<mlir::arith::SelectOp>(loc, gt, maxv, v0).getResult();
        };

        auto max_h = lb.create<mlir::arith::SubIOp>(loc, c_H_in, c1).getResult();
        auto max_w = lb.create<mlir::arith::SubIOp>(loc, c_W_in, c1).getResult();
        h0 = clamp_idx(h0, max_h);
        w0 = clamp_idx(w0, max_w);

        mlir::Value value;
        if (nearest) {
            auto nc = lb.create<mlir::arith::AddIOp>(loc,
                                                     lb.create<mlir::arith::MulIOp>(loc, n, c_C).getResult(),
                                                     c).getResult();
            auto nch = lb.create<mlir::arith::AddIOp>(loc,
                                                      lb.create<mlir::arith::MulIOp>(loc, nc, c_H_in).getResult(),
                                                      h0).getResult();
            value = lb.create<mlir::tensor::ExtractOp>(loc,
                                                       in_tensor,
                                                       mlir::ValueRange{n, c, h0, w0})
                        .getResult();
        } else {
            auto h1 = lb.create<mlir::arith::AddIOp>(loc, h0, c1).getResult();
            auto w1 = lb.create<mlir::arith::AddIOp>(loc, w0, c1).getResult();
            h1 = clamp_idx(h1, max_h);
            w1 = clamp_idx(w1, max_w);

            auto v00 = lb.create<mlir::tensor::ExtractOp>(loc,
                                                          in_tensor,
                                                          mlir::ValueRange{n, c, h0, w0})
                          .getResult();
            auto v01 = lb.create<mlir::tensor::ExtractOp>(loc,
                                                          in_tensor,
                                                          mlir::ValueRange{n, c, h0, w1})
                          .getResult();
            auto v10 = lb.create<mlir::tensor::ExtractOp>(loc,
                                                          in_tensor,
                                                          mlir::ValueRange{n, c, h1, w0})
                          .getResult();
            auto v11 = lb.create<mlir::tensor::ExtractOp>(loc,
                                                          in_tensor,
                                                          mlir::ValueRange{n, c, h1, w1})
                          .getResult();

            mlir::Value v00f = v00;
            mlir::Value v01f = v01;
            mlir::Value v10f = v10;
            mlir::Value v11f = v11;
            if (auto ft = mlir::dyn_cast<mlir::FloatType>(elem_ty); ft && ft.getWidth() == 16) {
                v00f = lb.create<mlir::arith::ExtFOp>(loc, f32, v00).getResult();
                v01f = lb.create<mlir::arith::ExtFOp>(loc, f32, v01).getResult();
                v10f = lb.create<mlir::arith::ExtFOp>(loc, f32, v10).getResult();
                v11f = lb.create<mlir::arith::ExtFOp>(loc, f32, v11).getResult();
            }

            auto dh = lb.create<mlir::arith::SubFOp>(loc, fh, h0f).getResult();
            auto dw = lb.create<mlir::arith::SubFOp>(loc, fw, w0f).getResult();
            auto one = lb.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0f)).getResult();
            auto one_minus_dw = lb.create<mlir::arith::SubFOp>(loc, one, dw).getResult();
            auto one_minus_dh = lb.create<mlir::arith::SubFOp>(loc, one, dh).getResult();
            auto v0 = lb.create<mlir::arith::AddFOp>(loc,
                                                     lb.create<mlir::arith::MulFOp>(loc, v00f, one_minus_dw).getResult(),
                                                     lb.create<mlir::arith::MulFOp>(loc, v01f, dw).getResult())
                         .getResult();
            auto v1 = lb.create<mlir::arith::AddFOp>(loc,
                                                     lb.create<mlir::arith::MulFOp>(loc, v10f, one_minus_dw).getResult(),
                                                     lb.create<mlir::arith::MulFOp>(loc, v11f, dw).getResult())
                         .getResult();
            auto v = lb.create<mlir::arith::AddFOp>(loc,
                                                    lb.create<mlir::arith::MulFOp>(loc, v0, one_minus_dh).getResult(),
                                                    lb.create<mlir::arith::MulFOp>(loc, v1, dh).getResult())
                         .getResult();
            if (elem_ty == f32) {
                value = v;
            } else {
                value = lb.create<mlir::arith::TruncFOp>(loc, elem_ty, v).getResult();
            }
        }

        auto updated = lb.create<mlir::tensor::InsertOp>(loc,
                                                         value,
                                                         acc,
                                                         mlir::ValueRange{n, c, h, w})
                           .getResult();
        lb.create<mlir::scf::YieldOp>(loc, updated);
    }
    out_tensor = loop.getResults()[0];

    b.create<mlir::func::ReturnOp>(loc, out_tensor);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
