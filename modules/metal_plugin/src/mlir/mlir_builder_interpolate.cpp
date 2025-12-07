// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"

#include "openvino/core/except.hpp"
#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_interpolate_from_op(const KernelOp& op, mlir::MLIRContext& ctx) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Interpolate, "Interpolate builder expects Interpolate op");
    ctx.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect, mlir::math::MathDialect, mlir::scf::SCFDialect>();

    mlir::Type elem_ty = mlir::Float32Type::get(&ctx);
    if (op.output && op.output->dtype.ov_type == ov::element::f16)
        elem_ty = mlir::Float16Type::get(&ctx);

    auto input_ty = mlir::MemRefType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic,
                                            mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, elem_ty);
    auto output_ty = mlir::MemRefType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic,
                                             mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({input_ty, output_ty}, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "interpolate_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto input = func.getArgument(0);
    auto output = func.getArgument(1);

    auto N = b.create<mlir::memref::DimOp>(loc, input, 0);
    auto C = b.create<mlir::memref::DimOp>(loc, input, 1);
    auto H_in = b.create<mlir::memref::DimOp>(loc, input, 2);
    auto W_in = b.create<mlir::memref::DimOp>(loc, input, 3);
    auto H_out = b.create<mlir::memref::DimOp>(loc, output, 2);
    auto W_out = b.create<mlir::memref::DimOp>(loc, output, 3);

    auto to_f32 = [&](mlir::Value idx) {
        auto i64 = b.create<mlir::arith::IndexCastOp>(loc, b.getI64Type(), idx);
        return b.create<mlir::arith::SIToFPOp>(loc, b.getF32Type(), i64);
    };
    auto H_in_f = to_f32(H_in);
    auto W_in_f = to_f32(W_in);
    auto H_out_f = to_f32(H_out);
    auto W_out_f = to_f32(W_out);
    auto scale_h = b.create<mlir::arith::DivFOp>(loc, H_out_f, H_in_f);
    auto scale_w = b.create<mlir::arith::DivFOp>(loc, W_out_f, W_in_f);
    auto align_corners = op.interpolate.align_corners;
    auto nearest = op.interpolate.nearest;

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto fpToIndex = [&](mlir::OpBuilder& bb, mlir::Value fval) {
        auto i64 = bb.create<mlir::arith::FPToSIOp>(loc, b.getI64Type(), fval);
        return bb.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), i64);
    };

    auto for_n = b.create<mlir::scf::ForOp>(loc, c0, N, c1, std::nullopt,
        [&](mlir::OpBuilder& bn, mlir::Location loc, mlir::Value n, mlir::ValueRange) {
            bn.create<mlir::scf::ForOp>(loc, c0, C, c1, std::nullopt,
                [&](mlir::OpBuilder& bc, mlir::Location loc, mlir::Value c, mlir::ValueRange) {
                    bc.create<mlir::scf::ForOp>(loc, c0, H_out, c1, std::nullopt,
                        [&](mlir::OpBuilder& bh, mlir::Location loc, mlir::Value h_idx, mlir::ValueRange) {
                            bh.create<mlir::scf::ForOp>(loc, c0, W_out, c1, std::nullopt,
                                [&](mlir::OpBuilder& bw, mlir::Location loc, mlir::Value w_idx, mlir::ValueRange) {
                                    // Compute fh, fw (float)
                                    mlir::Value fh, fw;
                                    if (align_corners) {
                                        auto h_in_minus1 = bw.create<mlir::arith::SubIOp>(loc, H_in, c1);
                                        auto h_out_minus1 = bw.create<mlir::arith::SubIOp>(loc, H_out, c1);
                                        auto h_in_m1_f = to_f32(h_in_minus1);
                                        auto h_out_m1_f = to_f32(h_out_minus1);
                                        auto h_f = to_f32(h_idx);
                                        auto num = bw.create<mlir::arith::MulFOp>(loc, h_f, h_in_m1_f);
                                        fh = bw.create<mlir::arith::DivFOp>(loc, num, h_out_m1_f);

                                        auto w_in_minus1 = bw.create<mlir::arith::SubIOp>(loc, W_in, c1);
                                        auto w_out_minus1 = bw.create<mlir::arith::SubIOp>(loc, W_out, c1);
                                        auto w_in_m1_f = to_f32(w_in_minus1);
                                        auto w_out_m1_f = to_f32(w_out_minus1);
                                        auto w_f = to_f32(w_idx);
                                        auto numw = bw.create<mlir::arith::MulFOp>(loc, w_f, w_in_m1_f);
                                        fw = bw.create<mlir::arith::DivFOp>(loc, numw, w_out_m1_f);
                                    } else {
                                        auto h_f = to_f32(h_idx);
                                        auto w_f = to_f32(w_idx);
                                        auto half = bw.create<mlir::arith::ConstantOp>(loc, b.getF32FloatAttr(0.5f));
                                        auto h_plus = bw.create<mlir::arith::AddFOp>(loc, h_f, half);
                                        auto w_plus = bw.create<mlir::arith::AddFOp>(loc, w_f, half);
                                        auto h_scaled = bw.create<mlir::arith::MulFOp>(loc, h_plus, scale_h);
                                        auto w_scaled = bw.create<mlir::arith::MulFOp>(loc, w_plus, scale_w);
                                        fh = bw.create<mlir::arith::SubFOp>(loc, h_scaled, half);
                                        fw = bw.create<mlir::arith::SubFOp>(loc, w_scaled, half);
                                    }

                                    mlir::Value out_val;
                                    if (nearest) {
                                        auto half = bw.create<mlir::arith::ConstantOp>(loc, b.getF32FloatAttr(0.5f));
                                        auto fh_r_f = bw.create<mlir::arith::AddFOp>(loc, fh, half);
                                        auto fw_r_f = bw.create<mlir::arith::AddFOp>(loc, fw, half);
                                        auto fh_round = fpToIndex(bw, bw.create<mlir::math::FloorOp>(loc, fh_r_f));
                                        auto fw_round = fpToIndex(bw, bw.create<mlir::math::FloorOp>(loc, fw_r_f));
                                        auto h_clamp = bw.create<mlir::arith::MaxSIOp>(loc, c0, bw.create<mlir::arith::MinSIOp>(loc, fh_round, bw.create<mlir::arith::SubIOp>(loc, H_in, c1)));
                                        auto w_clamp = bw.create<mlir::arith::MaxSIOp>(loc, c0, bw.create<mlir::arith::MinSIOp>(loc, fw_round, bw.create<mlir::arith::SubIOp>(loc, W_in, c1)));
                                        out_val = bw.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{n, c, h_clamp, w_clamp});
                                    } else {
                                        auto fh_floor = bw.create<mlir::math::FloorOp>(loc, fh);
                                        auto fw_floor = bw.create<mlir::math::FloorOp>(loc, fw);
                                        auto h0 = fpToIndex(bw, fh_floor);
                                        auto w0 = fpToIndex(bw, fw_floor);
                                        auto h1 = bw.create<mlir::arith::MinSIOp>(loc, bw.create<mlir::arith::AddIOp>(loc, h0, c1), bw.create<mlir::arith::SubIOp>(loc, H_in, c1));
                                        auto w1 = bw.create<mlir::arith::MinSIOp>(loc, bw.create<mlir::arith::AddIOp>(loc, w0, c1), bw.create<mlir::arith::SubIOp>(loc, W_in, c1));
                                        auto dh = bw.create<mlir::arith::SubFOp>(loc, fh, bw.create<mlir::arith::SIToFPOp>(loc, b.getF32Type(), h0));
                                        auto dw = bw.create<mlir::arith::SubFOp>(loc, fw, bw.create<mlir::arith::SIToFPOp>(loc, b.getF32Type(), w0));

                                        auto v00 = bw.create<mlir::arith::SIToFPOp>(loc, b.getF32Type(), bw.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{n, c, h0, w0}));
                                        auto v01 = bw.create<mlir::arith::SIToFPOp>(loc, b.getF32Type(), bw.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{n, c, h0, w1}));
                                        auto v10 = bw.create<mlir::arith::SIToFPOp>(loc, b.getF32Type(), bw.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{n, c, h1, w0}));
                                        auto v11 = bw.create<mlir::arith::SIToFPOp>(loc, b.getF32Type(), bw.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{n, c, h1, w1}));
                                        auto v0 = bw.create<mlir::arith::AddFOp>(loc, v00, bw.create<mlir::arith::MulFOp>(loc, dh, bw.create<mlir::arith::SubFOp>(loc, v10, v00)));
                                        auto v1 = bw.create<mlir::arith::AddFOp>(loc, v01, bw.create<mlir::arith::MulFOp>(loc, dh, bw.create<mlir::arith::SubFOp>(loc, v11, v01)));
                                        auto v = bw.create<mlir::arith::AddFOp>(loc, v0, bw.create<mlir::arith::MulFOp>(loc, dw, bw.create<mlir::arith::SubFOp>(loc, v1, v0)));
                                        if (elem_ty == mlir::Float16Type::get(&ctx))
                                            out_val = bw.create<mlir::arith::TruncFOp>(loc, elem_ty, v);
                                        else
                                            out_val = v;
                                    }

                                    bw.create<mlir::memref::StoreOp>(loc, out_val, output, mlir::ValueRange{n, c, h_idx, w_idx});
                                    bw.create<mlir::scf::YieldOp>(loc);
                                });
                            bh.create<mlir::scf::YieldOp>(loc);
                        });
                    bc.create<mlir::scf::YieldOp>(loc);
                });
            bn.create<mlir::scf::YieldOp>(loc);
        });

    b.setInsertionPointAfter(for_n);
    b.create<mlir::func::ReturnOp>(loc);

    return module;
}

}  // namespace metal_plugin
}  // namespace ov
