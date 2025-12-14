// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "kernel_ir/kernel_ir_common.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

// Simple elementwise convert using memref dim to drive loop bounds (1D flattened).
mlir::ModuleOp build_mlir_convert_from_op(const KernelOp& op, mlir::MLIRContext& ctx) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Convert, "Convert builder expects Convert op");
    ctx.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();

    mlir::Type dst_ty = mlir::Float32Type::get(&ctx);
    switch (op.convert.dst_dtype.ov_type) {
        case ov::element::f16: dst_ty = mlir::Float16Type::get(&ctx); break;
        case ov::element::i32: dst_ty = mlir::IntegerType::get(&ctx, 32); break;
        case ov::element::i64: dst_ty = mlir::IntegerType::get(&ctx, 64); break;
        default: break;
    }
    mlir::Type src_ty = mlir::Float32Type::get(&ctx);
    switch (op.convert.src_dtype.ov_type) {
        case ov::element::f16: src_ty = mlir::Float16Type::get(&ctx); break;
        case ov::element::i32: src_ty = mlir::IntegerType::get(&ctx, 32); break;
        case ov::element::i64: src_ty = mlir::IntegerType::get(&ctx, 64); break;
        default: break;
    }
    auto noneFM = mlir::arith::FastMathFlagsAttr::get(&ctx, mlir::arith::FastMathFlags::none);

    auto src_memref = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, src_ty);
    auto dst_memref = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, dst_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({src_memref, dst_memref}, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "convert_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto src = func.getArgument(0);
    auto dst = func.getArgument(1);

    auto total = b.create<mlir::memref::DimOp>(loc, src, 0);
    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto for_i = b.create<mlir::scf::ForOp>(loc, c0, total, c1, std::nullopt,
        [&](mlir::OpBuilder& bb, mlir::Location loc, mlir::Value i, mlir::ValueRange) {
            auto v = bb.create<mlir::memref::LoadOp>(loc, src, mlir::ValueRange{i});
            mlir::Value casted = v;
            auto srcElemTy = v.getType();
            if (srcElemTy == dst_ty) {
                casted = v;
            } else if (mlir::isa<mlir::FloatType>(srcElemTy) && mlir::isa<mlir::FloatType>(dst_ty)) {
                auto srcF = mlir::cast<mlir::FloatType>(srcElemTy);
                auto dstF = mlir::cast<mlir::FloatType>(dst_ty);
                if (dstF.getWidth() > srcF.getWidth())
                    casted = bb.create<mlir::arith::ExtFOp>(loc, dst_ty, v, noneFM);
                else
                    casted = bb.create<mlir::arith::TruncFOp>(loc, dst_ty, v,
                                                              mlir::arith::RoundingModeAttr(), noneFM);
            } else if (mlir::isa<mlir::IntegerType>(srcElemTy) && mlir::isa<mlir::FloatType>(dst_ty)) {
                casted = bb.create<mlir::arith::SIToFPOp>(loc, dst_ty, v);
            } else if (mlir::isa<mlir::FloatType>(srcElemTy) && mlir::isa<mlir::IntegerType>(dst_ty)) {
                casted = bb.create<mlir::arith::FPToSIOp>(loc, dst_ty, v);
            } else if (mlir::isa<mlir::IntegerType>(srcElemTy) && mlir::isa<mlir::IntegerType>(dst_ty)) {
                auto srcI = mlir::cast<mlir::IntegerType>(srcElemTy);
                auto dstI = mlir::cast<mlir::IntegerType>(dst_ty);
                if (dstI.getWidth() > srcI.getWidth())
                    casted = bb.create<mlir::arith::ExtSIOp>(loc, dst_ty, v);
                else
                    casted = bb.create<mlir::arith::TruncIOp>(loc, dst_ty, v);
            } else {
                casted = bb.create<mlir::arith::BitcastOp>(loc, dst_ty, v);
            }
            bb.create<mlir::memref::StoreOp>(loc, casted, dst, mlir::ValueRange{i});
            bb.create<mlir::scf::YieldOp>(loc);
        });
    b.setInsertionPointAfter(for_i);
    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
