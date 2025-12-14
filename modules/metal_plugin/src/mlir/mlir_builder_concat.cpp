// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "kernel_ir/kernel_ir_common.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_concat_from_op(const KernelOp& op, mlir::MLIRContext& ctx) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Concat, "Concat builder expects Concat op");
    ctx.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();

    // Flattened view: input [total] -> output [total + axis_offset*inner]
    mlir::Type elem_ty = mlir::Float32Type::get(&ctx);
    if (op.output && op.output->dtype.ov_type == ov::element::f16)
        elem_ty = mlir::Float16Type::get(&ctx);
    else if (op.output && op.output->dtype.ov_type == ov::element::i32)
        elem_ty = mlir::IntegerType::get(&ctx, 32);
    else if (op.output && op.output->dtype.ov_type == ov::element::i64)
        elem_ty = mlir::IntegerType::get(&ctx, 64);

    auto memref_ty = mlir::MemRefType::get(mlir::ShapedType::kDynamic, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({memref_ty, memref_ty}, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "concat_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto input = func.getArgument(0);
    auto output = func.getArgument(1);

    const int64_t axis_offset = static_cast<int64_t>(op.concat.axis_offsets.empty() ? 0 : op.concat.axis_offsets[0]);
    auto axis_offset_c = b.create<mlir::arith::ConstantIndexOp>(loc, axis_offset);
    auto inner_c = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(op.concat.inner));
    auto axis_len_c = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(op.concat.axis_sizes.empty() ? 0 : op.concat.axis_sizes[0]));
    auto axis_total_c = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(op.concat.axis_total));
    auto outer_c = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(op.concat.outer));

    // total elements in this slice: outer * axis_len * inner
    auto total = b.create<mlir::arith::MulIOp>(loc, outer_c,
                   b.create<mlir::arith::MulIOp>(loc, axis_len_c, inner_c));

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto for_i = b.create<mlir::scf::ForOp>(loc, c0, total, c1, std::nullopt,
        [&](mlir::OpBuilder& bb, mlir::Location loc, mlir::Value i, mlir::ValueRange) {
            // Decode flat index i -> (outer, axis, inner)
            auto axis_stride = bb.create<mlir::arith::MulIOp>(loc, axis_len_c, inner_c);
            auto outer = bb.create<mlir::arith::DivUIOp>(loc, i, axis_stride);
            auto rem0 = bb.create<mlir::arith::SubIOp>(loc, i, bb.create<mlir::arith::MulIOp>(loc, outer, axis_stride));
            auto axis = bb.create<mlir::arith::DivUIOp>(loc, rem0, inner_c);
            auto inner = bb.create<mlir::arith::SubIOp>(loc, rem0, bb.create<mlir::arith::MulIOp>(loc, axis, inner_c));

            auto dst_outer_stride = bb.create<mlir::arith::MulIOp>(loc, outer, axis_total_c);
            auto dst_axis = bb.create<mlir::arith::AddIOp>(loc, axis_offset_c, axis);
            auto dst_flat = bb.create<mlir::arith::AddIOp>(loc,
                               bb.create<mlir::arith::MulIOp>(loc,
                                   bb.create<mlir::arith::AddIOp>(loc, dst_outer_stride, dst_axis),
                                   inner_c),
                               inner);
            auto src_outer_stride = bb.create<mlir::arith::MulIOp>(loc, outer, axis_stride);
            auto src_flat = bb.create<mlir::arith::AddIOp>(loc, src_outer_stride,
                              bb.create<mlir::arith::AddIOp>(loc, bb.create<mlir::arith::MulIOp>(loc, axis, inner_c), inner));

            auto val = bb.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{src_flat});
            bb.create<mlir::memref::StoreOp>(loc, val, output, mlir::ValueRange{dst_flat});
            bb.create<mlir::scf::YieldOp>(loc);
        });

    b.setInsertionPointAfter(for_i);
    b.create<mlir::func::ReturnOp>(loc);

    return module;
}

}  // namespace metal_plugin
}  // namespace ov
