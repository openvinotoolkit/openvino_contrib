// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_split_from_op(const KernelOp& op, mlir::MLIRContext& ctx) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Split || op.kind == KernelOpKind::Slice,
                    "Split builder expects Split/Slice op");
    ctx.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();

    mlir::Type elem_ty = mlir::Float32Type::get(&ctx);
    if (op.output && op.output->dtype.ov_type == ov::element::f16)
        elem_ty = mlir::Float16Type::get(&ctx);

    auto input_ty = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elem_ty);
    auto output_ty = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({input_ty, output_ty}, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "split_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto input = func.getArgument(0);
    auto output = func.getArgument(1);

    std::vector<int64_t> ishape;
    int64_t axis = 0;
    int64_t axis_dim = 0;
    int64_t split_size = 0;
    int64_t axis_offset = 0;
    int64_t inner = 1;

    if (op.kind == KernelOpKind::Split) {
        ishape.assign(op.split.input_shape.begin(), op.split.input_shape.end());
        axis = op.split.axis;
        axis_dim = (axis >= 0 && axis < static_cast<int64_t>(ishape.size()))
                       ? ishape[static_cast<size_t>(axis)]
                       : 0;
        split_size = static_cast<int64_t>(op.split.split_sizes.empty() ? 0 : op.split.split_sizes[0]);
        axis_offset = static_cast<int64_t>(op.split.axis_offsets.empty() ? 0 : op.split.axis_offsets[0]);
        inner = static_cast<int64_t>(op.split.inner);
    } else {  // Slice
        ishape.assign(op.slice.in_shape.begin(), op.slice.in_shape.end());
        // infer axis: first dim where out_shape differs or start non-zero
        const auto& osh = op.slice.out_shape;
        axis = 0;
        for (size_t i = 0; i < ishape.size(); ++i) {
            if (i >= osh.size() || ishape[i] != osh[i] || (i < op.slice.starts.size() && op.slice.starts[i] != 0)) {
                axis = static_cast<int64_t>(i);
                break;
            }
        }
        axis_dim = ishape.empty() ? 0 : ishape[static_cast<size_t>(axis)];
        split_size = (axis < static_cast<int64_t>(osh.size())) ? osh[static_cast<size_t>(axis)] : 0;
        axis_offset = (axis < static_cast<int64_t>(op.slice.starts.size())) ? op.slice.starts[static_cast<size_t>(axis)] : 0;
        for (size_t k = static_cast<size_t>(axis + 1); k < ishape.size(); ++k) inner *= static_cast<int64_t>(ishape[k]);
    }
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= ishape[static_cast<size_t>(i)];

    auto c_axis_total = b.create<mlir::arith::ConstantIndexOp>(loc, axis_dim);
    auto c_axis_offset = b.create<mlir::arith::ConstantIndexOp>(loc, axis_offset);
    auto c_split = b.create<mlir::arith::ConstantIndexOp>(loc, split_size);
    auto c_inner = b.create<mlir::arith::ConstantIndexOp>(loc, inner);
    auto c_outer = b.create<mlir::arith::ConstantIndexOp>(loc, outer);
    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto for_outer = b.create<mlir::scf::ForOp>(loc, c0, c_outer, c1, std::nullopt,
        [&](mlir::OpBuilder& bo, mlir::Location loc, mlir::Value o, mlir::ValueRange){
            auto for_axis = bo.create<mlir::scf::ForOp>(loc, c0, c_split, c1, std::nullopt,
                [&](mlir::OpBuilder& ba, mlir::Location loc, mlir::Value a, mlir::ValueRange){
                    auto for_in = ba.create<mlir::scf::ForOp>(loc, c0, c_inner, c1, std::nullopt,
                        [&](mlir::OpBuilder& bi, mlir::Location loc, mlir::Value i, mlir::ValueRange){
                            auto outer_stride = bi.create<mlir::arith::MulIOp>(loc, o,
                                bi.create<mlir::arith::MulIOp>(loc, c_axis_total, c_inner));
                            auto src_idx = bi.create<mlir::arith::AddIOp>(loc, outer_stride,
                                bi.create<mlir::arith::AddIOp>(loc,
                                    bi.create<mlir::arith::MulIOp>(loc,
                                        bi.create<mlir::arith::AddIOp>(loc, c_axis_offset, a),
                                        c_inner),
                                    i));
                            auto dst_idx = bi.create<mlir::arith::AddIOp>(loc,
                                bi.create<mlir::arith::MulIOp>(loc,
                                    bi.create<mlir::arith::MulIOp>(loc, o, c_split),
                                    c_inner),
                                bi.create<mlir::arith::AddIOp>(loc,
                                    bi.create<mlir::arith::MulIOp>(loc, a, c_inner), i));
                            auto val = bi.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{src_idx});
                            bi.create<mlir::memref::StoreOp>(loc, val, output, mlir::ValueRange{dst_idx});
                            bi.create<mlir::scf::YieldOp>(loc);
                        });
                    ba.create<mlir::scf::YieldOp>(loc);
                });
            bo.create<mlir::scf::YieldOp>(loc);
        });

    b.setInsertionPointAfter(for_outer);
    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
