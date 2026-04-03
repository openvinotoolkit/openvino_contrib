// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/conv_im2col_parallel_lowering.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "runtime/gfx_logger.hpp"

#include <optional>

namespace ov {
namespace gfx_plugin {

namespace {

mlir::Value strip_memref_casts_impl(mlir::Value value) {
    mlir::Value current = value;
    while (current) {
        if (auto cast = current.getDefiningOp<mlir::memref::CastOp>()) {
            current = cast.getSource();
            continue;
        }
        if (auto cast = current.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
            current = cast.getSource();
            continue;
        }
        if (auto subview = current.getDefiningOp<mlir::memref::SubViewOp>()) {
            current = subview.getSource();
            continue;
        }
        break;
    }
    return current;
}

mlir::Value get_dim(mlir::OpBuilder& b, mlir::Location loc, mlir::Value value, int64_t dim) {
    auto mem_ty = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (mem_ty && dim < mem_ty.getRank()) {
        const int64_t sz = mem_ty.getDimSize(dim);
        if (sz != mlir::ShapedType::kDynamic) {
            return b.create<mlir::arith::ConstantIndexOp>(loc, sz);
        }
    }
    return b.create<mlir::memref::DimOp>(loc, value, dim);
}

std::optional<int64_t> int_attr(mlir::Operation* op, llvm::StringRef name) {
    if (!op) {
        return std::nullopt;
    }
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr) {
        return std::nullopt;
    }
    return attr.getInt();
}

llvm::StringRef stage_attr(mlir::Operation* op) {
    if (!op) {
        return {};
    }
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("gfx.im2col_stage")) {
        return attr.getValue();
    }
    return {};
}

bool lower_im2col_extract(mlir::linalg::GenericOp op, mlir::IRRewriter& rewriter) {
    auto input = strip_memref_casts_impl(op.getDpsInputs()[0]);
    auto output = strip_memref_casts_impl(op.getDpsInits()[0]);
    auto input_ty = mlir::dyn_cast<mlir::MemRefType>(input.getType());
    auto output_ty = mlir::dyn_cast<mlir::MemRefType>(output.getType());
    if (!input_ty || !output_ty || input_ty.getRank() != 4 || (output_ty.getRank() != 2 && output_ty.getRank() != 3)) {
        return false;
    }
    const auto out_w = int_attr(op, "gfx.im2col_out_w");
    const auto kernel_h = int_attr(op, "gfx.im2col_kernel_h");
    const auto kernel_w = int_attr(op, "gfx.im2col_kernel_w");
    const auto stride_h = int_attr(op, "gfx.im2col_stride_h");
    const auto stride_w = int_attr(op, "gfx.im2col_stride_w");
    const auto dil_h = int_attr(op, "gfx.im2col_dil_h");
    const auto dil_w = int_attr(op, "gfx.im2col_dil_w");
    if (!out_w || !kernel_h || !kernel_w || !stride_h || !stride_w || !dil_h || !dil_w) {
        return false;
    }
    const bool transposed = op->getAttrOfType<mlir::BoolAttr>("gfx.im2col_transposed") &&
                            op->getAttrOfType<mlir::BoolAttr>("gfx.im2col_transposed").getValue();

    const auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto outW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *out_w);
    auto kH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *kernel_h);
    auto kW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *kernel_w);
    auto strideH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *stride_h);
    auto strideW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *stride_w);
    auto dilH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *dil_h);
    auto dilW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *dil_w);
    auto kernelInner = rewriter.create<mlir::arith::MulIOp>(loc, kH, kW);

    mlir::scf::ParallelOp par;
    if (output_ty.getRank() == 3) {
        auto B = get_dim(rewriter, loc, output, 0);
        auto M = get_dim(rewriter, loc, output, 1);
        auto K = get_dim(rewriter, loc, output, 2);
        par = rewriter.create<mlir::scf::ParallelOp>(
            loc,
            mlir::ValueRange{c0, c0, c0},
            mlir::ValueRange{B, M, K},
            mlir::ValueRange{c1, c1, c1});
    } else if (transposed) {
        auto K = get_dim(rewriter, loc, output, 0);
        auto M = get_dim(rewriter, loc, output, 1);
        par = rewriter.create<mlir::scf::ParallelOp>(
            loc,
            mlir::ValueRange{c0, c0},
            mlir::ValueRange{K, M},
            mlir::ValueRange{c1, c1});
    } else {
        auto M = get_dim(rewriter, loc, output, 0);
        auto K = get_dim(rewriter, loc, output, 1);
        par = rewriter.create<mlir::scf::ParallelOp>(
            loc,
            mlir::ValueRange{c0, c0},
            mlir::ValueRange{M, K},
            mlir::ValueRange{c1, c1});
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(par.getBody()->getTerminator());
    auto ivs = par.getInductionVars();
    auto batch_idx = output_ty.getRank() == 3 ? ivs[0] : c0;
    auto m_idx = output_ty.getRank() == 3 ? ivs[1] : (transposed ? ivs[1] : ivs[0]);
    auto k_idx = output_ty.getRank() == 3 ? ivs[2] : (transposed ? ivs[0] : ivs[1]);
    auto oh = rewriter.create<mlir::arith::DivSIOp>(loc, m_idx, outW);
    auto ow = rewriter.create<mlir::arith::RemSIOp>(loc, m_idx, outW);
    auto c = rewriter.create<mlir::arith::DivSIOp>(loc, k_idx, kernelInner);
    auto kRem = rewriter.create<mlir::arith::RemSIOp>(loc, k_idx, kernelInner);
    auto kh = rewriter.create<mlir::arith::DivSIOp>(loc, kRem, kW);
    auto kw = rewriter.create<mlir::arith::RemSIOp>(loc, kRem, kW);
    auto ih = rewriter.create<mlir::arith::AddIOp>(
        loc,
        rewriter.create<mlir::arith::MulIOp>(loc, oh, strideH),
        rewriter.create<mlir::arith::MulIOp>(loc, kh, dilH));
    auto iw = rewriter.create<mlir::arith::AddIOp>(
        loc,
        rewriter.create<mlir::arith::MulIOp>(loc, ow, strideW),
        rewriter.create<mlir::arith::MulIOp>(loc, kw, dilW));
    auto value = rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{batch_idx, c, ih, iw});
    if (output_ty.getRank() == 3) {
        rewriter.create<mlir::memref::StoreOp>(loc, value, output, mlir::ValueRange{batch_idx, m_idx, k_idx});
    } else if (transposed) {
        rewriter.create<mlir::memref::StoreOp>(loc, value, output, mlir::ValueRange{k_idx, m_idx});
    } else {
        rewriter.create<mlir::memref::StoreOp>(loc, value, output, mlir::ValueRange{m_idx, k_idx});
    }

    if (op->getNumResults() > 0) {
        op.getResult(0).replaceAllUsesWith(op.getDpsInits()[0]);
    }
    rewriter.eraseOp(op);
    return true;
}

bool lower_im2col_pack_weight(mlir::linalg::GenericOp op, mlir::IRRewriter& rewriter) {
    auto input = strip_memref_casts_impl(op.getDpsInputs()[0]);
    auto output = strip_memref_casts_impl(op.getDpsInits()[0]);
    auto input_ty = mlir::dyn_cast<mlir::MemRefType>(input.getType());
    auto output_ty = mlir::dyn_cast<mlir::MemRefType>(output.getType());
    if (!input_ty || !output_ty || input_ty.getRank() != 4 || (output_ty.getRank() != 2 && output_ty.getRank() != 3)) {
        return false;
    }
    const auto kernel_h = int_attr(op, "gfx.im2col_kernel_h");
    const auto kernel_w = int_attr(op, "gfx.im2col_kernel_w");
    if (!kernel_h || !kernel_w) {
        return false;
    }

    const auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto kH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *kernel_h);
    auto kW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *kernel_w);
    auto kernelInner = rewriter.create<mlir::arith::MulIOp>(loc, kH, kW);

    auto K = get_dim(rewriter, loc, output, output_ty.getRank() == 3 ? 1 : 0);
    auto N = get_dim(rewriter, loc, output, output_ty.getRank() == 3 ? 2 : 1);

    auto par = rewriter.create<mlir::scf::ParallelOp>(
        loc,
        mlir::ValueRange{c0, c0},
        mlir::ValueRange{K, N},
        mlir::ValueRange{c1, c1});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(par.getBody()->getTerminator());
    auto ivs = par.getInductionVars();
    auto c = rewriter.create<mlir::arith::DivSIOp>(loc, ivs[0], kernelInner);
    auto kRem = rewriter.create<mlir::arith::RemSIOp>(loc, ivs[0], kernelInner);
    auto kh = rewriter.create<mlir::arith::DivSIOp>(loc, kRem, kW);
    auto kw = rewriter.create<mlir::arith::RemSIOp>(loc, kRem, kW);
    auto value = rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{ivs[1], c, kh, kw});
    if (output_ty.getRank() == 3) {
        rewriter.create<mlir::memref::StoreOp>(loc, value, output, mlir::ValueRange{c0, ivs[0], ivs[1]});
    } else {
        rewriter.create<mlir::memref::StoreOp>(loc, value, output, mlir::ValueRange{ivs[0], ivs[1]});
    }

    if (op->getNumResults() > 0) {
        op.getResult(0).replaceAllUsesWith(op.getDpsInits()[0]);
    }
    rewriter.eraseOp(op);
    return true;
}

bool lower_im2col_restore_output(mlir::linalg::GenericOp op, mlir::IRRewriter& rewriter) {
    auto input = strip_memref_casts_impl(op.getDpsInputs()[0]);
    auto output = strip_memref_casts_impl(op.getDpsInits()[0]);
    auto input_ty = mlir::dyn_cast<mlir::MemRefType>(input.getType());
    auto output_ty = mlir::dyn_cast<mlir::MemRefType>(output.getType());
    if (!input_ty || !output_ty || output_ty.getRank() != 4 || (input_ty.getRank() != 2 && input_ty.getRank() != 3)) {
        return false;
    }
    const auto out_w = int_attr(op, "gfx.im2col_out_w");
    if (!out_w) {
        return false;
    }
    const bool transposed_input = op->getAttrOfType<mlir::BoolAttr>("gfx.im2col_restore_transposed") &&
                                  op->getAttrOfType<mlir::BoolAttr>("gfx.im2col_restore_transposed").getValue();

    const auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto outW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, *out_w);

    auto B = get_dim(rewriter, loc, output, 0);
    auto N = get_dim(rewriter, loc, output, 1);
    auto OH = get_dim(rewriter, loc, output, 2);
    auto OW = get_dim(rewriter, loc, output, 3);

    auto par = rewriter.create<mlir::scf::ParallelOp>(
        loc,
        mlir::ValueRange{c0, c0, c0, c0},
        mlir::ValueRange{B, N, OH, OW},
        mlir::ValueRange{c1, c1, c1, c1});

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(par.getBody()->getTerminator());
    auto ivs = par.getInductionVars();
    auto m = rewriter.create<mlir::arith::AddIOp>(
        loc,
        rewriter.create<mlir::arith::MulIOp>(loc, ivs[2], outW),
        ivs[3]);
    mlir::Value value;
    if (input_ty.getRank() == 3) {
        value = rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{ivs[0], m, ivs[1]});
    } else if (transposed_input) {
        value = rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{ivs[1], m});
    } else {
        value = rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{m, ivs[1]});
    }
    rewriter.create<mlir::memref::StoreOp>(loc, value, output, mlir::ValueRange{ivs[0], ivs[1], ivs[2], ivs[3]});

    if (op->getNumResults() > 0) {
        op.getResult(0).replaceAllUsesWith(op.getDpsInits()[0]);
    }
    rewriter.eraseOp(op);
    return true;
}

bool lower_op(mlir::linalg::GenericOp op, mlir::IRRewriter& rewriter) {
    const auto stage = stage_attr(op);
    if (stage.empty()) {
        return false;
    }
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) {
        return false;
    }
    if (stage == "extract") {
        return lower_im2col_extract(op, rewriter);
    }
    if (stage == "pack_weight") {
        return lower_im2col_pack_weight(op, rewriter);
    }
    if (stage == "restore_output") {
        return lower_im2col_restore_output(op, rewriter);
    }
    return false;
}

}  // namespace

void run_conv_im2col_parallel_lowering(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    mlir::IRRewriter rewriter(module.getContext());
    llvm::SmallVector<mlir::linalg::GenericOp, 8> ops;
    module.walk([&](mlir::linalg::GenericOp op) {
        if (!stage_attr(op).empty()) {
            ops.push_back(op);
        }
    });
    size_t rewritten = 0;
    for (auto op : ops) {
        if (!op || !op->getParentOp()) {
            continue;
        }
        if (lower_op(op, rewriter)) {
            ++rewritten;
        }
    }
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIR") << "Im2Col parallel lowering: ops=" << ops.size()
                              << " rewritten=" << rewritten;
    }
}

}  // namespace gfx_plugin
}  // namespace ov
