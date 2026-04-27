// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/conv_im2col_matmul_rewrite.hpp"

#include "mlir/gfx_mlir_debug.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace ov {
namespace gfx_plugin {
namespace {

mlir::Value make_zero(mlir::RewriterBase& rewriter, mlir::Location loc, mlir::Type elem_ty) {
    if (auto float_ty = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
        return rewriter.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(float_ty, 0.0));
    }
    if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
        return rewriter.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(int_ty, 0));
    }
    return {};
}

bool module_requests_im2col(mlir::ModuleOp module) {
    if (!module) {
        return false;
    }
    auto attr = module->getAttrOfType<mlir::StringAttr>("gfx.conv_algorithm_kind");
    return attr && attr.getValue() == "im2col_matmul";
}

bool rewrite_conv_to_im2col_matmul(mlir::linalg::Conv2DNchwFchwOp op, mlir::RewriterBase& rewriter) {
    const bool debug = gfx_mlir_debug_enabled();
    auto fail = [&](const char* reason) {
        if (debug) {
            llvm::errs() << "[GFX][MLIR] im2col rewrite skip: " << reason << "\n";
            llvm::errs() << "  op: ";
            op->print(llvm::errs());
            llvm::errs() << "\n";
        }
        return false;
    };

    auto input_ty = mlir::dyn_cast<mlir::RankedTensorType>(op.getInputs()[0].getType());
    auto filter_ty = mlir::dyn_cast<mlir::RankedTensorType>(op.getInputs()[1].getType());
    auto output_ty = mlir::dyn_cast<mlir::RankedTensorType>(op.getOutputs()[0].getType());
    auto result_ty = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
    if (!input_ty || !filter_ty || !output_ty || !result_ty) {
        return fail("expected ranked tensor operands/results");
    }
    if (!input_ty.hasStaticShape() || !filter_ty.hasStaticShape() || !result_ty.hasStaticShape()) {
        return fail("expected static tensor shapes");
    }
    if (input_ty.getRank() != 4 || filter_ty.getRank() != 4 || result_ty.getRank() != 4) {
        return fail("expected 4D input/filter/result");
    }
    auto elem_ty = result_ty.getElementType();
    if (!mlir::isa<mlir::FloatType>(elem_ty)) {
        return fail("expected floating-point result type");
    }

    int64_t stride_h = 0;
    int64_t stride_w = 0;
    int64_t dil_h = 0;
    int64_t dil_w = 0;
    auto read_hw = [](mlir::DenseIntElementsAttr attr, int64_t& h, int64_t& w) -> bool {
        if (!attr || attr.getNumElements() < 2) {
            return false;
        }
        auto vals = attr.getValues<int64_t>();
        auto it = vals.begin();
        for (int64_t i = 0, e = static_cast<int64_t>(attr.getNumElements()) - 2; i < e; ++i) {
            ++it;
        }
        h = *it++;
        w = *it++;
        return true;
    };
    if (!read_hw(op.getStrides(), stride_h, stride_w) || !read_hw(op.getDilations(), dil_h, dil_w)) {
        return fail("missing stride/dilation attributes");
    }

    const auto in_shape = input_ty.getShape();
    const auto w_shape = filter_ty.getShape();
    const auto out_shape = result_ty.getShape();
    const int64_t batch = in_shape[0];
    const int64_t channels = in_shape[1];
    const int64_t out_channels = out_shape[1];
    const int64_t out_h = out_shape[2];
    const int64_t out_w = out_shape[3];
    const int64_t kernel_h = w_shape[2];
    const int64_t kernel_w = w_shape[3];
    if (batch <= 0 || channels <= 0 || out_channels <= 0 || out_h <= 0 || out_w <= 0 || kernel_h <= 0 ||
        kernel_w <= 0) {
        return false;
    }

    const int64_t m_dim = out_h * out_w;
    const int64_t k_inner = kernel_h * kernel_w;
    const int64_t k_dim = channels * k_inner;

    auto* ctx = op.getContext();
    const auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto b0 = mlir::getAffineDimExpr(0, ctx);
    auto d1 = mlir::getAffineDimExpr(1, ctx);
    auto d2 = mlir::getAffineDimExpr(2, ctx);

    auto oh_expr = d1.floorDiv(out_w);
    auto ow_expr = d1 % out_w;
    auto c_expr = d2.floorDiv(k_inner);
    auto k_rem = d2 % k_inner;
    auto kh_expr = k_rem.floorDiv(kernel_w);
    auto kw_expr = k_rem % kernel_w;
    auto ih_expr = oh_expr * stride_h + kh_expr * dil_h;
    auto iw_expr = ow_expr * stride_w + kw_expr * dil_w;

    if (batch == 1) {
        auto im2col_ty = mlir::RankedTensorType::get({k_dim, m_dim}, elem_ty);
        auto weight_ty = mlir::RankedTensorType::get({out_channels, k_dim}, elem_ty);
        auto matmul_ty = mlir::RankedTensorType::get({out_channels, m_dim}, elem_ty);
        auto im2col_empty = rewriter.create<mlir::tensor::EmptyOp>(loc, im2col_ty.getShape(), elem_ty);
        auto matmul_empty = rewriter.create<mlir::tensor::EmptyOp>(loc, matmul_ty.getShape(), elem_ty);

        auto zero = make_zero(rewriter, loc, elem_ty);
        if (!zero) {
            return fail("failed to materialize zero init");
        }
        auto matmul_init =
            rewriter.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{matmul_empty.getResult()});

        auto m0 = mlir::getAffineDimExpr(0, ctx);
        auto k0 = mlir::getAffineDimExpr(1, ctx);
        auto oh_expr_2d = m0.floorDiv(out_w);
        auto ow_expr_2d = m0 % out_w;
        auto c_expr_2d = k0.floorDiv(k_inner);
        auto k_rem_2d = k0 % k_inner;
        auto kh_expr_2d = k_rem_2d.floorDiv(kernel_w);
        auto kw_expr_2d = k_rem_2d % kernel_w;
        auto ih_expr_2d = oh_expr_2d * stride_h + kh_expr_2d * dil_h;
        auto iw_expr_2d = ow_expr_2d * stride_w + kw_expr_2d * dil_w;

        auto im2col_in_map =
            mlir::AffineMap::get(2, 0, {mlir::getAffineConstantExpr(0, ctx), c_expr_2d, ih_expr_2d, iw_expr_2d}, ctx);
        auto im2col_out_map = mlir::AffineMap::get(2, 0, {k0, m0}, ctx);
        llvm::SmallVector<mlir::utils::IteratorType, 4> im2col_iters(2, mlir::utils::IteratorType::parallel);
        auto im2col = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            mlir::TypeRange{im2col_ty},
            mlir::ValueRange{op.getInputs()[0]},
            mlir::ValueRange{im2col_empty.getResult()},
            mlir::ArrayRef<mlir::AffineMap>{im2col_in_map, im2col_out_map},
            llvm::ArrayRef<mlir::utils::IteratorType>(im2col_iters));
        im2col->setAttr("gfx.im2col_stage", rewriter.getStringAttr("extract"));
        im2col->setAttr("gfx.im2col_out_w", rewriter.getI64IntegerAttr(out_w));
        im2col->setAttr("gfx.im2col_kernel_h", rewriter.getI64IntegerAttr(kernel_h));
        im2col->setAttr("gfx.im2col_kernel_w", rewriter.getI64IntegerAttr(kernel_w));
        im2col->setAttr("gfx.im2col_stride_h", rewriter.getI64IntegerAttr(stride_h));
        im2col->setAttr("gfx.im2col_stride_w", rewriter.getI64IntegerAttr(stride_w));
        im2col->setAttr("gfx.im2col_dil_h", rewriter.getI64IntegerAttr(dil_h));
        im2col->setAttr("gfx.im2col_dil_w", rewriter.getI64IntegerAttr(dil_w));
        im2col->setAttr("gfx.im2col_transposed", rewriter.getBoolAttr(true));
        {
            auto& region = im2col.getRegion();
            region.getBlocks().clear();
            auto* block = &region.emplaceBlock();
            block->addArguments({elem_ty, elem_ty}, {loc, loc});
            mlir::OpBuilder body(block, block->begin());
            body.create<mlir::linalg::YieldOp>(loc, block->getArgument(0));
        }

        llvm::SmallVector<mlir::ReassociationIndices, 4> weight_reassoc = {{0}, {1, 2, 3}};
        auto weight = rewriter.create<mlir::tensor::CollapseShapeOp>(loc, weight_ty, op.getInputs()[1], weight_reassoc);

        auto matmul = rewriter.create<mlir::linalg::MatmulOp>(
            loc,
            mlir::TypeRange{matmul_ty},
            mlir::ValueRange{weight.getResult(), im2col.getResult(0)},
            mlir::ValueRange{matmul_init.getResult(0)},
            mlir::ArrayRef<mlir::NamedAttribute>{});

        llvm::SmallVector<mlir::ReassociationIndices, 4> result_reassoc = {{0, 1}, {2, 3}};
        auto expanded = rewriter.create<mlir::tensor::ExpandShapeOp>(loc, result_ty, matmul.getResult(0), result_reassoc);
        rewriter.replaceOp(op, expanded.getResult());
        if (debug) {
            llvm::errs() << "[GFX][MLIR] im2col rewrite applied for batch=1\n";
        }
        return true;
    }

    auto im2col_ty = mlir::RankedTensorType::get({batch, m_dim, k_dim}, elem_ty);
    auto packed_weight_ty = mlir::RankedTensorType::get({batch, k_dim, out_channels}, elem_ty);
    auto matmul_ty = mlir::RankedTensorType::get({batch, m_dim, out_channels}, elem_ty);

    auto im2col_empty = rewriter.create<mlir::tensor::EmptyOp>(loc, im2col_ty.getShape(), elem_ty);
    auto packed_weight_empty = rewriter.create<mlir::tensor::EmptyOp>(loc, packed_weight_ty.getShape(), elem_ty);
    auto matmul_empty = rewriter.create<mlir::tensor::EmptyOp>(loc, matmul_ty.getShape(), elem_ty);
    auto result_empty = rewriter.create<mlir::tensor::EmptyOp>(loc, result_ty.getShape(), elem_ty);

    auto zero = make_zero(rewriter, loc, elem_ty);
    if (!zero) {
        return fail("failed to materialize zero init");
    }
    auto matmul_init =
        rewriter.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{matmul_empty.getResult()});

    auto im2col_in_map = mlir::AffineMap::get(3, 0, {b0, c_expr, ih_expr, iw_expr}, ctx);
    auto im2col_out_map = mlir::AffineMap::getMultiDimIdentityMap(3, ctx);
    llvm::SmallVector<mlir::utils::IteratorType, 4> im2col_iters(3, mlir::utils::IteratorType::parallel);
    auto im2col = rewriter.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{im2col_ty},
        mlir::ValueRange{op.getInputs()[0]},
        mlir::ValueRange{im2col_empty.getResult()},
        mlir::ArrayRef<mlir::AffineMap>{im2col_in_map, im2col_out_map},
        llvm::ArrayRef<mlir::utils::IteratorType>(im2col_iters));
    im2col->setAttr("gfx.im2col_stage", rewriter.getStringAttr("extract"));
    im2col->setAttr("gfx.im2col_out_w", rewriter.getI64IntegerAttr(out_w));
    im2col->setAttr("gfx.im2col_kernel_h", rewriter.getI64IntegerAttr(kernel_h));
    im2col->setAttr("gfx.im2col_kernel_w", rewriter.getI64IntegerAttr(kernel_w));
    im2col->setAttr("gfx.im2col_stride_h", rewriter.getI64IntegerAttr(stride_h));
    im2col->setAttr("gfx.im2col_stride_w", rewriter.getI64IntegerAttr(stride_w));
    im2col->setAttr("gfx.im2col_dil_h", rewriter.getI64IntegerAttr(dil_h));
    im2col->setAttr("gfx.im2col_dil_w", rewriter.getI64IntegerAttr(dil_w));
    {
        auto& region = im2col.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty}, {loc, loc});
        mlir::OpBuilder body(block, block->begin());
        body.create<mlir::linalg::YieldOp>(loc, block->getArgument(0));
    }

    auto weight_c_expr = d1.floorDiv(k_inner);
    auto weight_k_rem = d1 % k_inner;
    auto weight_kh_expr = weight_k_rem.floorDiv(kernel_w);
    auto weight_kw_expr = weight_k_rem % kernel_w;
    auto packed_weight_in_map =
        mlir::AffineMap::get(3, 0, {d2, weight_c_expr, weight_kh_expr, weight_kw_expr}, ctx);
    auto packed_weight_out_map = mlir::AffineMap::getMultiDimIdentityMap(3, ctx);
    llvm::SmallVector<mlir::utils::IteratorType, 4> packed_weight_iters(3, mlir::utils::IteratorType::parallel);
    auto packed_weight = rewriter.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{packed_weight_ty},
        mlir::ValueRange{op.getInputs()[1]},
        mlir::ValueRange{packed_weight_empty.getResult()},
        mlir::ArrayRef<mlir::AffineMap>{packed_weight_in_map, packed_weight_out_map},
        llvm::ArrayRef<mlir::utils::IteratorType>(packed_weight_iters));
    packed_weight->setAttr("gfx.im2col_stage", rewriter.getStringAttr("pack_weight"));
    packed_weight->setAttr("gfx.im2col_kernel_h", rewriter.getI64IntegerAttr(kernel_h));
    packed_weight->setAttr("gfx.im2col_kernel_w", rewriter.getI64IntegerAttr(kernel_w));
    {
        auto& region = packed_weight.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty}, {loc, loc});
        mlir::OpBuilder body(block, block->begin());
        body.create<mlir::linalg::YieldOp>(loc, block->getArgument(0));
    }

    auto matmul = rewriter.create<mlir::linalg::BatchMatmulOp>(
        loc,
        mlir::TypeRange{matmul_ty},
        mlir::ValueRange{im2col.getResult(0), packed_weight.getResult(0)},
        mlir::ValueRange{matmul_init.getResult(0)},
        mlir::ArrayRef<mlir::NamedAttribute>{});

    auto oc_expr = mlir::getAffineDimExpr(1, ctx);
    auto oh_out_expr = mlir::getAffineDimExpr(2, ctx);
    auto ow_out_expr = mlir::getAffineDimExpr(3, ctx);
    auto m_expr = oh_out_expr * out_w + ow_out_expr;
    auto result_in_map = mlir::AffineMap::get(4, 0, {b0, m_expr, oc_expr}, ctx);
    auto result_out_map = mlir::AffineMap::getMultiDimIdentityMap(4, ctx);
    llvm::SmallVector<mlir::utils::IteratorType, 4> result_iters(4, mlir::utils::IteratorType::parallel);
    auto result = rewriter.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{result_ty},
        mlir::ValueRange{matmul.getResult(0)},
        mlir::ValueRange{result_empty.getResult()},
        mlir::ArrayRef<mlir::AffineMap>{result_in_map, result_out_map},
        llvm::ArrayRef<mlir::utils::IteratorType>(result_iters));
    result->setAttr("gfx.im2col_stage", rewriter.getStringAttr("restore_output"));
    result->setAttr("gfx.im2col_out_w", rewriter.getI64IntegerAttr(out_w));
    {
        auto& region = result.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty}, {loc, loc});
        mlir::OpBuilder body(block, block->begin());
        body.create<mlir::linalg::YieldOp>(loc, block->getArgument(0));
    }

    rewriter.replaceOp(op, result.getResults());
    if (debug) {
        llvm::errs() << "[GFX][MLIR] im2col rewrite applied for batch>1\n";
    }
    return true;
}

}  // namespace

void run_conv_im2col_matmul_rewrite(mlir::ModuleOp module) {
    if (!module_requests_im2col(module)) {
        return;
    }
    llvm::SmallVector<mlir::linalg::Conv2DNchwFchwOp, 4> convs;
    module.walk([&](mlir::linalg::Conv2DNchwFchwOp op) {
        convs.push_back(op);
    });
    if (convs.empty()) {
        return;
    }
    mlir::IRRewriter rewriter(module.getContext());
    for (auto conv : convs) {
        (void)rewrite_conv_to_im2col_matmul(conv, rewriter);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
