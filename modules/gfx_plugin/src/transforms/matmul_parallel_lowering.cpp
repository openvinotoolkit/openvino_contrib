// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/matmul_parallel_lowering.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "openvino/core/except.hpp"

#include "runtime/gfx_logger.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace ov {
namespace gfx_plugin {

namespace {

mlir::Value strip_memref_casts(mlir::Value value) {
    while (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
        value = cast.getSource();
    }
    while (auto cast = value.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
        value = cast.getSource();
    }
    return value;
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

int64_t find_k_dim(const mlir::AffineMap& map) {
    for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
        if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
            if (dim.getPosition() == 3) {
                return static_cast<int64_t>(idx);
            }
        }
    }
    return -1;
}

bool is_pure_matmul_body(mlir::linalg::GenericOp op) {
    auto& body = op.getRegion().front();
    if (body.getNumArguments() != 3) {
        return false;
    }
    auto* term = body.getTerminator();
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(term);
    if (!yield || yield.getNumOperands() != 1) {
        return false;
    }
    llvm::SmallVector<mlir::Operation*, 4> ops;
    for (auto& it : body.without_terminator()) {
        ops.push_back(&it);
    }
    if (ops.size() != 2) {
        return false;
    }
    mlir::Operation* mul_op = nullptr;
    mlir::Operation* add_op = nullptr;
    for (auto* op_it : ops) {
        if (mlir::isa<mlir::arith::MulFOp, mlir::arith::MulIOp>(op_it)) {
            mul_op = op_it;
            continue;
        }
        if (mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(op_it)) {
            add_op = op_it;
            continue;
        }
        return false;
    }
    if (!mul_op || !add_op) {
        return false;
    }
    auto add_operands = add_op->getOperands();
    if (add_operands.size() != 2) {
        return false;
    }
    auto mul_result = mul_op->getResult(0);
    bool has_mul = (add_operands[0] == mul_result) || (add_operands[1] == mul_result);
    bool has_acc = (add_operands[0] == body.getArgument(2)) ||
                   (add_operands[1] == body.getArgument(2));
    if (!has_mul || !has_acc) {
        return false;
    }
    if (yield.getOperand(0) != add_op->getResult(0)) {
        return false;
    }
    return true;
}

bool is_zero_fill(mlir::linalg::FillOp fill_op) {
    if (!fill_op || fill_op.getInputs().empty()) {
        return false;
    }
    auto cst = fill_op.getInputs()[0].getDefiningOp<mlir::arith::ConstantOp>();
    if (!cst) {
        return false;
    }
    if (auto fattr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
        return fattr.getValueAsDouble() == 0.0;
    }
    if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
        return iattr.getInt() == 0;
    }
    return false;
}

bool is_passthrough_copy_body(mlir::linalg::GenericOp op) {
    auto& body = op.getRegion().front();
    if (body.getNumArguments() != 2) {
        return false;
    }
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    return yield && yield.getNumOperands() == 1 && yield.getOperand(0) == body.getArgument(0);
}

bool force_zero_init(mlir::Operation* op) {
    if (!op) {
        return false;
    }
    if (auto attr = op->getAttrOfType<mlir::BoolAttr>("gfx.zero_init_output")) {
        return attr.getValue();
    }
    return false;
}

struct TransposeProducerInfo {
    mlir::Value source;
    llvm::SmallVector<unsigned, 4> source_indices_from_output;
    mlir::linalg::GenericOp producer;
    mlir::Value temp_buffer;
};

std::optional<TransposeProducerInfo> match_transpose_copy_producer(mlir::Operation* consumer, mlir::Value buffer) {
    buffer = strip_memref_casts(buffer);
    auto buffer_type = mlir::dyn_cast<mlir::MemRefType>(buffer.getType());
    if (!consumer || !buffer || !buffer_type) {
        return std::nullopt;
    }

    for (auto* user : buffer.getUsers()) {
        auto gen = mlir::dyn_cast<mlir::linalg::GenericOp>(user);
        if (!gen || !gen->isBeforeInBlock(consumer) || gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1) {
            continue;
        }
        if (strip_memref_casts(gen.getDpsInits()[0]) != buffer) {
            continue;
        }
        if (!is_passthrough_copy_body(gen)) {
            continue;
        }

        auto source = strip_memref_casts(gen.getDpsInputs()[0]);
        auto source_type = mlir::dyn_cast<mlir::MemRefType>(source.getType());
        if (!source_type || source_type.getRank() != buffer_type.getRank()) {
            continue;
        }

        const auto maps = gen.getIndexingMapsArray();
        if (maps.size() != 2) {
            continue;
        }
        auto in_map = maps[0];
        auto out_map = maps[1];
        if (in_map.getNumResults() != static_cast<int64_t>(source_type.getRank()) ||
            out_map.getNumResults() != static_cast<int64_t>(buffer_type.getRank())) {
            continue;
        }

        llvm::SmallVector<unsigned, 4> permutation;
        permutation.reserve(static_cast<size_t>(source_type.getRank()));
        bool valid = true;
        for (int64_t i = 0; i < out_map.getNumResults(); ++i) {
            auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(out_map.getResult(i));
            if (!dim || dim.getPosition() != static_cast<unsigned>(i)) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }
        llvm::SmallDenseSet<unsigned, 4> seen;
        for (auto expr : in_map.getResults()) {
            auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
            if (!dim || !seen.insert(dim.getPosition()).second) {
                valid = false;
                break;
            }
            permutation.push_back(dim.getPosition());
        }
        if (!valid || permutation.size() != static_cast<size_t>(source_type.getRank())) {
            continue;
        }
        return TransposeProducerInfo{source, permutation, gen, buffer};
    }
    return std::nullopt;
}

llvm::SmallVector<mlir::Value, 4> permute_indices(llvm::ArrayRef<mlir::Value> logical_indices,
                                                  llvm::ArrayRef<unsigned> permutation) {
    llvm::SmallVector<mlir::Value, 4> result;
    result.reserve(permutation.size());
    for (auto pos : permutation) {
        OPENVINO_ASSERT(pos < logical_indices.size(), "MatMul transpose permutation index out of range");
        result.push_back(logical_indices[pos]);
    }
    return result;
}

void cleanup_dead_transpose_buffers(mlir::ModuleOp module) {
    if (!module) {
        return;
    }

    llvm::SmallVector<mlir::Operation*, 16> ops_to_erase;
    module.walk([&](mlir::linalg::GenericOp gen) {
        if (!is_passthrough_copy_body(gen) || !gen->use_empty()) {
            return;
        }
        ops_to_erase.push_back(gen.getOperation());
        auto out = strip_memref_casts(gen.getDpsInits()[0]);
        if (out && out.use_empty()) {
            if (auto* def = out.getDefiningOp()) {
                ops_to_erase.push_back(def);
            }
        }
    });

    for (auto* op : llvm::reverse(ops_to_erase)) {
        if (op && op->use_empty()) {
            op->erase();
        }
    }
}

bool has_inplace_elementwise_consumer(mlir::Operation* producer, mlir::Value output) {
    if (!producer || !output) {
        return false;
    }
    for (auto* user : output.getUsers()) {
        auto gen = mlir::dyn_cast<mlir::linalg::GenericOp>(user);
        if (!gen || gen == producer || gen->isBeforeInBlock(producer)) {
            continue;
        }
        const auto iters = gen.getIteratorTypesArray();
        bool all_parallel = llvm::all_of(iters, [](mlir::utils::IteratorType it) {
            return it == mlir::utils::IteratorType::parallel;
        });
        if (!all_parallel) {
            continue;
        }
        bool uses_output = false;
        for (auto in : gen.getDpsInputs()) {
            if (strip_memref_casts(in) == output) {
                uses_output = true;
                break;
            }
        }
        if (!uses_output) {
            continue;
        }
        bool inplace = false;
        for (auto out : gen.getDpsInits()) {
            if (strip_memref_casts(out) == output) {
                inplace = true;
                break;
            }
        }
        if (inplace) {
            return true;
        }
    }
    return false;
}

bool should_lower_linear_matmul(mlir::ModuleOp module) {
    if (!module) {
        return true;
    }
    if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.linear_matmul_parallel")) {
        if (attr.getValue()) {
            return true;
        }
    }
    if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel")) {
        return attr.getValue();
    }
    return true;
}

mlir::linalg::FillOp find_zero_fill(mlir::Operation* op, mlir::Value output) {
    if (!op || !output) {
        return nullptr;
    }
    for (auto* user : output.getUsers()) {
        auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(user);
        if (!fill || !fill->isBeforeInBlock(op)) {
            continue;
        }
        if (strip_memref_casts(fill.getOutputs()[0]) != output) {
            continue;
        }
        if (is_zero_fill(fill)) {
            return fill;
        }
    }
    return nullptr;
}

void build_affine_indices(mlir::PatternRewriter& rewriter,
                          mlir::Location loc,
                          const mlir::AffineMap& map,
                          mlir::ValueRange dims,
                          llvm::SmallVectorImpl<mlir::Value>& out) {
    out.clear();
    out.reserve(map.getNumResults());
    for (auto expr : map.getResults()) {
        auto one_map = mlir::AffineMap::get(map.getNumDims(),
                                            map.getNumSymbols(),
                                            expr,
                                            rewriter.getContext());
        out.push_back(rewriter.create<mlir::affine::AffineApplyOp>(loc, one_map, dims));
    }
}

struct MatMulLoweringPattern final : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
    using mlir::OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
            return mlir::failure();
        }
        if (op.getNumLoops() != 4) {
            return mlir::failure();
        }
        if (!is_pure_matmul_body(op)) {
            return mlir::failure();
        }
        const auto iters = op.getIteratorTypesArray();
        if (iters.size() != 4 ||
            iters[0] != mlir::utils::IteratorType::parallel ||
            iters[1] != mlir::utils::IteratorType::parallel ||
            iters[2] != mlir::utils::IteratorType::parallel ||
            iters[3] != mlir::utils::IteratorType::reduction) {
            return mlir::failure();
        }
        auto maps = op.getIndexingMapsArray();
        if (maps.size() != 3) {
            return mlir::failure();
        }
        const auto c_map = maps[2];
        if (c_map.getNumResults() != 3) {
            return mlir::failure();
        }
        for (int64_t i = 0; i < 3; ++i) {
            auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(c_map.getResult(i));
            if (!dim || dim.getPosition() != i) {
                return mlir::failure();
            }
        }
        auto module = op->getParentOfType<mlir::ModuleOp>();

        auto output = strip_memref_casts(op.getDpsInits()[0]);
        auto output_type = mlir::dyn_cast<mlir::MemRefType>(output.getType());
        if (!output_type || output_type.getRank() != 3) {
            return mlir::failure();
        }
        const bool force_linear_matmul =
            module && module->getAttrOfType<mlir::BoolAttr>("gfx.linear_matmul_parallel") &&
            module->getAttrOfType<mlir::BoolAttr>("gfx.linear_matmul_parallel").getValue() &&
            output_type.getDimSize(0) == 1;
        if (!force_linear_matmul) {
            if (module) {
                if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel")) {
                    if (!attr.getValue()) {
                        return mlir::failure();
                    }
                }
            }
        }
        if (has_inplace_elementwise_consumer(op.getOperation(), output)) {
            return mlir::failure();
        }
        auto zero_fill = find_zero_fill(op.getOperation(), output);
        auto input_a = strip_memref_casts(op.getDpsInputs()[0]);
        auto input_b = strip_memref_casts(op.getDpsInputs()[1]);
        auto input_a_type = mlir::dyn_cast<mlir::MemRefType>(input_a.getType());
        if (!input_a_type) {
            return mlir::failure();
        }

        const auto a_map = maps[0];
        const auto b_map = maps[1];
        const int64_t k_dim = find_k_dim(a_map);
        if (k_dim < 0) {
            return mlir::failure();
        }

        const auto loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        int64_t kThreadH = 8;
        int64_t kThreadW = 8;
        if (module) {
            if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_h")) {
                kThreadH = std::max<int64_t>(1, attr.getInt());
            }
            if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_w")) {
                kThreadW = std::max<int64_t>(1, attr.getInt());
            }
        }
        const int64_t tile_h = kThreadH;
        const int64_t tile_w = kThreadW;
        if (module) {
            auto* ctx = module.getContext();
            module->setAttr("gfx.dispatch_tile_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_h));
            module->setAttr("gfx.dispatch_tile_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_w));
            module->setAttr("gfx.dispatch_threads_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadH));
            module->setAttr("gfx.dispatch_threads_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadW));
        }

        auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto tileH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h);
        auto tileW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w);
        auto threadH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadH);
        auto threadW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadW);
        auto tileH_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h - 1);
        auto tileW_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w - 1);

        auto B = get_dim(rewriter, loc, output, 0);
        auto M = get_dim(rewriter, loc, output, 1);
        auto N = get_dim(rewriter, loc, output, 2);
        auto K = get_dim(rewriter, loc, input_a, k_dim);

        auto h_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, M, tileH_minus1);
        auto w_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, N, tileW_minus1);
        auto H_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, h_tiles_num, tileH);
        auto W_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, w_tiles_num, tileW);

        auto par = rewriter.create<mlir::scf::ParallelOp>(
            loc,
            mlir::ValueRange{c0, c0, c0, c0, c0},
            mlir::ValueRange{B, H_tiles, W_tiles, threadH, threadW},
            mlir::ValueRange{c1, c1, c1, c1, c1});

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(par.getBody()->getTerminator());

        auto ivs = par.getInductionVars();
        auto iv_b = ivs[0];
        auto iv_m_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[1], tileH);
        auto iv_n_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[2], tileW);
        auto iv_m = rewriter.create<mlir::arith::AddIOp>(loc, iv_m_base, ivs[3]);
        auto iv_n = rewriter.create<mlir::arith::AddIOp>(loc, iv_n_base, ivs[4]);

        auto m_in = rewriter.create<mlir::arith::CmpIOp>(loc,
                                                         mlir::arith::CmpIPredicate::slt,
                                                         iv_m,
                                                         M);
        auto n_in = rewriter.create<mlir::arith::CmpIOp>(loc,
                                                         mlir::arith::CmpIPredicate::slt,
                                                         iv_n,
                                                         N);
        auto in_bounds = rewriter.create<mlir::arith::AndIOp>(loc, m_in, n_in);
        auto if_op = rewriter.create<mlir::scf::IfOp>(loc, in_bounds, /*withElseRegion=*/false);

        rewriter.setInsertionPointToStart(if_op.thenBlock());
        auto elem_ty = output_type.getElementType();
        if (!mlir::isa<mlir::FloatType, mlir::IntegerType>(elem_ty)) {
            return mlir::failure();
        }
        mlir::Value init_val;
        if (zero_fill) {
            if (auto ft = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
                init_val =
                    rewriter.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(ft, 0.0));
            } else if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
                init_val =
                    rewriter.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(it, 0));
            }
        } else {
            init_val = rewriter.create<mlir::memref::LoadOp>(
                loc, output, mlir::ValueRange{iv_b, iv_m, iv_n});
        }
        auto k_for = rewriter.create<mlir::scf::ForOp>(loc, c0, K, c1, mlir::ValueRange{init_val});
        {
            mlir::OpBuilder::InsertionGuard loop_guard(rewriter);
            rewriter.setInsertionPointToStart(k_for.getBody());
            auto iv_k = k_for.getInductionVar();
            auto acc = k_for.getRegionIterArgs()[0];

            llvm::SmallVector<mlir::Value, 4> dims = {iv_b, iv_m, iv_n, iv_k};
            llvm::SmallVector<mlir::Value, 4> a_indices;
            llvm::SmallVector<mlir::Value, 4> b_indices;
            build_affine_indices(rewriter, loc, a_map, dims, a_indices);
            build_affine_indices(rewriter, loc, b_map, dims, b_indices);

            auto lhs = rewriter.create<mlir::memref::LoadOp>(loc, input_a, a_indices);
            auto rhs = rewriter.create<mlir::memref::LoadOp>(loc, input_b, b_indices);

            mlir::Value mul;
            mlir::Value sum;
            if (mlir::isa<mlir::FloatType>(elem_ty)) {
                mul = rewriter.create<mlir::arith::MulFOp>(loc, lhs, rhs);
                sum = rewriter.create<mlir::arith::AddFOp>(loc, acc, mul);
            } else {
                mul = rewriter.create<mlir::arith::MulIOp>(loc, lhs, rhs);
                sum = rewriter.create<mlir::arith::AddIOp>(loc, acc, mul);
            }
            rewriter.create<mlir::scf::YieldOp>(loc, sum);
        }

        auto acc_val = k_for.getResult(0);
        rewriter.create<mlir::memref::StoreOp>(loc, acc_val, output, mlir::ValueRange{iv_b, iv_m, iv_n});

        if (op->getNumResults() > 0) {
            op.getResult(0).replaceAllUsesWith(op.getDpsInits()[0]);
        }
        if (zero_fill) {
            rewriter.eraseOp(zero_fill);
        }
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct MatmulOpLoweringPattern final : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
    using mlir::OpRewritePattern<mlir::linalg::MatmulOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::linalg::MatmulOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
            return mlir::failure();
        }
        auto module = op->getParentOfType<mlir::ModuleOp>();
        if (!should_lower_linear_matmul(module)) {
            return mlir::failure();
        }

        auto output = strip_memref_casts(op.getDpsInits()[0]);
        auto output_type = mlir::dyn_cast<mlir::MemRefType>(output.getType());
        if (!output_type || output_type.getRank() != 2) {
            return mlir::failure();
        }
        auto input_a = strip_memref_casts(op.getDpsInputs()[0]);
        auto input_b = strip_memref_casts(op.getDpsInputs()[1]);
        auto input_a_type = mlir::dyn_cast<mlir::MemRefType>(input_a.getType());
        auto input_b_type = mlir::dyn_cast<mlir::MemRefType>(input_b.getType());
        if (!input_a_type || !input_b_type || input_a_type.getRank() != 2 || input_b_type.getRank() != 2) {
            return mlir::failure();
        }
        if (has_inplace_elementwise_consumer(op.getOperation(), output)) {
            return mlir::failure();
        }
        auto zero_fill = find_zero_fill(op.getOperation(), output);
        const auto loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        int64_t kThreadH = 8;
        int64_t kThreadW = 8;
        if (module) {
            if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_h")) {
                kThreadH = std::max<int64_t>(1, attr.getInt());
            }
            if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_w")) {
                kThreadW = std::max<int64_t>(1, attr.getInt());
            }
        }
        const int64_t tile_h = kThreadH;
        const int64_t tile_w = kThreadW;
        if (module) {
            auto* ctx = module.getContext();
            module->setAttr("gfx.dispatch_tile_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_h));
            module->setAttr("gfx.dispatch_tile_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_w));
            module->setAttr("gfx.dispatch_threads_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadH));
            module->setAttr("gfx.dispatch_threads_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadW));
        }

        auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto tileH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h);
        auto tileW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w);
        auto threadH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadH);
        auto threadW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadW);
        auto tileH_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h - 1);
        auto tileW_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w - 1);

        auto M = get_dim(rewriter, loc, output, 0);
        auto N = get_dim(rewriter, loc, output, 1);
        auto K = get_dim(rewriter, loc, input_a, 1);

        auto h_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, M, tileH_minus1);
        auto w_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, N, tileW_minus1);
        auto H_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, h_tiles_num, tileH);
        auto W_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, w_tiles_num, tileW);

        auto par = rewriter.create<mlir::scf::ParallelOp>(
            loc,
            mlir::ValueRange{c0, c0, c0, c0},
            mlir::ValueRange{H_tiles, W_tiles, threadH, threadW},
            mlir::ValueRange{c1, c1, c1, c1});

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(par.getBody()->getTerminator());
        auto ivs = par.getInductionVars();
        auto iv_m_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[0], tileH);
        auto iv_n_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[1], tileW);
        auto iv_m = rewriter.create<mlir::arith::AddIOp>(loc, iv_m_base, ivs[2]);
        auto iv_n = rewriter.create<mlir::arith::AddIOp>(loc, iv_n_base, ivs[3]);

        auto m_in = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iv_m, M);
        auto n_in = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iv_n, N);
        auto in_bounds = rewriter.create<mlir::arith::AndIOp>(loc, m_in, n_in);
        auto if_op = rewriter.create<mlir::scf::IfOp>(loc, in_bounds, false);

        rewriter.setInsertionPointToStart(if_op.thenBlock());
        auto elem_ty = output_type.getElementType();
        if (!mlir::isa<mlir::FloatType, mlir::IntegerType>(elem_ty)) {
            return mlir::failure();
        }
        mlir::Value init_val;
        if (zero_fill) {
            if (auto ft = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
                init_val = rewriter.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(ft, 0.0));
            } else if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
                init_val = rewriter.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(it, 0));
            }
        } else {
            init_val = rewriter.create<mlir::memref::LoadOp>(loc, output, mlir::ValueRange{iv_m, iv_n});
        }

        auto k_for = rewriter.create<mlir::scf::ForOp>(loc, c0, K, c1, mlir::ValueRange{init_val});
        {
            mlir::OpBuilder::InsertionGuard loop_guard(rewriter);
            rewriter.setInsertionPointToStart(k_for.getBody());
            auto iv_k = k_for.getInductionVar();
            auto acc = k_for.getRegionIterArgs()[0];
            auto lhs = rewriter.create<mlir::memref::LoadOp>(loc, input_a, mlir::ValueRange{iv_m, iv_k});
            auto rhs = rewriter.create<mlir::memref::LoadOp>(loc, input_b, mlir::ValueRange{iv_k, iv_n});
            mlir::Value mul;
            mlir::Value sum;
            if (mlir::isa<mlir::FloatType>(elem_ty)) {
                mul = rewriter.create<mlir::arith::MulFOp>(loc, lhs, rhs);
                sum = rewriter.create<mlir::arith::AddFOp>(loc, acc, mul);
            } else {
                mul = rewriter.create<mlir::arith::MulIOp>(loc, lhs, rhs);
                sum = rewriter.create<mlir::arith::AddIOp>(loc, acc, mul);
            }
            rewriter.create<mlir::scf::YieldOp>(loc, sum);
        }

        auto acc_val = k_for.getResult(0);
        rewriter.create<mlir::memref::StoreOp>(loc, acc_val, output, mlir::ValueRange{iv_m, iv_n});

        if (op->getNumResults() > 0) {
            op.getResult(0).replaceAllUsesWith(op.getDpsInits()[0]);
        }
        if (zero_fill) {
            rewriter.eraseOp(zero_fill);
        }
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct BatchMatMulLoweringPattern final : public mlir::OpRewritePattern<mlir::linalg::BatchMatmulOp> {
    using mlir::OpRewritePattern<mlir::linalg::BatchMatmulOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::linalg::BatchMatmulOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
            return mlir::failure();
        }
        auto module = op->getParentOfType<mlir::ModuleOp>();
        if (module) {
            if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel")) {
                if (!attr.getValue()) {
                    return mlir::failure();
                }
            }
        }

        auto output = strip_memref_casts(op.getDpsInits()[0]);
        auto output_type = mlir::dyn_cast<mlir::MemRefType>(output.getType());
        if (!output_type || output_type.getRank() != 3) {
            return mlir::failure();
        }
        auto maps = op.getIndexingMapsArray();
        if (maps.size() != 3) {
            return mlir::failure();
        }
        const auto a_map = maps[0];
        const auto b_map = maps[1];
        const int64_t k_dim = find_k_dim(a_map);
        if (k_dim < 0) {
            return mlir::failure();
        }

        auto logical_input_a = strip_memref_casts(op.getDpsInputs()[0]);
        auto logical_input_b = strip_memref_casts(op.getDpsInputs()[1]);
        auto logical_input_a_type = mlir::dyn_cast<mlir::MemRefType>(logical_input_a.getType());
        auto logical_input_b_type = mlir::dyn_cast<mlir::MemRefType>(logical_input_b.getType());
        if (!logical_input_a_type || !logical_input_b_type ||
            logical_input_a_type.getRank() != 3 || logical_input_b_type.getRank() != 3) {
            return mlir::failure();
        }

        auto input_a = logical_input_a;
        auto input_b = logical_input_b;
        auto input_a_transpose = match_transpose_copy_producer(op.getOperation(), input_a);
        auto input_b_transpose = match_transpose_copy_producer(op.getOperation(), input_b);
        if (input_a_transpose) {
            input_a = input_a_transpose->source;
        }
        if (input_b_transpose) {
            input_b = input_b_transpose->source;
        }
        auto input_a_type = mlir::dyn_cast<mlir::MemRefType>(input_a.getType());
        auto input_b_type = mlir::dyn_cast<mlir::MemRefType>(input_b.getType());
        if (!input_a_type || !input_b_type || input_a_type.getRank() != 3 || input_b_type.getRank() != 3) {
            return mlir::failure();
        }
        if (has_inplace_elementwise_consumer(op.getOperation(), output)) {
            return mlir::failure();
        }

        auto zero_fill = find_zero_fill(op.getOperation(), output);
        const bool zero_init = static_cast<bool>(zero_fill) || force_zero_init(op.getOperation());
        const auto loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        int64_t kThreadH = 8;
        int64_t kThreadW = 8;
        if (module) {
            if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_h")) {
                kThreadH = std::max<int64_t>(1, attr.getInt());
            }
            if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_w")) {
                kThreadW = std::max<int64_t>(1, attr.getInt());
            }
        }
        const int64_t tile_h = kThreadH;
        const int64_t tile_w = kThreadW;
        if (module) {
            auto* ctx = module.getContext();
            module->setAttr("gfx.dispatch_tile_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_h));
            module->setAttr("gfx.dispatch_tile_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_w));
            module->setAttr("gfx.dispatch_threads_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadH));
            module->setAttr("gfx.dispatch_threads_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadW));
        }

        auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto tileH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h);
        auto tileW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w);
        auto threadH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadH);
        auto threadW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadW);
        auto tileH_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h - 1);
        auto tileW_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w - 1);

        auto B = get_dim(rewriter, loc, output, 0);
        auto M = get_dim(rewriter, loc, output, 1);
        auto N = get_dim(rewriter, loc, output, 2);
        auto K = get_dim(rewriter, loc, logical_input_a, k_dim);

        auto h_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, M, tileH_minus1);
        auto w_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, N, tileW_minus1);
        auto H_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, h_tiles_num, tileH);
        auto W_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, w_tiles_num, tileW);

        auto par = rewriter.create<mlir::scf::ParallelOp>(
            loc,
            mlir::ValueRange{c0, c0, c0, c0, c0},
            mlir::ValueRange{B, H_tiles, W_tiles, threadH, threadW},
            mlir::ValueRange{c1, c1, c1, c1, c1});

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(par.getBody()->getTerminator());

        auto ivs = par.getInductionVars();
        auto iv_b = ivs[0];
        auto iv_m_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[1], tileH);
        auto iv_n_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[2], tileW);
        auto iv_m = rewriter.create<mlir::arith::AddIOp>(loc, iv_m_base, ivs[3]);
        auto iv_n = rewriter.create<mlir::arith::AddIOp>(loc, iv_n_base, ivs[4]);

        auto m_in = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iv_m, M);
        auto n_in = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iv_n, N);
        auto in_bounds = rewriter.create<mlir::arith::AndIOp>(loc, m_in, n_in);
        auto if_op = rewriter.create<mlir::scf::IfOp>(loc, in_bounds, /*withElseRegion=*/false);

        rewriter.setInsertionPointToStart(if_op.thenBlock());
        auto elem_ty = output_type.getElementType();
        if (!mlir::isa<mlir::FloatType, mlir::IntegerType>(elem_ty)) {
            return mlir::failure();
        }
        mlir::Value init_val;
        if (zero_init) {
            if (auto ft = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
                init_val = rewriter.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(ft, 0.0));
            } else if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
                init_val = rewriter.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(it, 0));
            }
        } else {
            init_val = rewriter.create<mlir::memref::LoadOp>(
                loc, output, mlir::ValueRange{iv_b, iv_m, iv_n});
        }

        auto k_for = rewriter.create<mlir::scf::ForOp>(loc, c0, K, c1, mlir::ValueRange{init_val});
        {
            mlir::OpBuilder::InsertionGuard loop_guard(rewriter);
            rewriter.setInsertionPointToStart(k_for.getBody());
            auto iv_k = k_for.getInductionVar();
            auto acc = k_for.getRegionIterArgs()[0];
            const llvm::SmallVector<mlir::Value, 4> dims = {iv_b, iv_m, iv_n, iv_k};
            llvm::SmallVector<mlir::Value, 4> lhs_logical_indices;
            llvm::SmallVector<mlir::Value, 4> rhs_logical_indices;
            build_affine_indices(rewriter, loc, a_map, dims, lhs_logical_indices);
            build_affine_indices(rewriter, loc, b_map, dims, rhs_logical_indices);
            auto lhs = rewriter.create<mlir::memref::LoadOp>(
                loc,
                input_a,
                input_a_transpose ? permute_indices(lhs_logical_indices, input_a_transpose->source_indices_from_output)
                                  : llvm::SmallVector<mlir::Value, 4>(lhs_logical_indices.begin(),
                                                                       lhs_logical_indices.end()));
            auto rhs = rewriter.create<mlir::memref::LoadOp>(
                loc,
                input_b,
                input_b_transpose ? permute_indices(rhs_logical_indices, input_b_transpose->source_indices_from_output)
                                  : llvm::SmallVector<mlir::Value, 4>(rhs_logical_indices.begin(),
                                                                       rhs_logical_indices.end()));

            mlir::Value mul;
            mlir::Value sum;
            if (mlir::isa<mlir::FloatType>(elem_ty)) {
                mul = rewriter.create<mlir::arith::MulFOp>(loc, lhs, rhs);
                sum = rewriter.create<mlir::arith::AddFOp>(loc, acc, mul);
            } else {
                mul = rewriter.create<mlir::arith::MulIOp>(loc, lhs, rhs);
                sum = rewriter.create<mlir::arith::AddIOp>(loc, acc, mul);
            }
            rewriter.create<mlir::scf::YieldOp>(loc, sum);
        }

        auto acc_val = k_for.getResult(0);
        rewriter.create<mlir::memref::StoreOp>(loc, acc_val, output, mlir::ValueRange{iv_b, iv_m, iv_n});

        if (op->getNumResults() > 0) {
            op.getResult(0).replaceAllUsesWith(op.getDpsInits()[0]);
        }
        if (zero_fill) {
            rewriter.eraseOp(zero_fill);
        }
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct BatchMatMulTransposeBLoweringPattern final : public mlir::OpRewritePattern<mlir::linalg::BatchMatmulTransposeBOp> {
    using mlir::OpRewritePattern<mlir::linalg::BatchMatmulTransposeBOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::linalg::BatchMatmulTransposeBOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
            return mlir::failure();
        }
        auto module = op->getParentOfType<mlir::ModuleOp>();
        if (module) {
            if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel")) {
                if (!attr.getValue()) {
                    return mlir::failure();
                }
            }
        }

        auto output = strip_memref_casts(op.getDpsInits()[0]);
        auto output_type = mlir::dyn_cast<mlir::MemRefType>(output.getType());
        if (!output_type || output_type.getRank() != 3) {
            return mlir::failure();
        }

        auto logical_input_a = strip_memref_casts(op.getDpsInputs()[0]);
        auto logical_input_b = strip_memref_casts(op.getDpsInputs()[1]);
        auto logical_input_a_type = mlir::dyn_cast<mlir::MemRefType>(logical_input_a.getType());
        auto logical_input_b_type = mlir::dyn_cast<mlir::MemRefType>(logical_input_b.getType());
        if (!logical_input_a_type || !logical_input_b_type ||
            logical_input_a_type.getRank() != 3 || logical_input_b_type.getRank() != 3) {
            return mlir::failure();
        }

        auto input_a = logical_input_a;
        auto input_b = logical_input_b;
        auto input_a_transpose = match_transpose_copy_producer(op.getOperation(), input_a);
        auto input_b_transpose = match_transpose_copy_producer(op.getOperation(), input_b);
        if (input_a_transpose) {
            input_a = input_a_transpose->source;
        }
        if (input_b_transpose) {
            input_b = input_b_transpose->source;
        }
        auto input_a_type = mlir::dyn_cast<mlir::MemRefType>(input_a.getType());
        auto input_b_type = mlir::dyn_cast<mlir::MemRefType>(input_b.getType());
        if (!input_a_type || !input_b_type || input_a_type.getRank() != 3 || input_b_type.getRank() != 3) {
            return mlir::failure();
        }
        if (has_inplace_elementwise_consumer(op.getOperation(), output)) {
            return mlir::failure();
        }

        auto zero_fill = find_zero_fill(op.getOperation(), output);
        const bool zero_init = static_cast<bool>(zero_fill) || force_zero_init(op.getOperation());
        const auto loc = op.getLoc();
        rewriter.setInsertionPoint(op);

        int64_t kThreadH = 8;
        int64_t kThreadW = 8;
        if (module) {
            if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_h")) {
                kThreadH = std::max<int64_t>(1, attr.getInt());
            }
            if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_w")) {
                kThreadW = std::max<int64_t>(1, attr.getInt());
            }
        }
        const int64_t tile_h = kThreadH;
        const int64_t tile_w = kThreadW;
        if (module) {
            auto* ctx = module.getContext();
            module->setAttr("gfx.dispatch_tile_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_h));
            module->setAttr("gfx.dispatch_tile_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), tile_w));
            module->setAttr("gfx.dispatch_threads_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadH));
            module->setAttr("gfx.dispatch_threads_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), kThreadW));
        }

        auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto tileH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h);
        auto tileW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w);
        auto threadH = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadH);
        auto threadW = rewriter.create<mlir::arith::ConstantIndexOp>(loc, kThreadW);
        auto tileH_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_h - 1);
        auto tileW_minus1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, tile_w - 1);

        auto B = get_dim(rewriter, loc, output, 0);
        auto M = get_dim(rewriter, loc, output, 1);
        auto N = get_dim(rewriter, loc, output, 2);
        auto K = get_dim(rewriter, loc, logical_input_a, 2);

        auto h_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, M, tileH_minus1);
        auto w_tiles_num = rewriter.create<mlir::arith::AddIOp>(loc, N, tileW_minus1);
        auto H_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, h_tiles_num, tileH);
        auto W_tiles = rewriter.create<mlir::arith::DivSIOp>(loc, w_tiles_num, tileW);

        auto par = rewriter.create<mlir::scf::ParallelOp>(
            loc,
            mlir::ValueRange{c0, c0, c0, c0, c0},
            mlir::ValueRange{B, H_tiles, W_tiles, threadH, threadW},
            mlir::ValueRange{c1, c1, c1, c1, c1});

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(par.getBody()->getTerminator());

        auto ivs = par.getInductionVars();
        auto iv_b = ivs[0];
        auto iv_m_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[1], tileH);
        auto iv_n_base = rewriter.create<mlir::arith::MulIOp>(loc, ivs[2], tileW);
        auto iv_m = rewriter.create<mlir::arith::AddIOp>(loc, iv_m_base, ivs[3]);
        auto iv_n = rewriter.create<mlir::arith::AddIOp>(loc, iv_n_base, ivs[4]);

        auto m_in = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iv_m, M);
        auto n_in = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iv_n, N);
        auto in_bounds = rewriter.create<mlir::arith::AndIOp>(loc, m_in, n_in);
        auto if_op = rewriter.create<mlir::scf::IfOp>(loc, in_bounds, /*withElseRegion=*/false);

        rewriter.setInsertionPointToStart(if_op.thenBlock());
        auto elem_ty = output_type.getElementType();
        if (!mlir::isa<mlir::FloatType, mlir::IntegerType>(elem_ty)) {
            return mlir::failure();
        }
        mlir::Value init_val;
        if (zero_init) {
            if (auto ft = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
                init_val = rewriter.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(ft, 0.0));
            } else if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
                init_val = rewriter.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(it, 0));
            }
        } else {
            init_val = rewriter.create<mlir::memref::LoadOp>(
                loc, output, mlir::ValueRange{iv_b, iv_m, iv_n});
        }

        auto k_for = rewriter.create<mlir::scf::ForOp>(loc, c0, K, c1, mlir::ValueRange{init_val});
        {
            mlir::OpBuilder::InsertionGuard loop_guard(rewriter);
            rewriter.setInsertionPointToStart(k_for.getBody());
            auto iv_k = k_for.getInductionVar();
            auto acc = k_for.getRegionIterArgs()[0];
            const llvm::SmallVector<mlir::Value, 4> lhs_logical_indices = {iv_b, iv_m, iv_k};
            const llvm::SmallVector<mlir::Value, 4> rhs_logical_indices = {iv_b, iv_n, iv_k};
            auto lhs = rewriter.create<mlir::memref::LoadOp>(
                loc,
                input_a,
                input_a_transpose ? permute_indices(lhs_logical_indices, input_a_transpose->source_indices_from_output)
                                  : llvm::SmallVector<mlir::Value, 4>(lhs_logical_indices.begin(),
                                                                       lhs_logical_indices.end()));
            auto rhs = rewriter.create<mlir::memref::LoadOp>(
                loc,
                input_b,
                input_b_transpose ? permute_indices(rhs_logical_indices, input_b_transpose->source_indices_from_output)
                                  : llvm::SmallVector<mlir::Value, 4>(rhs_logical_indices.begin(),
                                                                       rhs_logical_indices.end()));

            mlir::Value mul;
            mlir::Value sum;
            if (mlir::isa<mlir::FloatType>(elem_ty)) {
                mul = rewriter.create<mlir::arith::MulFOp>(loc, lhs, rhs);
                sum = rewriter.create<mlir::arith::AddFOp>(loc, acc, mul);
            } else {
                mul = rewriter.create<mlir::arith::MulIOp>(loc, lhs, rhs);
                sum = rewriter.create<mlir::arith::AddIOp>(loc, acc, mul);
            }
            rewriter.create<mlir::scf::YieldOp>(loc, sum);
        }

        auto acc_val = k_for.getResult(0);
        rewriter.create<mlir::memref::StoreOp>(loc, acc_val, output, mlir::ValueRange{iv_b, iv_m, iv_n});

        if (op->getNumResults() > 0) {
            op.getResult(0).replaceAllUsesWith(op.getDpsInits()[0]);
        }
        if (zero_fill) {
            rewriter.eraseOp(zero_fill);
        }
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

}  // namespace

void run_matmul_parallel_lowering(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    if (auto skip = module->getAttrOfType<mlir::BoolAttr>("gfx.skip_matmul_parallel")) {
        if (skip.getValue()) {
            return;
        }
    }
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIR") << "MatMul parallel lowering pass";
    }
    auto* ctx = module.getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<MatMulLoweringPattern>(ctx);
    patterns.add<MatmulOpLoweringPattern>(ctx);
    patterns.add<BatchMatMulLoweringPattern>(ctx);
    patterns.add<BatchMatMulTransposeBLoweringPattern>(ctx);
    (void)mlir::applyPatternsGreedily(module, std::move(patterns));
    cleanup_dead_transpose_buffers(module);
}

}  // namespace gfx_plugin
}  // namespace ov
