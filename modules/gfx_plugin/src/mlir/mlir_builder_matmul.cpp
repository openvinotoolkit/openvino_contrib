// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
std::shared_ptr<const ov::op::v0::MatMul> find_single_matmul(const std::shared_ptr<const ov::Model>& model) {
    std::shared_ptr<const ov::op::v0::MatMul> matmul;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
            OPENVINO_ASSERT(!matmul, "Only single MatMul is supported in MLIR path for now");
            matmul = mm;
        }
    }
    OPENVINO_ASSERT(matmul, "MLIR MatMul builder: MatMul op not found");
    return matmul;
}

int64_t batch_product(const ov::Shape& shape) {
    if (shape.size() <= 2) {
        return 1;
    }
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= static_cast<int64_t>(shape[i]);
    }
    return batch;
}

mlir::Type to_elem_ty(ov::element::Type et, mlir::MLIRContext& ctx) {
    switch (et) {
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        case ov::element::i32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        case ov::element::i64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
        case ov::element::u32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned);
        case ov::element::u64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Unsigned);
        default: return mlir::Float32Type::get(&ctx);
    }
}

}  // namespace

mlir::ModuleOp build_mlir_module_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();

    auto matmul = find_single_matmul(model);
    const auto shape_a = matmul->get_input_shape(0);
    const auto shape_b = matmul->get_input_shape(1);
    OPENVINO_ASSERT(!shape_a.empty() && !shape_b.empty(), "MatMul: shapes required");
    OPENVINO_ASSERT(shape_a.size() >= 2 && shape_a.size() <= 4, "MatMul supports ranks 2–4");
    OPENVINO_ASSERT(shape_b.size() >= 2 && shape_b.size() <= 4, "MatMul supports ranks 2–4");

    const bool ta = matmul->get_transpose_a();
    const bool tb = matmul->get_transpose_b();

    const int64_t batch_a = batch_product(shape_a);
    const int64_t batch_b = batch_product(shape_b);
    const int64_t batch = std::max(batch_a, batch_b);
    OPENVINO_ASSERT(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                    "MatMul batch dims are not broadcastable");

    const int64_t a_dim1 = static_cast<int64_t>(shape_a[shape_a.size() - 2]);
    const int64_t a_dim2 = static_cast<int64_t>(shape_a[shape_a.size() - 1]);
    const int64_t b_dim1 = static_cast<int64_t>(shape_b[shape_b.size() - 2]);
    const int64_t b_dim2 = static_cast<int64_t>(shape_b[shape_b.size() - 1]);

    const int64_t M = ta ? a_dim2 : a_dim1;
    const int64_t K_a = ta ? a_dim1 : a_dim2;
    const int64_t K_b = tb ? b_dim2 : b_dim1;
    const int64_t N = tb ? b_dim1 : b_dim2;
    OPENVINO_ASSERT(K_a == K_b, "MatMul K dimension mismatch");
    const int64_t K = K_a;

    auto elem_ty = to_elem_ty(matmul->get_output_element_type(0), ctx);

    mlir::SmallVector<int64_t> dims_a{batch_a, a_dim1, a_dim2};
    mlir::SmallVector<int64_t> dims_b{batch_b, b_dim1, b_dim2};
    mlir::SmallVector<int64_t> dims_out{batch, M, N};

    auto type_a = mlir::RankedTensorType::get(dims_a, elem_ty);
    auto type_b = mlir::RankedTensorType::get(dims_b, elem_ty);
    auto type_out = mlir::RankedTensorType::get(dims_out, elem_ty);

    mlir::OpBuilder module_builder(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    module_builder.setInsertionPointToStart(module.getBody());

    auto func_type = module_builder.getFunctionType({type_a, type_b}, {type_out});
    auto func = module_builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "matmul_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    mlir::Value a3 = func.getArgument(0);
    mlir::Value b3 = func.getArgument(1);

    auto out3d_ty = type_out;
    auto empty = b.create<mlir::tensor::EmptyOp>(loc, mlir::ArrayRef<int64_t>({batch, M, N}), elem_ty);

    mlir::Value zero;
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
        zero = b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(ft, 0.0)).getResult();
    } else if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
        zero = b.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(it, 0)).getResult();
    } else {
        zero = b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(mlir::Float32Type::get(&ctx), 0.0))
                   .getResult();
    }

    auto filled = b.create<mlir::linalg::FillOp>(loc,
                                                 mlir::ValueRange{zero},
                                                 mlir::ValueRange{empty.getResult()});

    auto b_expr = mlir::getAffineDimExpr(0, &ctx);
    auto m_expr = mlir::getAffineDimExpr(1, &ctx);
    auto n_expr = mlir::getAffineDimExpr(2, &ctx);
    auto k_expr = mlir::getAffineDimExpr(3, &ctx);
    auto zero_expr = mlir::getAffineConstantExpr(0, &ctx);

    auto a_batch_expr = (batch_a == 1) ? zero_expr : b_expr;
    auto b_batch_expr = (batch_b == 1) ? zero_expr : b_expr;

    auto a_map = mlir::AffineMap::get(4,
                                      0,
                                      {a_batch_expr, ta ? k_expr : m_expr, ta ? m_expr : k_expr},
                                      &ctx);
    auto b_map = mlir::AffineMap::get(4,
                                      0,
                                      {b_batch_expr, tb ? n_expr : k_expr, tb ? k_expr : n_expr},
                                      &ctx);
    auto c_map = mlir::AffineMap::get(4, 0, {b_expr, m_expr, n_expr}, &ctx);

    llvm::SmallVector<mlir::utils::IteratorType> iterators = {
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction
    };

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        out3d_ty,
        mlir::ValueRange{a3, b3},
        mlir::ValueRange{filled.getResult(0)},
        mlir::ArrayRef<mlir::AffineMap>{a_map, b_map, c_map},
        mlir::ArrayRef<mlir::utils::IteratorType>(iterators));
    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty, elem_ty}, {loc, loc, loc});
        mlir::OpBuilder body(block, block->begin());
        auto lhs = block->getArgument(0);
        auto rhs = block->getArgument(1);
        auto acc = block->getArgument(2);
        mlir::Value mul;
        mlir::Value sum;
        if (mlir::isa<mlir::FloatType>(elem_ty)) {
            mul = body.create<mlir::arith::MulFOp>(loc, lhs, rhs);
            sum = body.create<mlir::arith::AddFOp>(loc, acc, mul);
        } else {
            mul = body.create<mlir::arith::MulIOp>(loc, lhs, rhs);
            sum = body.create<mlir::arith::AddIOp>(loc, acc, mul);
        }
        body.create<mlir::linalg::YieldOp>(loc, sum);
    }

    mlir::Value out3d = generic.getResult(0);
    b.create<mlir::func::ReturnOp>(loc, out3d);

    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
