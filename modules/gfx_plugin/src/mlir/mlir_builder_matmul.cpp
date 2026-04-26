// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

int64_t batch_product(const ov::PartialShape& shape) {
    if (shape.rank().get_length() <= 2) {
        return 1;
    }
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < static_cast<size_t>(shape.rank().get_length()); ++i) {
        if (shape[i].is_dynamic()) {
            return mlir::ShapedType::kDynamic;
        }
        batch *= static_cast<int64_t>(shape[i].get_length());
    }
    return batch;
}

mlir::Type to_elem_ty(ov::element::Type et, mlir::MLIRContext& ctx) {
    switch (et) {
        case ov::element::f16:
            return mlir::Float16Type::get(&ctx);
        case ov::element::f32:
            return mlir::Float32Type::get(&ctx);
        case ov::element::i32:
            return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        case ov::element::i64:
            return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
        case ov::element::u32:
            return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned);
        case ov::element::u64:
            return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Unsigned);
        default:
            return mlir::Float32Type::get(&ctx);
    }
}

mlir::Value make_zero_scalar(mlir::OpBuilder& builder,
                             mlir::Location loc,
                             mlir::MLIRContext& ctx,
                             mlir::Type elem_ty) {
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
        return builder.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(ft, 0.0)).getResult();
    }
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
        return builder.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(it, 0)).getResult();
    }
    return builder.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(mlir::Float32Type::get(&ctx), 0.0))
        .getResult();
}

mlir::Value make_index_const(mlir::OpBuilder& builder, mlir::Location loc, int64_t value) {
    return builder.create<mlir::arith::ConstantIndexOp>(loc, value);
}

mlir::Value build_tensor_permutation(mlir::OpBuilder& builder,
                                     mlir::Location loc,
                                     mlir::Value input,
                                     llvm::ArrayRef<int64_t> output_dims,
                                     llvm::ArrayRef<unsigned> permutation,
                                     mlir::Type elem_ty,
                                     mlir::MLIRContext& ctx) {
    OPENVINO_ASSERT(permutation.size() == output_dims.size(), "MatMul permutation rank mismatch");
    auto output_ty = mlir::RankedTensorType::get(output_dims, elem_ty);
    auto empty = builder.create<mlir::tensor::EmptyOp>(loc, output_dims, elem_ty).getResult();

    llvm::SmallVector<mlir::AffineExpr> in_exprs;
    llvm::SmallVector<mlir::AffineExpr> out_exprs;
    in_exprs.reserve(permutation.size());
    out_exprs.reserve(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        in_exprs.push_back(mlir::getAffineDimExpr(static_cast<unsigned>(permutation[i]), &ctx));
        out_exprs.push_back(mlir::getAffineDimExpr(static_cast<unsigned>(i), &ctx));
    }

    llvm::SmallVector<mlir::utils::IteratorType> iterators(
        permutation.size(), mlir::utils::IteratorType::parallel);
    auto generic = builder.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{output_ty},
        mlir::ValueRange{input},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{
            mlir::AffineMap::get(static_cast<unsigned>(permutation.size()), 0, in_exprs, &ctx),
            mlir::AffineMap::get(static_cast<unsigned>(permutation.size()), 0, out_exprs, &ctx),
        },
        iterators,
        [&](mlir::OpBuilder& nested_builder, mlir::Location nested_loc, mlir::ValueRange args) {
            OPENVINO_ASSERT(args.size() == 2, "MatMul permutation body expects input and output");
            nested_builder.create<mlir::linalg::YieldOp>(nested_loc, args[0]);
        });
    return generic.getResult(0);
}

}  // namespace

mlir::ModuleOp build_mlir_module_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect,
                    mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::scf::SCFDialect>();

    auto matmul = find_single_matmul(model);
    const auto shape_a = matmul->get_input_partial_shape(0);
    const auto shape_b = matmul->get_input_partial_shape(1);
    OPENVINO_ASSERT(shape_a.rank().is_static() && shape_b.rank().is_static(),
                    "MatMul: ranks must be static");
    OPENVINO_ASSERT(shape_a.rank().get_length() >= 2 && shape_a.rank().get_length() <= 4, "MatMul supports ranks 2-4");
    OPENVINO_ASSERT(shape_b.rank().get_length() >= 2 && shape_b.rank().get_length() <= 4, "MatMul supports ranks 2-4");

    const bool ta = matmul->get_transpose_a();
    const bool tb = matmul->get_transpose_b();

    const int64_t batch_a = batch_product(shape_a);
    const int64_t batch_b = batch_product(shape_b);
    const int64_t batch = std::max(batch_a, batch_b);
    OPENVINO_ASSERT(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                    "MatMul batch dims are not broadcastable");

    const auto rank_a = static_cast<size_t>(shape_a.rank().get_length());
    const auto rank_b = static_cast<size_t>(shape_b.rank().get_length());
    auto dim_or_dynamic = [](const ov::PartialShape& shape, size_t idx) -> int64_t {
        return shape[idx].is_dynamic() ? mlir::ShapedType::kDynamic
                                       : static_cast<int64_t>(shape[idx].get_length());
    };

    const int64_t a_dim1 = dim_or_dynamic(shape_a, rank_a - 2);
    const int64_t a_dim2 = dim_or_dynamic(shape_a, rank_a - 1);
    const int64_t b_dim1 = dim_or_dynamic(shape_b, rank_b - 2);
    const int64_t b_dim2 = dim_or_dynamic(shape_b, rank_b - 1);

    const int64_t M = ta ? a_dim2 : a_dim1;
    const int64_t K_a = ta ? a_dim1 : a_dim2;
    const int64_t K_b = tb ? b_dim2 : b_dim1;
    const int64_t N = tb ? b_dim1 : b_dim2;
    OPENVINO_ASSERT(K_a == mlir::ShapedType::kDynamic || K_b == mlir::ShapedType::kDynamic || K_a == K_b,
                    "MatMul K dimension mismatch");
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

    llvm::SmallVector<mlir::Value> out_dyn_dims;
    out_dyn_dims.reserve(dims_out.size());
    for (size_t i = 0; i < dims_out.size(); ++i) {
        if (dims_out[i] != mlir::ShapedType::kDynamic) {
            continue;
        }
        if (i < dims_a.size() && dims_a[i] == mlir::ShapedType::kDynamic) {
            out_dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(0), static_cast<int64_t>(i)).getResult());
            continue;
        }
        if (i < dims_b.size() && dims_b[i] == mlir::ShapedType::kDynamic) {
            out_dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(1), static_cast<int64_t>(i)).getResult());
            continue;
        }
        if (i + 1 == dims_out.size() && b_dim2 == mlir::ShapedType::kDynamic && !tb) {
            out_dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(1), static_cast<int64_t>(rank_b - 1)).getResult());
            continue;
        }
        if (i + 1 == dims_out.size() && b_dim1 == mlir::ShapedType::kDynamic && tb) {
            out_dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(1), static_cast<int64_t>(rank_b - 2)).getResult());
            continue;
        }
        if (i + 2 == dims_out.size() && a_dim2 == mlir::ShapedType::kDynamic && ta) {
            out_dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(0), static_cast<int64_t>(rank_a - 1)).getResult());
            continue;
        }
        if (i + 2 == dims_out.size() && a_dim1 == mlir::ShapedType::kDynamic && !ta) {
            out_dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(0), static_cast<int64_t>(rank_a - 2)).getResult());
            continue;
        }
    }
    auto empty = b.create<mlir::tensor::EmptyOp>(loc, dims_out, elem_ty, out_dyn_dims).getResult();
    auto zero = make_zero_scalar(b, loc, ctx, elem_ty);
    auto init = b.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});

    if (batch_a == batch_b) {
        mlir::Value lhs = func.getArgument(0);
        mlir::Value rhs = func.getArgument(1);

        if (ta) {
            lhs = build_tensor_permutation(b,
                                           loc,
                                           lhs,
                                           llvm::ArrayRef<int64_t>({batch, M, K}),
                                           llvm::ArrayRef<unsigned>({0u, 2u, 1u}),
                                           elem_ty,
                                           ctx);
        }

        if (tb) {
            auto batch_matmul = b.create<mlir::linalg::BatchMatmulTransposeBOp>(
                loc,
                mlir::TypeRange{type_out},
                mlir::ValueRange{lhs, rhs},
                mlir::ValueRange{init.getResult(0)},
                mlir::ArrayRef<mlir::NamedAttribute>{});
            b.create<mlir::func::ReturnOp>(loc, batch_matmul.getResult(0));
        } else {
            auto batch_matmul = b.create<mlir::linalg::BatchMatmulOp>(
                loc,
                mlir::TypeRange{type_out},
                mlir::ValueRange{lhs, rhs},
                mlir::ValueRange{init.getResult(0)},
                mlir::ArrayRef<mlir::NamedAttribute>{});
            b.create<mlir::func::ReturnOp>(loc, batch_matmul.getResult(0));
        }
        return module;
    }

    auto d0 = mlir::getAffineDimExpr(0, &ctx);
    auto d1 = mlir::getAffineDimExpr(1, &ctx);
    auto d2 = mlir::getAffineDimExpr(2, &ctx);
    auto d3 = mlir::getAffineDimExpr(3, &ctx);
    auto const0 = mlir::getAffineConstantExpr(0, &ctx);

    mlir::SmallVector<mlir::AffineMap> indexing_maps;
    indexing_maps.reserve(3);
    indexing_maps.push_back(mlir::AffineMap::get(
        /*dimCount=*/4,
        /*symbolCount=*/0,
        {batch_a == 1 ? const0 : d0, ta ? d3 : d1, ta ? d1 : d3},
        &ctx));
    indexing_maps.push_back(mlir::AffineMap::get(
        /*dimCount=*/4,
        /*symbolCount=*/0,
        {batch_b == 1 ? const0 : d0, tb ? d2 : d3, tb ? d3 : d2},
        &ctx));
    indexing_maps.push_back(mlir::AffineMap::get(
        /*dimCount=*/4,
        /*symbolCount=*/0,
        {d0, d1, d2},
        &ctx));

    mlir::SmallVector<mlir::utils::IteratorType> iterators = {
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction,
    };

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{type_out},
        mlir::ValueRange{func.getArgument(0), func.getArgument(1)},
        mlir::ValueRange{init.getResult(0)},
        indexing_maps,
        iterators,
        [&](mlir::OpBuilder& nested_builder, mlir::Location nested_loc, mlir::ValueRange args) {
            OPENVINO_ASSERT(args.size() == 3, "MatMul generic body expects lhs, rhs, acc");
            mlir::Value mul;
            mlir::Value sum;
            if (mlir::isa<mlir::FloatType>(elem_ty)) {
                mul = nested_builder.create<mlir::arith::MulFOp>(nested_loc, args[0], args[1]).getResult();
                sum = nested_builder.create<mlir::arith::AddFOp>(nested_loc, args[2], mul).getResult();
            } else {
                mul = nested_builder.create<mlir::arith::MulIOp>(nested_loc, args[0], args[1]).getResult();
                sum = nested_builder.create<mlir::arith::AddIOp>(nested_loc, args[2], mul).getResult();
            }
            nested_builder.create<mlir::linalg::YieldOp>(nested_loc, sum);
        });

    b.create<mlir::func::ReturnOp>(loc, generic.getResult(0));
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
