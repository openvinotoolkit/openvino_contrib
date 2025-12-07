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
#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_unary_from_node(const std::shared_ptr<const ov::Node>& node,
                                          mlir::MLIRContext& ctx,
                                          ActivationKind kind,
                                          float alpha) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();

    const auto shape = node->get_input_shape(0);
    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> dims(shape.begin(), shape.end());
    auto tensor_ty = mlir::RankedTensorType::get(dims, f32);

    mlir::OpBuilder module_builder(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    module_builder.setInsertionPointToStart(module.getBody());

    auto func_type = module_builder.getFunctionType({tensor_ty}, {tensor_ty});
    auto func = module_builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "unary_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto empty = b.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&ctx), dims, f32);

    llvm::SmallVector<mlir::utils::IteratorType> iterators(dims.size(), mlir::utils::IteratorType::parallel);
    auto map = mlir::AffineMap::getMultiDimIdentityMap(dims.size(), &ctx);

    auto generic = b.create<mlir::linalg::GenericOp>(
        mlir::UnknownLoc::get(&ctx),
        tensor_ty,
        mlir::ValueRange{func.getArgument(0)},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{map, map},
        mlir::ArrayRef<mlir::utils::IteratorType>(iterators),
        [&](mlir::OpBuilder& bodyBuilder, mlir::Location loc, mlir::ValueRange args) {
            auto x = args[0];
            auto zero = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(0.0f));
            mlir::Value result;
            switch (kind) {
                case ActivationKind::Relu: {
                    result = bodyBuilder.create<mlir::arith::MaximumFOp>(loc, x, zero);
                    break;
                }
                case ActivationKind::Sigmoid: {
                    auto neg = bodyBuilder.create<mlir::arith::NegFOp>(loc, x);
                    auto exp = bodyBuilder.create<mlir::math::ExpOp>(loc, neg);
                    auto one = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(1.0f));
                    auto denom = bodyBuilder.create<mlir::arith::AddFOp>(loc, one, exp);
                    result = bodyBuilder.create<mlir::arith::DivFOp>(loc, one, denom);
                    break;
                }
                case ActivationKind::Tanh: {
                    result = bodyBuilder.create<mlir::math::TanhOp>(loc, x);
                    break;
                }
                case ActivationKind::Elu: {
                    auto alpha_c = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(alpha));
                    auto exp = bodyBuilder.create<mlir::math::ExpOp>(loc, x);
                    auto one = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(1.0f));
                    auto expm1 = bodyBuilder.create<mlir::arith::SubFOp>(loc, exp, one);
                    auto neg_branch = bodyBuilder.create<mlir::arith::MulFOp>(loc, alpha_c, expm1);
                    auto cond = bodyBuilder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, x, zero);
                    result = bodyBuilder.create<mlir::arith::SelectOp>(loc, cond, x, neg_branch);
                    break;
                }
                case ActivationKind::Prelu: {
                    auto alpha_c = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(alpha));
                    auto neg_branch = bodyBuilder.create<mlir::arith::MulFOp>(loc, alpha_c, x);
                    auto cond = bodyBuilder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, x, zero);
                    result = bodyBuilder.create<mlir::arith::SelectOp>(loc, cond, x, neg_branch);
                    break;
                }
                case ActivationKind::Gelu: {
                    auto half = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(0.5f));
                    auto one = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(1.0f));
                    auto c0 = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(0.79788456f));
                    auto c1 = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(0.044715f));
                    auto x3 = bodyBuilder.create<mlir::arith::MulFOp>(loc, x, x);
                    x3 = bodyBuilder.create<mlir::arith::MulFOp>(loc, x3, x);
                    auto inner = bodyBuilder.create<mlir::arith::AddFOp>(loc, x,
                                                                         bodyBuilder.create<mlir::arith::MulFOp>(loc, c1, x3));
                    auto tanh_arg = bodyBuilder.create<mlir::arith::MulFOp>(loc, c0, inner);
                    auto tanh = bodyBuilder.create<mlir::math::TanhOp>(loc, tanh_arg);
                    auto term = bodyBuilder.create<mlir::arith::AddFOp>(loc, one, tanh);
                    auto mul = bodyBuilder.create<mlir::arith::MulFOp>(loc, half,
                                                                       bodyBuilder.create<mlir::arith::MulFOp>(loc, x, term));
                    result = mul;
                    break;
                }
                case ActivationKind::Swish: {
                    auto neg = bodyBuilder.create<mlir::arith::NegFOp>(loc, x);
                    auto exp = bodyBuilder.create<mlir::math::ExpOp>(loc, neg);
                    auto one = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(1.0f));
                    auto denom = bodyBuilder.create<mlir::arith::AddFOp>(loc, one, exp);
                    auto sigmoid = bodyBuilder.create<mlir::arith::DivFOp>(loc, one, denom);
                    result = bodyBuilder.create<mlir::arith::MulFOp>(loc, x, sigmoid);
                    break;
                }
            }
            bodyBuilder.create<mlir::linalg::YieldOp>(loc, result);
        });

    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), generic.getResults());
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
