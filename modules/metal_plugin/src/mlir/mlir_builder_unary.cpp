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
        mlir::ArrayRef<mlir::utils::IteratorType>(iterators));
    {
        // Explicitly add one block argument per input and per output to avoid verifier
        // complaints when the default builder drops output args.
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({tensor_ty.getElementType(), tensor_ty.getElementType()},
                            {mlir::UnknownLoc::get(&ctx), mlir::UnknownLoc::get(&ctx)});
        mlir::OpBuilder body(block, block->begin());
        auto x = block->getArgument(0);
        auto zero = body.create<mlir::arith::ConstantOp>(block->getArgument(1).getLoc(),
                                                         body.getF32FloatAttr(0.0f));
        mlir::Value result;
        switch (kind) {
            case ActivationKind::Relu: {
                result = body.create<mlir::arith::MaximumFOp>(mlir::UnknownLoc::get(&ctx), x, zero);
                break;
            }
            case ActivationKind::Sigmoid: {
                auto neg = body.create<mlir::arith::NegFOp>(mlir::UnknownLoc::get(&ctx), x);
                auto exp = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), neg);
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), body.getF32FloatAttr(1.0f));
                auto denom = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), one, exp);
                result = body.create<mlir::arith::DivFOp>(mlir::UnknownLoc::get(&ctx), one, denom);
                break;
            }
            case ActivationKind::Tanh: {
                result = body.create<mlir::math::TanhOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Elu: {
                auto alpha_c = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                    body.getF32FloatAttr(alpha));
                auto exp = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), x);
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), body.getF32FloatAttr(1.0f));
                auto expm1 = body.create<mlir::arith::SubFOp>(mlir::UnknownLoc::get(&ctx), exp, one);
                auto neg_branch = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), alpha_c, expm1);
                auto cond = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                             mlir::arith::CmpFPredicate::OGT, x, zero);
                result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, x, neg_branch);
                break;
            }
            case ActivationKind::Prelu: {
                auto alpha_c = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                    body.getF32FloatAttr(alpha));
                auto neg_branch = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), alpha_c, x);
                auto cond = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                             mlir::arith::CmpFPredicate::OGT, x, zero);
                result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, x, neg_branch);
                break;
            }
            case ActivationKind::Gelu: {
                auto half = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), body.getF32FloatAttr(0.5f));
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), body.getF32FloatAttr(1.0f));
                auto c0 = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), body.getF32FloatAttr(0.79788456f));
                auto c1 = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), body.getF32FloatAttr(0.044715f));
                auto x3 = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), x, x);
                x3 = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), x3, x);
                auto inner = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), x,
                                                              body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), c1, x3));
                auto tanh_arg = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), c0, inner);
                auto tanh = body.create<mlir::math::TanhOp>(mlir::UnknownLoc::get(&ctx), tanh_arg);
                auto term = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), one, tanh);
                auto mul = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), half,
                                                            body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), x, term));
                result = mul;
                break;
            }
            case ActivationKind::Swish: {
                auto neg = body.create<mlir::arith::NegFOp>(mlir::UnknownLoc::get(&ctx), x);
                auto exp = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), neg);
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), body.getF32FloatAttr(1.0f));
                auto denom = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), one, exp);
                auto sigmoid = body.create<mlir::arith::DivFOp>(mlir::UnknownLoc::get(&ctx), one, denom);
                result = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), x, sigmoid);
                break;
            }
        }
        body.create<mlir::linalg::YieldOp>(mlir::UnknownLoc::get(&ctx), result);
    }

    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), generic.getResults());
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
