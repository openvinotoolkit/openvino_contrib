// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "runtime/gfx_activation.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_unary_from_node(const std::shared_ptr<const ov::Node>& node,
                                          mlir::MLIRContext& ctx,
                                          ActivationKind kind,
                                          float alpha,
                                          std::optional<std::pair<double, double>> clamp_range) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();

    const auto shape = node->get_input_shape(0);
    auto elem_ty = to_mlir_type(node->get_input_element_type(0),
                                ctx,
                                /*fallback_f32=*/true,
                                /*allow_unsigned=*/false,
                                /*allow_small_ints=*/false,
                                /*allow_bf16=*/false,
                                /*allow_boolean=*/true,
                                /*signless_integers=*/true);
    mlir::SmallVector<int64_t> dims(shape.begin(), shape.end());
    auto tensor_ty = mlir::RankedTensorType::get(dims, elem_ty);
    auto make_float_attr = [&](double v) {
        if (auto ft = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
            return mlir::FloatAttr::get(ft, v);
        }
        return mlir::FloatAttr::get(mlir::Float32Type::get(&ctx), v);
    };

    mlir::OpBuilder module_builder(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    module_builder.setInsertionPointToStart(module.getBody());

    auto func_type = module_builder.getFunctionType({tensor_ty}, {tensor_ty});
    auto func = module_builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "unary_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto empty = b.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&ctx), dims, elem_ty);

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
        mlir::Value zero;
        if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
            zero = body.create<mlir::arith::ConstantOp>(block->getArgument(1).getLoc(),
                                                        body.getIntegerAttr(it, 0));
        } else {
            zero = body.create<mlir::arith::ConstantOp>(block->getArgument(1).getLoc(),
                                                        body.getFloatAttr(elem_ty, 0.0));
        }
        mlir::Value result;
        switch (kind) {
            case ActivationKind::Relu: {
                result = body.create<mlir::arith::MaximumFOp>(mlir::UnknownLoc::get(&ctx), x, zero);
                break;
            }
            case ActivationKind::Sigmoid: {
                auto neg = body.create<mlir::arith::NegFOp>(mlir::UnknownLoc::get(&ctx), x);
                auto exp = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), neg);
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(1.0f));
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
                                                                    make_float_attr(alpha));
                auto exp = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), x);
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(1.0f));
                auto expm1 = body.create<mlir::arith::SubFOp>(mlir::UnknownLoc::get(&ctx), exp, one);
                auto neg_branch = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), alpha_c, expm1);
                auto cond = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                             mlir::arith::CmpFPredicate::OGT, x, zero);
                result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, x, neg_branch);
                break;
            }
            case ActivationKind::Prelu: {
                auto alpha_c = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                    make_float_attr(alpha));
                auto neg_branch = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), alpha_c, x);
                auto cond = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                             mlir::arith::CmpFPredicate::OGT, x, zero);
                result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, x, neg_branch);
                break;
            }
            case ActivationKind::Gelu: {
                auto half = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(0.5f));
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(1.0f));
                auto c0 = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(0.79788456f));
                auto c1 = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(0.044715f));
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
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(1.0f));
                auto cond = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                             mlir::arith::CmpFPredicate::OGE, x, zero);
                auto neg = body.create<mlir::arith::NegFOp>(mlir::UnknownLoc::get(&ctx), x);
                auto exp_neg = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), neg);
                auto pos_denom = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), one, exp_neg);
                auto pos = body.create<mlir::arith::DivFOp>(mlir::UnknownLoc::get(&ctx), x, pos_denom);
                auto exp_pos = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), x);
                auto neg_denom = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), one, exp_pos);
                auto neg_num = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), x, exp_pos);
                auto neg_res = body.create<mlir::arith::DivFOp>(mlir::UnknownLoc::get(&ctx), neg_num, neg_denom);
                result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, pos, neg_res);
                break;
            }
            case ActivationKind::HSwish:
            case ActivationKind::HSigmoid: {
                auto three = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(3.0f));
                auto six = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(6.0f));
                auto inv6 = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(1.0f / 6.0f));
                auto x_plus = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), x, three);
                auto max0 = body.create<mlir::arith::MaximumFOp>(mlir::UnknownLoc::get(&ctx), x_plus, zero);
                auto min6 = body.create<mlir::arith::MinimumFOp>(mlir::UnknownLoc::get(&ctx), max0, six);
                auto hsig = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), min6, inv6);
                if (kind == ActivationKind::HSwish) {
                    result = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), x, hsig);
                } else {
                    result = hsig;
                }
                break;
            }
            case ActivationKind::SoftPlus: {
                auto exp = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), x);
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(1.0f));
                auto sum = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), one, exp);
                result = body.create<mlir::math::LogOp>(mlir::UnknownLoc::get(&ctx), sum);
                break;
            }
            case ActivationKind::Mish: {
                auto exp = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), x);
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(1.0f));
                auto sum = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), one, exp);
                auto softplus = body.create<mlir::math::LogOp>(mlir::UnknownLoc::get(&ctx), sum);
                auto t = body.create<mlir::math::TanhOp>(mlir::UnknownLoc::get(&ctx), softplus);
                result = body.create<mlir::arith::MulFOp>(mlir::UnknownLoc::get(&ctx), x, t);
                break;
            }
            case ActivationKind::SoftSign: {
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(1.0f));
                auto is_int = mlir::dyn_cast<mlir::IntegerType>(elem_ty);
                mlir::Value abs_val;
                if (is_int) {
                    auto it = mlir::cast<mlir::IntegerType>(elem_ty);
                    auto cond = body.create<mlir::arith::CmpIOp>(mlir::UnknownLoc::get(&ctx),
                                                                 it.isUnsigned() ? mlir::arith::CmpIPredicate::ult
                                                                                 : mlir::arith::CmpIPredicate::slt,
                                                                 x, zero);
                    auto neg = body.create<mlir::arith::SubIOp>(mlir::UnknownLoc::get(&ctx), zero, x);
                    abs_val = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, neg, x);
                } else {
                    auto cond = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                                 mlir::arith::CmpFPredicate::OLT, x, zero);
                    auto neg = body.create<mlir::arith::NegFOp>(mlir::UnknownLoc::get(&ctx), x);
                    abs_val = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, neg, x);
                }
                auto denom = body.create<mlir::arith::AddFOp>(mlir::UnknownLoc::get(&ctx), one, abs_val);
                result = body.create<mlir::arith::DivFOp>(mlir::UnknownLoc::get(&ctx), x, denom);
                break;
            }
            case ActivationKind::Exp: {
                result = body.create<mlir::math::ExpOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Log: {
                result = body.create<mlir::math::LogOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Sqrt: {
                result = body.create<mlir::math::SqrtOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Floor: {
                result = body.create<mlir::math::FloorOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Ceil: {
                result = body.create<mlir::math::CeilOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Negative: {
                result = body.create<mlir::arith::NegFOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Sin: {
                result = body.create<mlir::math::SinOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Cos: {
                result = body.create<mlir::math::CosOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Tan: {
                result = body.create<mlir::math::TanOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Erf: {
                result = body.create<mlir::math::ErfOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Abs: {
                if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
                    auto cond = body.create<mlir::arith::CmpIOp>(mlir::UnknownLoc::get(&ctx),
                                                                 it.isUnsigned() ? mlir::arith::CmpIPredicate::ult
                                                                                 : mlir::arith::CmpIPredicate::slt,
                                                                 x, zero);
                    auto neg = body.create<mlir::arith::SubIOp>(mlir::UnknownLoc::get(&ctx), zero, x);
                    result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, neg, x);
                } else {
                    auto cond = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                                 mlir::arith::CmpFPredicate::OLT, x, zero);
                    auto neg = body.create<mlir::arith::NegFOp>(mlir::UnknownLoc::get(&ctx), x);
                    result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, neg, x);
                }
                break;
            }
            case ActivationKind::Sign: {
                if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
                    auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                    body.getIntegerAttr(it, 1));
                    if (it.isUnsigned()) {
                        auto cond = body.create<mlir::arith::CmpIOp>(mlir::UnknownLoc::get(&ctx),
                                                                     mlir::arith::CmpIPredicate::ugt, x, zero);
                        result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, one, zero);
                    } else {
                        auto neg_one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                            body.getIntegerAttr(it, -1));
                        auto cond_pos = body.create<mlir::arith::CmpIOp>(mlir::UnknownLoc::get(&ctx),
                                                                         mlir::arith::CmpIPredicate::sgt, x, zero);
                        auto cond_neg = body.create<mlir::arith::CmpIOp>(mlir::UnknownLoc::get(&ctx),
                                                                         mlir::arith::CmpIPredicate::slt, x, zero);
                        auto neg_or_zero = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond_neg, neg_one, zero);
                        result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond_pos, one, neg_or_zero);
                    }
                } else {
                    auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(1.0f));
                    auto neg_one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx), make_float_attr(-1.0f));
                    auto cond_pos = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                                     mlir::arith::CmpFPredicate::OGT, x, zero);
                    auto cond_neg = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                                     mlir::arith::CmpFPredicate::OLT, x, zero);
                    auto neg_or_zero = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond_neg, neg_one, zero);
                    result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond_pos, one, neg_or_zero);
                }
                break;
            }
            case ActivationKind::Asin: {
                result = body.create<mlir::math::AsinOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Acos: {
                result = body.create<mlir::math::AcosOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Atan: {
                result = body.create<mlir::math::AtanOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Asinh: {
                result = body.create<mlir::math::AsinhOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Acosh: {
                result = body.create<mlir::math::AcoshOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Atanh: {
                result = body.create<mlir::math::AtanhOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Sinh: {
                result = body.create<mlir::math::SinhOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Cosh: {
                result = body.create<mlir::math::CoshOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::RoundEven: {
                result = body.create<mlir::math::RoundEvenOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::RoundAway: {
                result = body.create<mlir::math::RoundOp>(mlir::UnknownLoc::get(&ctx), x);
                break;
            }
            case ActivationKind::Clamp: {
                OPENVINO_ASSERT(clamp_range.has_value(), "Clamp requires min/max range");
                auto clamp_min = clamp_range->first;
                auto clamp_max = clamp_range->second;
                mlir::Value min_c;
                mlir::Value max_c;
                if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
                    min_c = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                 body.getIntegerAttr(it, static_cast<int64_t>(clamp_min)));
                    max_c = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                 body.getIntegerAttr(it, static_cast<int64_t>(clamp_max)));
                    auto cmp_min = body.create<mlir::arith::CmpIOp>(mlir::UnknownLoc::get(&ctx),
                                                                    it.isUnsigned() ? mlir::arith::CmpIPredicate::ult
                                                                                    : mlir::arith::CmpIPredicate::slt,
                                                                    x, min_c);
                    auto after_min = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cmp_min, min_c, x);
                    auto cmp_max = body.create<mlir::arith::CmpIOp>(mlir::UnknownLoc::get(&ctx),
                                                                    it.isUnsigned() ? mlir::arith::CmpIPredicate::ugt
                                                                                    : mlir::arith::CmpIPredicate::sgt,
                                                                    after_min, max_c);
                    result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cmp_max, max_c, after_min);
                } else {
                    min_c = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                 body.getFloatAttr(elem_ty, clamp_min));
                    max_c = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                 body.getFloatAttr(elem_ty, clamp_max));
                    auto cmp_min = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                                    mlir::arith::CmpFPredicate::OLT, x, min_c);
                    auto after_min = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cmp_min, min_c, x);
                    auto cmp_max = body.create<mlir::arith::CmpFOp>(mlir::UnknownLoc::get(&ctx),
                                                                    mlir::arith::CmpFPredicate::OGT, after_min, max_c);
                    result = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cmp_max, max_c, after_min);
                }
                break;
            }
            case ActivationKind::LogicalNot: {
                auto one = body.create<mlir::arith::ConstantOp>(mlir::UnknownLoc::get(&ctx),
                                                                body.getIntegerAttr(elem_ty, 1));
                result = body.create<mlir::arith::XOrIOp>(mlir::UnknownLoc::get(&ctx), x, one);
                break;
            }
        }
        body.create<mlir::linalg::YieldOp>(mlir::UnknownLoc::get(&ctx), result);
    }

    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), generic.getResults());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
