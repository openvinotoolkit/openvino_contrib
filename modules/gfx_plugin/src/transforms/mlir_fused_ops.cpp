// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/mlir_fused_ops.hpp"

#include <cmath>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AffineMap.h"

#include "transforms/fusion_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

mlir::func::FuncOp find_entry_func(mlir::ModuleOp module) {
    mlir::func::FuncOp entry;
    module.walk([&](mlir::func::FuncOp func) {
        if (!entry) {
            entry = func;
        }
    });
    return entry;
}

mlir::func::ReturnOp find_return_op(mlir::func::FuncOp func) {
    if (!func || func.getBody().empty()) {
        return {};
    }
    auto& block = func.getBody().front();
    if (auto* term = block.getTerminator()) {
        return mlir::dyn_cast<mlir::func::ReturnOp>(term);
    }
    return {};
}

mlir::Value emit_activation(mlir::OpBuilder& b,
                            mlir::Location loc,
                            mlir::Value x,
                            ActivationKind kind,
                            float alpha,
                            mlir::Type elem_ty) {
    auto make_float_attr = [&](double v) { return mlir::FloatAttr::get(elem_ty, v); };

    mlir::Value zero = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.0));
    switch (kind) {
        case ActivationKind::Relu: {
            return b.create<mlir::arith::MaximumFOp>(loc, x, zero);
        }
        case ActivationKind::Sigmoid: {
            auto neg = b.create<mlir::arith::NegFOp>(loc, x);
            auto exp = b.create<mlir::math::ExpOp>(loc, neg);
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto denom = b.create<mlir::arith::AddFOp>(loc, one, exp);
            return b.create<mlir::arith::DivFOp>(loc, one, denom);
        }
        case ActivationKind::Tanh: {
            return b.create<mlir::math::TanhOp>(loc, x);
        }
        case ActivationKind::Elu: {
            auto alpha_c = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(alpha));
            auto exp = b.create<mlir::math::ExpOp>(loc, x);
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto expm1 = b.create<mlir::arith::SubFOp>(loc, exp, one);
            auto neg_branch = b.create<mlir::arith::MulFOp>(loc, alpha_c, expm1);
            auto cond = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, x, zero);
            return b.create<mlir::arith::SelectOp>(loc, cond, x, neg_branch);
        }
        case ActivationKind::Prelu: {
            auto alpha_c = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(alpha));
            auto neg_branch = b.create<mlir::arith::MulFOp>(loc, alpha_c, x);
            auto cond = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, x, zero);
            return b.create<mlir::arith::SelectOp>(loc, cond, x, neg_branch);
        }
        case ActivationKind::Gelu: {
            auto half = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.5));
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto c0 = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.79788456));
            auto c1 = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(0.044715));
            auto x2 = b.create<mlir::arith::MulFOp>(loc, x, x);
            auto x3 = b.create<mlir::arith::MulFOp>(loc, x2, x);
            auto inner = b.create<mlir::arith::AddFOp>(loc, x, b.create<mlir::arith::MulFOp>(loc, c1, x3));
            auto tanh_arg = b.create<mlir::arith::MulFOp>(loc, c0, inner);
            auto tanh = b.create<mlir::math::TanhOp>(loc, tanh_arg);
            auto term = b.create<mlir::arith::AddFOp>(loc, one, tanh);
            auto mul = b.create<mlir::arith::MulFOp>(loc, half, b.create<mlir::arith::MulFOp>(loc, x, term));
            return mul;
        }
        case ActivationKind::Swish: {
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto cond = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, x, zero);
            auto neg = b.create<mlir::arith::NegFOp>(loc, x);
            auto exp_neg = b.create<mlir::math::ExpOp>(loc, neg);
            auto pos_denom = b.create<mlir::arith::AddFOp>(loc, one, exp_neg);
            auto pos = b.create<mlir::arith::DivFOp>(loc, x, pos_denom);
            auto exp_pos = b.create<mlir::math::ExpOp>(loc, x);
            auto neg_denom = b.create<mlir::arith::AddFOp>(loc, one, exp_pos);
            auto neg_num = b.create<mlir::arith::MulFOp>(loc, x, exp_pos);
            auto neg_res = b.create<mlir::arith::DivFOp>(loc, neg_num, neg_denom);
            return b.create<mlir::arith::SelectOp>(loc, cond, pos, neg_res);
        }
        case ActivationKind::HSwish:
        case ActivationKind::HSigmoid: {
            auto three = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(3.0));
            auto six = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(6.0));
            auto inv6 = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0 / 6.0));
            auto x_plus = b.create<mlir::arith::AddFOp>(loc, x, three);
            auto max0 = b.create<mlir::arith::MaximumFOp>(loc, x_plus, zero);
            auto min6 = b.create<mlir::arith::MinimumFOp>(loc, max0, six);
            auto hsig = b.create<mlir::arith::MulFOp>(loc, min6, inv6);
            if (kind == ActivationKind::HSwish) {
                return b.create<mlir::arith::MulFOp>(loc, x, hsig);
            }
            return hsig;
        }
        case ActivationKind::Abs: {
            auto cond = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, x, zero);
            auto neg = b.create<mlir::arith::NegFOp>(loc, x);
            return b.create<mlir::arith::SelectOp>(loc, cond, neg, x);
        }
        case ActivationKind::Sign: {
            auto one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(1.0));
            auto neg_one = b.create<mlir::arith::ConstantOp>(loc, make_float_attr(-1.0));
            auto gt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, x, zero);
            auto lt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, x, zero);
            auto pos = b.create<mlir::arith::SelectOp>(loc, gt, one, zero);
            return b.create<mlir::arith::SelectOp>(loc, lt, neg_one, pos);
        }
        default:
            break;
    }
    return x;
}

bool annotate_conv_activation(mlir::ModuleOp module, ActivationKind kind, float alpha) {
    mlir::linalg::Conv2DNchwFchwOp conv;
    module.walk([&](mlir::linalg::Conv2DNchwFchwOp op) {
        if (!conv) {
            conv = op;
        }
    });
    if (!conv) {
        return false;
    }
    auto* ctx = module.getContext();
    conv->setAttr("gfx.activation_kind",
                  mlir::StringAttr::get(ctx, fusion_utils::activation_kind_name(kind)));
    conv->setAttr("gfx.activation_alpha", mlir::FloatAttr::get(mlir::Float32Type::get(ctx), alpha));
    return true;
}

bool annotate_conv_bias(mlir::ModuleOp module, const BiasParams& params) {
    if (!module || params.empty()) {
        return false;
    }
    mlir::linalg::Conv2DNchwFchwOp conv;
    module.walk([&](mlir::linalg::Conv2DNchwFchwOp op) {
        if (!conv) {
            conv = op;
        }
    });
    if (!conv) {
        return false;
    }
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getOutputs()[0].getType());
    if (!out_type || out_type.getRank() != 4) {
        return false;
    }
    auto elem_ty = out_type.getElementType();
    if (!mlir::isa<mlir::FloatType>(elem_ty)) {
        return false;
    }
    int64_t channels = out_type.getDimSize(1);
    if (channels == mlir::ShapedType::kDynamic) {
        if (params.shape.size() == 4) {
            channels = params.shape[1];
        } else if (params.shape.size() == 1) {
            channels = params.shape[0];
        }
    }
    if (channels <= 0 || params.values.size() != static_cast<size_t>(channels)) {
        return false;
    }
    auto c_type = mlir::RankedTensorType::get({channels}, elem_ty);
    auto bias_attr = mlir::DenseFPElementsAttr::get(c_type, params.values);
    conv->setAttr("gfx.bias", bias_attr);
    return true;
}

bool apply_activation_to_generic(mlir::ModuleOp module, ActivationKind kind, float alpha) {
    auto func = find_entry_func(module);
    auto ret = find_return_op(func);
    if (!func || !ret || ret.getNumOperands() != 1) {
        return false;
    }
    auto result = ret.getOperand(0);
    auto generic = result.getDefiningOp<mlir::linalg::GenericOp>();
    if (!generic) {
        return false;
    }
    bool has_reduction = false;
    if (auto iter_attr = generic.getIteratorTypesAttr()) {
        for (auto attr : iter_attr) {
            if (auto str = mlir::dyn_cast<mlir::StringAttr>(attr)) {
                if (str.getValue() == "reduction") {
                    has_reduction = true;
                    break;
                }
            }
        }
    }
    if (has_reduction) {
        auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
        if (!out_type) {
            return false;
        }
        auto elem_ty = out_type.getElementType();
        if (!mlir::isa<mlir::FloatType>(elem_ty)) {
            return false;
        }
        mlir::OpBuilder b(ret);
        auto loc = ret.getLoc();
        auto empty = b.create<mlir::tensor::EmptyOp>(loc, out_type.getShape(), elem_ty);
        const auto rank = out_type.getRank();
        auto map = mlir::AffineMap::getMultiDimIdentityMap(rank, b.getContext());
        llvm::SmallVector<mlir::utils::IteratorType, 4> iters;
        iters.assign(static_cast<size_t>(rank), mlir::utils::IteratorType::parallel);
        auto fused = b.create<mlir::linalg::GenericOp>(
            loc,
            out_type,
            mlir::ValueRange{result},
            mlir::ValueRange{empty.getResult()},
            mlir::ArrayRef<mlir::AffineMap>{map, map},
            llvm::ArrayRef<mlir::utils::IteratorType>(iters));
        auto& region = fused.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty}, {loc, loc});
        mlir::OpBuilder body(block, block->begin());
        auto activated = emit_activation(body, loc, block->getArgument(0), kind, alpha, elem_ty);
        body.create<mlir::linalg::YieldOp>(loc, activated);
        ret.setOperand(0, fused.getResult(0));
        return true;
    }
    auto* block = generic.getBody();
    auto* term = block ? block->getTerminator() : nullptr;
    auto yield = mlir::dyn_cast_or_null<mlir::linalg::YieldOp>(term);
    if (!yield || yield.getNumOperands() != 1) {
        return false;
    }
    auto elem_ty = yield.getOperand(0).getType();
    if (!mlir::isa<mlir::FloatType>(elem_ty)) {
        return false;
    }
    mlir::OpBuilder b(yield);
    auto loc = yield.getLoc();
    auto activated = emit_activation(b, loc, yield.getOperand(0), kind, alpha, elem_ty);
    yield.setOperand(0, activated);
    return true;
}

bool apply_activation_to_output(mlir::ModuleOp module, ActivationKind kind, float alpha) {
    auto func = find_entry_func(module);
    auto ret = find_return_op(func);
    if (!func || !ret || ret.getNumOperands() != 1) {
        return false;
    }
    auto result = ret.getOperand(0);
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!out_type) {
        return false;
    }
    auto elem_ty = out_type.getElementType();
    if (!mlir::isa<mlir::FloatType>(elem_ty)) {
        return false;
    }
    const int64_t rank = out_type.getRank();
    if (rank <= 0) {
        return false;
    }
    mlir::OpBuilder b(ret);
    auto loc = ret.getLoc();
    auto empty = b.create<mlir::tensor::EmptyOp>(loc, out_type.getShape(), elem_ty);
    auto map = mlir::AffineMap::getMultiDimIdentityMap(rank, b.getContext());
    llvm::SmallVector<mlir::utils::IteratorType, 4> iters;
    iters.assign(static_cast<size_t>(rank), mlir::utils::IteratorType::parallel);
    auto fused = b.create<mlir::linalg::GenericOp>(
        loc,
        out_type,
        mlir::ValueRange{result},
        mlir::ValueRange{empty.getResult()},
        mlir::ArrayRef<mlir::AffineMap>{map, map},
        llvm::ArrayRef<mlir::utils::IteratorType>(iters));
    {
        auto& region = fused.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty}, {loc, loc});
        mlir::OpBuilder body(block, block->begin());
        auto activated = emit_activation(body, loc, block->getArgument(0), kind, alpha, elem_ty);
        body.create<mlir::linalg::YieldOp>(loc, activated);
    }
    ret.setOperand(0, fused.getResult(0));
    return true;
}

bool apply_fused_activation_impl(mlir::ModuleOp module, ActivationKind kind, float alpha) {
    if (!module) {
        return false;
    }
    module.getContext()->loadDialect<mlir::func::FuncDialect,
                                     mlir::linalg::LinalgDialect,
                                     mlir::tensor::TensorDialect,
                                     mlir::arith::ArithDialect,
                                     mlir::math::MathDialect>();
    if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.post_activation_only")) {
        if (attr.getValue()) {
            return apply_activation_to_output(module, kind, alpha);
        }
    }
    if (annotate_conv_activation(module, kind, alpha)) {
        return true;
    }
    return apply_activation_to_generic(module, kind, alpha);
}

bool apply_fused_batchnorm_impl(mlir::ModuleOp module, const BatchNormParams& params) {
    if (!module || params.empty()) {
        return false;
    }
    module.getContext()->loadDialect<mlir::func::FuncDialect,
                                     mlir::linalg::LinalgDialect,
                                     mlir::tensor::TensorDialect,
                                     mlir::arith::ArithDialect>();

    mlir::linalg::Conv2DNchwFchwOp conv;
    module.walk([&](mlir::linalg::Conv2DNchwFchwOp op) {
        if (!conv) {
            conv = op;
        }
    });
    if (!conv) {
        return false;
    }
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(conv.getOutputs()[0].getType());
    if (!out_type) {
        return false;
    }
    auto elem_ty = out_type.getElementType();
    if (!mlir::isa<mlir::FloatType>(elem_ty)) {
        return false;
    }
    if (out_type.getRank() < 2) {
        return false;
    }
    const int64_t channels = static_cast<int64_t>(params.gamma.size());
    if (channels <= 0 ||
        params.beta.size() != static_cast<size_t>(channels) ||
        params.mean.size() != static_cast<size_t>(channels) ||
        params.var.size() != static_cast<size_t>(channels)) {
        return false;
    }
    if (out_type.getDimSize(1) != mlir::ShapedType::kDynamic &&
        out_type.getDimSize(1) != channels) {
        return false;
    }

    std::vector<float> scale_vals(channels);
    std::vector<float> bias_vals(channels);
    for (int64_t c = 0; c < channels; ++c) {
        const float gamma = params.gamma[c];
        const float beta = params.beta[c];
        const float mean = params.mean[c];
        const float var = params.var[c];
        const float inv_std = 1.0f / std::sqrt(var + params.epsilon);
        const float scale = gamma * inv_std;
        const float bias = beta - mean * scale;
        scale_vals[c] = scale;
        bias_vals[c] = bias;
    }

    auto c_type = mlir::RankedTensorType::get({channels}, elem_ty);
    auto scale_attr = mlir::DenseFPElementsAttr::get(c_type, scale_vals);
    auto bias_attr = mlir::DenseFPElementsAttr::get(c_type, bias_vals);
    conv->setAttr("gfx.bn_scale", scale_attr);
    conv->setAttr("gfx.bn_bias", bias_attr);
    return true;
}

}  // namespace

bool apply_fused_bias_impl(mlir::ModuleOp module, const BiasParams& params) {
    if (!module || params.empty()) {
        return false;
    }
    module.getContext()->loadDialect<mlir::func::FuncDialect,
                                     mlir::linalg::LinalgDialect,
                                     mlir::tensor::TensorDialect,
                                     mlir::arith::ArithDialect>();

    if (annotate_conv_bias(module, params)) {
        return true;
    }

    auto func = find_entry_func(module);
    auto ret = find_return_op(func);
    if (!func || !ret || ret.getNumOperands() != 1) {
        return false;
    }
    auto result = ret.getOperand(0);
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!out_type) {
        return false;
    }
    auto elem_ty = out_type.getElementType();
    if (!mlir::isa<mlir::FloatType>(elem_ty)) {
        return false;
    }

    const int64_t out_rank = out_type.getRank();
    if (out_rank <= 0) {
        return false;
    }
    if (static_cast<int64_t>(params.shape.size()) > out_rank) {
        return false;
    }
    std::vector<int64_t> aligned_shape(out_rank, 1);
    const int64_t offset = out_rank - static_cast<int64_t>(params.shape.size());
    for (size_t i = 0; i < params.shape.size(); ++i) {
        aligned_shape[static_cast<size_t>(offset) + i] = params.shape[i];
    }
    for (int64_t i = 0; i < out_rank; ++i) {
        const int64_t bias_dim = aligned_shape[static_cast<size_t>(i)];
        if (bias_dim <= 0) {
            return false;
        }
        const int64_t out_dim = out_type.getDimSize(i);
        if (out_dim != mlir::ShapedType::kDynamic && bias_dim != 1 && bias_dim != out_dim) {
            return false;
        }
    }
    size_t expected = 1;
    for (auto d : params.shape) {
        expected *= static_cast<size_t>(d);
    }
    if (expected != params.values.size()) {
        return false;
    }

    mlir::OpBuilder b(ret);
    auto loc = ret.getLoc();
    auto bias_type = mlir::RankedTensorType::get(aligned_shape, elem_ty);
    auto func_type = func.getFunctionType();
    llvm::SmallVector<mlir::Type, 8> inputs(func_type.getInputs().begin(),
                                            func_type.getInputs().end());
    inputs.push_back(bias_type);
    auto new_type = mlir::FunctionType::get(func.getContext(), inputs, func_type.getResults());
    func.setType(new_type);
    auto& entry = func.getBody().front();
    auto bias_arg = entry.addArgument(bias_type, loc);

    auto out_shape = out_type.getShape();
    auto out_empty = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);
    auto out_map = mlir::AffineMap::getMultiDimIdentityMap(out_rank, b.getContext());
    llvm::SmallVector<mlir::AffineExpr, 4> bias_exprs;
    bias_exprs.reserve(static_cast<size_t>(out_rank));
    for (int64_t i = 0; i < out_rank; ++i) {
        if (aligned_shape[static_cast<size_t>(i)] == 1) {
            bias_exprs.push_back(b.getAffineConstantExpr(0));
        } else {
            bias_exprs.push_back(b.getAffineDimExpr(i));
        }
    }
    auto bias_map = mlir::AffineMap::get(out_rank, 0, bias_exprs, b.getContext());
    llvm::SmallVector<mlir::utils::IteratorType, 4> iters;
    iters.assign(static_cast<size_t>(out_rank), mlir::utils::IteratorType::parallel);

    auto fused = b.create<mlir::linalg::GenericOp>(
        loc,
        out_type,
        mlir::ValueRange{result, bias_arg},
        mlir::ValueRange{out_empty.getResult()},
        mlir::ArrayRef<mlir::AffineMap>{out_map, bias_map, out_map},
        llvm::ArrayRef<mlir::utils::IteratorType>(iters));
    {
        auto& region = fused.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty, elem_ty}, {loc, loc, loc});
        mlir::OpBuilder body(block, block->begin());
        auto sum = body.create<mlir::arith::AddFOp>(loc, block->getArgument(0), block->getArgument(1));
        body.create<mlir::linalg::YieldOp>(loc, sum.getResult());
    }
    ret.setOperand(0, fused.getResult(0));
    module->setAttr("gfx.post_activation_only", mlir::BoolAttr::get(module.getContext(), true));
    return true;
}

bool apply_fused_activation(mlir::ModuleOp module, ActivationKind kind, float alpha) {
    return apply_fused_activation_impl(module, kind, alpha);
}

bool apply_fused_batchnorm(mlir::ModuleOp module, const BatchNormParams& params) {
    return apply_fused_batchnorm_impl(module, params);
}

bool apply_fused_bias(mlir::ModuleOp module, const BiasParams& params) {
    return apply_fused_bias_impl(module, params);
}

}  // namespace gfx_plugin
}  // namespace ov
