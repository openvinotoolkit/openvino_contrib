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

#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/batch_norm.hpp"
#if __has_include("openvino/op/layer_norm.hpp")
#include "openvino/op/layer_norm.hpp"
#endif
#include "openvino/op/constant.hpp"
#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
namespace metal_plugin {

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

std::shared_ptr<const ov::op::v1::Add> find_single_add(const std::shared_ptr<const ov::Model>& model) {
    std::shared_ptr<const ov::op::v1::Add> add;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto a = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
            OPENVINO_ASSERT(!add, "Only single Add is supported in MLIR Add builder for now");
            add = a;
        }
    }
    OPENVINO_ASSERT(add, "MLIR Add builder: Add op not found");
    return add;
}

std::shared_ptr<const ov::Node> find_single_softmax(const std::shared_ptr<const ov::Model>& model) {
    std::shared_ptr<const ov::Node> sm;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v1::Softmax>(node.get()) || ov::is_type<ov::op::v8::Softmax>(node.get())) {
            OPENVINO_ASSERT(!sm, "Only single Softmax is supported in MLIR Softmax builder for now");
            sm = node;
        }
    }
    OPENVINO_ASSERT(sm, "MLIR Softmax builder: Softmax op not found");
    return sm;
}
}  // namespace

mlir::ModuleOp build_mlir_module_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();

    auto matmul = find_single_matmul(model);
    OPENVINO_ASSERT(!matmul->get_transpose_a() && !matmul->get_transpose_b(), "Transposed MatMul is not supported");
    const auto shape_a = matmul->get_input_shape(0);
    const auto shape_b = matmul->get_input_shape(1);
    OPENVINO_ASSERT(shape_a.size() == 2 && shape_b.size() == 2, "Only 2D MatMul is supported");
    const int64_t M = static_cast<int64_t>(shape_a[0]);
    const int64_t K = static_cast<int64_t>(shape_a[1]);
    OPENVINO_ASSERT(shape_b[0] == static_cast<size_t>(K), "MatMul K dimension mismatch");
    const int64_t N = static_cast<int64_t>(shape_b[1]);

    auto f32 = mlir::Float32Type::get(&ctx);
    auto type_a = mlir::RankedTensorType::get({M, K}, f32);
    auto type_b = mlir::RankedTensorType::get({K, N}, f32);
    auto type_c = mlir::RankedTensorType::get({M, N}, f32);

    mlir::OpBuilder module_builder(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    module_builder.setInsertionPointToStart(module.getBody());

    auto func_type = module_builder.getFunctionType({type_a, type_b}, {type_c});
    auto func = module_builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "matmul_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto empty = b.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&ctx), mlir::ArrayRef<int64_t>({M, N}), f32);
    auto mm = b.create<mlir::linalg::MatmulOp>(mlir::UnknownLoc::get(&ctx),
                                               mlir::ValueRange{func.getArgument(0), func.getArgument(1)},
                                               mlir::ValueRange{empty});
    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), mm.getResults());

    return module;
}

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

mlir::ModuleOp build_mlir_broadcast_add_from_model(const std::shared_ptr<const ov::Model>& model,
                                                   mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();

    auto add = find_single_add(model);
    const auto shape0 = add->get_input_shape(0);
    const auto shape1 = add->get_input_shape(1);
    const auto out_shape = add->get_output_shape(0);
    const size_t rank = out_shape.size();

    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> s0(shape0.begin(), shape0.end());
    mlir::SmallVector<int64_t> s1(shape1.begin(), shape1.end());
    mlir::SmallVector<int64_t> sout(out_shape.begin(), out_shape.end());
    auto ty0 = mlir::RankedTensorType::get(s0, f32);
    auto ty1 = mlir::RankedTensorType::get(s1, f32);
    auto ty_out = mlir::RankedTensorType::get(sout, f32);

    mlir::OpBuilder module_builder(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    module_builder.setInsertionPointToStart(module.getBody());

    auto func_type = module_builder.getFunctionType({ty0, ty1}, {ty_out});
    auto func = module_builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "add_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto empty = b.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&ctx), sout, f32);

    llvm::SmallVector<mlir::utils::IteratorType> iterators(rank, mlir::utils::IteratorType::parallel);

    auto make_broadcast_map = [&](const ov::Shape& in_shape) {
        size_t in_rank = in_shape.size();
        llvm::SmallVector<mlir::AffineExpr> exprs;
        exprs.reserve(in_rank);
        size_t start = rank - in_rank;
        for (size_t i = 0; i < in_rank; ++i) {
            if (in_shape[i] == 1) {
                exprs.push_back(mlir::getAffineConstantExpr(0, &ctx));
            } else {
                exprs.push_back(mlir::getAffineDimExpr(start + i, &ctx));
            }
        }
        return mlir::AffineMap::get(rank, 0, exprs, &ctx);
    };

    auto map0 = make_broadcast_map(shape0);
    auto map1 = make_broadcast_map(shape1);
    auto map_out = mlir::AffineMap::getMultiDimIdentityMap(rank, &ctx);

    auto generic = b.create<mlir::linalg::GenericOp>(
        mlir::UnknownLoc::get(&ctx),
        ty_out,
        mlir::ValueRange{func.getArgument(0), func.getArgument(1)},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{map0, map1, map_out},
        mlir::ArrayRef<mlir::utils::IteratorType>(iterators),
        [&](mlir::OpBuilder& bodyBuilder, mlir::Location loc, mlir::ValueRange args) {
            auto sum = bodyBuilder.create<mlir::arith::AddFOp>(loc, args[0], args[1]);
            bodyBuilder.create<mlir::linalg::YieldOp>(loc, sum.getResult());
        });

    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), generic.getResults());
    return module;
}

mlir::ModuleOp build_mlir_softmax_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx) {
    // For now emit a trivial identity function to satisfy MLIR pipeline; computation is handled by kernel IR.
    ctx.loadDialect<mlir::func::FuncDialect>();
    auto sm = find_single_softmax(model);
    const auto shape = sm->get_input_shape(0);
    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> dims(shape.begin(), shape.end());
    auto ty = mlir::RankedTensorType::get(dims, f32);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({ty}, {ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "softmax_main", func_type);
    func.addEntryBlock();
    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), func.getArguments());
    return module;
}

mlir::ModuleOp build_mlir_maxpool_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect>();
    std::shared_ptr<const ov::op::v1::MaxPool> pool;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto p = ov::as_type_ptr<const ov::op::v1::MaxPool>(node)) {
            pool = p;
            break;
        }
    }
    OPENVINO_ASSERT(pool, "MaxPool builder: MaxPool op not found");
    auto shape = pool->get_input_shape(0);
    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> dims(shape.begin(), shape.end());
    auto ty = mlir::RankedTensorType::get(dims, f32);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({ty}, {ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "maxpool_main", func_type);
    func.addEntryBlock();
    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), func.getArguments());
    return module;
}

mlir::ModuleOp build_mlir_avgpool_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect>();
    std::shared_ptr<const ov::op::v1::AvgPool> pool;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto p = ov::as_type_ptr<const ov::op::v1::AvgPool>(node)) {
            pool = p;
            break;
        }
    }
    OPENVINO_ASSERT(pool, "AvgPool builder: AvgPool op not found");
    auto shape = pool->get_input_shape(0);
    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> dims(shape.begin(), shape.end());
    auto ty = mlir::RankedTensorType::get(dims, f32);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({ty}, {ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "avgpool_main", func_type);
    func.addEntryBlock();
    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), func.getArguments());
    return module;
}

mlir::ModuleOp build_mlir_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect>();
    std::shared_ptr<const ov::op::v1::Convolution> conv;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            conv = c;
            break;
        }
    }
    OPENVINO_ASSERT(conv, "Conv2D builder: Convolution op not found");
    auto shape = conv->get_input_shape(0);
    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> dims(shape.begin(), shape.end());
    auto ty = mlir::RankedTensorType::get(dims, f32);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({ty}, {ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "conv2d_main", func_type);
    func.addEntryBlock();
    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), func.getArguments());
    return module;
}

mlir::ModuleOp build_mlir_batchnorm_from_model(const std::shared_ptr<const ov::Model>& model,
                                               mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();

    std::shared_ptr<const ov::op::v5::BatchNormInference> bn;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto b = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node)) {
            bn = b;
            break;
        }
    }
    OPENVINO_ASSERT(bn, "BatchNorm builder: node not found");

    const auto& in_shape = bn->get_input_shape(0);  // NCHW expected
    OPENVINO_ASSERT(in_shape.size() == 4, "BatchNorm builder expects rank-4 input");
    const size_t C = in_shape[1];

    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> dims(in_shape.begin(), in_shape.end());
    auto inputTy = mlir::RankedTensorType::get(dims, f32);
    auto outputTy = inputTy;
    auto paramTy = mlir::RankedTensorType::get({static_cast<int64_t>(C)}, f32);

    // Fetch parameter constants; fallback will be triggered in the caller if they are not constants.
    auto get_vec = [&](size_t idx) -> std::vector<float> {
        auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(bn->get_input_node_shared_ptr(idx));
        OPENVINO_ASSERT(c, "BatchNorm builder expects constant parameter");
        auto vec = c->cast_vector<float>();
        OPENVINO_ASSERT(vec.size() == C, "BatchNorm parameter size mismatch");
        return vec;
    };

    const auto gamma_vec = get_vec(1);
    const auto beta_vec  = get_vec(2);
    const auto mean_vec  = get_vec(3);
    const auto var_vec   = get_vec(4);
    const float eps = static_cast<float>(bn->get_eps_value());

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({inputTy}, {outputTy});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "batchnorm_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);

    auto gamma_c = b.create<mlir::arith::ConstantOp>(loc, paramTy,
        mlir::DenseElementsAttr::get(paramTy, llvm::ArrayRef<float>(gamma_vec)));
    auto beta_c = b.create<mlir::arith::ConstantOp>(loc, paramTy,
        mlir::DenseElementsAttr::get(paramTy, llvm::ArrayRef<float>(beta_vec)));
    auto mean_c = b.create<mlir::arith::ConstantOp>(loc, paramTy,
        mlir::DenseElementsAttr::get(paramTy, llvm::ArrayRef<float>(mean_vec)));
    auto var_c = b.create<mlir::arith::ConstantOp>(loc, paramTy,
        mlir::DenseElementsAttr::get(paramTy, llvm::ArrayRef<float>(var_vec)));

    auto empty = b.create<mlir::tensor::EmptyOp>(loc, dims, f32);

    auto idMap = mlir::AffineMap::getMultiDimIdentityMap(/*dimCount=*/4, &ctx);
    auto cMap = mlir::AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0,
                                     mlir::ArrayRef<mlir::AffineExpr>{mlir::getAffineDimExpr(1, &ctx)},
                                     &ctx);

    llvm::SmallVector<mlir::utils::IteratorType> iterators(4, mlir::utils::IteratorType::parallel);

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        outputTy,
        mlir::ValueRange{func.getArgument(0), gamma_c, beta_c, mean_c, var_c},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{idMap, cMap, cMap, cMap, cMap, idMap},
        mlir::ArrayRef<mlir::utils::IteratorType>(iterators),
        [&](mlir::OpBuilder& bodyBuilder, mlir::Location loc, mlir::ValueRange args) {
            auto x = args[0];
            auto gamma = args[1];
            auto beta = args[2];
            auto mean = args[3];
            auto var = args[4];

            auto eps_c = bodyBuilder.create<mlir::arith::ConstantOp>(loc, bodyBuilder.getF32FloatAttr(eps));
            auto x_centered = bodyBuilder.create<mlir::arith::SubFOp>(loc, x, mean);
            auto var_eps = bodyBuilder.create<mlir::arith::AddFOp>(loc, var, eps_c);
            auto stddev = bodyBuilder.create<mlir::math::SqrtOp>(loc, var_eps);
            auto norm = bodyBuilder.create<mlir::arith::DivFOp>(loc, x_centered, stddev);
            auto scaled = bodyBuilder.create<mlir::arith::MulFOp>(loc, gamma, norm);
            auto y = bodyBuilder.create<mlir::arith::AddFOp>(loc, scaled, beta);
            bodyBuilder.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange{y.getResult()});
        });

    b.create<mlir::func::ReturnOp>(loc, generic.getResults());
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
