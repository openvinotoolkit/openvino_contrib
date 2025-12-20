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
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace metal_plugin {

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

    // Fetch parameter constants; caller must ensure they are constants in the MLIR path.
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
        mlir::ArrayRef<mlir::utils::IteratorType>(iterators));
    {
        // One block argument per input and per output tensor.
        auto& region = generic.getRegion();
        auto* block = new mlir::Block();
        block->addArguments({f32, f32, f32, f32, f32, f32},
                            {loc, loc, loc, loc, loc, loc});
        region.push_back(block);
        mlir::OpBuilder body(block, block->begin());
        auto x = block->getArgument(0);
        auto gamma = block->getArgument(1);
        auto beta = block->getArgument(2);
        auto mean = block->getArgument(3);
        auto var = block->getArgument(4);

        auto eps_c = body.create<mlir::arith::ConstantOp>(loc, body.getF32FloatAttr(eps));
        auto x_centered = body.create<mlir::arith::SubFOp>(loc, x, mean);
        auto var_eps = body.create<mlir::arith::AddFOp>(loc, var, eps_c);
        auto stddev = body.create<mlir::math::SqrtOp>(loc, var_eps);
        auto norm = body.create<mlir::arith::DivFOp>(loc, x_centered, stddev);
        auto scaled = body.create<mlir::arith::MulFOp>(loc, gamma, norm);
        auto y = body.create<mlir::arith::AddFOp>(loc, scaled, beta);
        body.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange{y.getResult()});
    }

    b.create<mlir::func::ReturnOp>(loc, generic.getResults());
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
