// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Casting.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace metal_plugin {

inline mlir::Type to_mlir_type(ov::element::Type et, mlir::MLIRContext& ctx) {
    switch (et) {
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        case ov::element::i32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        default: OPENVINO_THROW("Eltwise MLIR: unsupported element type");
    }
}

template <class NodeT, class Emitter>
mlir::ModuleOp build_mlir_binary_eltwise_from_model(const std::shared_ptr<const ov::Model>& model,
                                                    mlir::MLIRContext& ctx,
                                                    Emitter&& emit) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();

    std::shared_ptr<const NodeT> node;
    for (const auto& n : model->get_ordered_ops()) {
        if (auto ptr = ov::as_type_ptr<const NodeT>(n)) {
            OPENVINO_ASSERT(!node, "Eltwise MLIR builder: expected single node of requested type");
            node = ptr;
        }
    }
    OPENVINO_ASSERT(node, "Eltwise MLIR builder: node not found");

    const auto pshape0 = node->get_input_partial_shape(0);
    const auto pshape1 = node->get_input_partial_shape(1);
    const auto pout = node->get_output_partial_shape(0);
    const size_t rank = pout.rank().get_length();

    auto elem_ty = to_mlir_type(node->get_output_element_type(0), ctx);

    auto to_shape = [](const ov::PartialShape& ps) {
        mlir::SmallVector<int64_t> dims;
        for (const auto& d : ps) dims.push_back(d.is_dynamic() ? mlir::ShapedType::kDynamic
                                                              : static_cast<int64_t>(d.get_length()));
        return dims;
    };
    auto s0 = to_shape(pshape0);
    auto s1 = to_shape(pshape1);
    auto sout = to_shape(pout);

    auto ty0 = mlir::RankedTensorType::get(s0, elem_ty);
    auto ty1 = mlir::RankedTensorType::get(s1, elem_ty);
    auto ty_out = mlir::RankedTensorType::get(sout, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({ty0, ty1}, {ty_out});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "eltwise_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);

    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);

    llvm::SmallVector<mlir::Value> out_dyn;
    out_dyn.reserve(sout.size());
    auto get_dim = [&](mlir::Value t, const mlir::SmallVector<int64_t>& shp, size_t out_idx) -> mlir::Value {
        const size_t tr = shp.size();
        if (out_idx + tr < rank) return mlir::Value{};
        size_t axis = out_idx - (rank - tr);
        int64_t dim = shp[axis];
        if (dim == 1) return mlir::Value{};
        return b.create<mlir::tensor::DimOp>(loc, t, axis).getResult();
    };
    for (size_t i = 0; i < rank; ++i) {
        if (sout[i] != mlir::ShapedType::kDynamic) continue;
        mlir::Value d = get_dim(func.getArgument(0), s0, i);
        if (!d) d = get_dim(func.getArgument(1), s1, i);
        if (!d) d = c1;
        out_dyn.push_back(d);
    }

    auto empty = b.create<mlir::tensor::EmptyOp>(loc, sout, elem_ty, out_dyn);

    llvm::SmallVector<mlir::utils::IteratorType> iters(rank, mlir::utils::IteratorType::parallel);
    auto make_map = [&](const mlir::SmallVector<int64_t>& in_shape) {
        size_t in_rank = in_shape.size();
        llvm::SmallVector<mlir::AffineExpr> exprs;
        size_t start = rank - in_rank;
        for (size_t i = 0; i < in_rank; ++i) {
            if (in_shape[i] == 1) exprs.push_back(mlir::getAffineConstantExpr(0, &ctx));
            else exprs.push_back(mlir::getAffineDimExpr(start + i, &ctx));
        }
        return mlir::AffineMap::get(rank, 0, exprs, &ctx);
    };
    auto map0 = make_map(s0);
    auto map1 = make_map(s1);
    auto map_out = mlir::AffineMap::getMultiDimIdentityMap(rank, &ctx);

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        ty_out,
        mlir::ValueRange{func.getArgument(0), func.getArgument(1)},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{map0, map1, map_out},
        mlir::ArrayRef<mlir::utils::IteratorType>(iters));
    {
        // linalg.generic requires one block argument per input and per output (for init tensor).
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty, elem_ty}, {loc, loc, loc});
        mlir::OpBuilder body(block, block->begin());
        mlir::Value res = emit(body, loc, block->getArguments().take_front(2), elem_ty, node);
        body.create<mlir::linalg::YieldOp>(loc, res);
    }

    b.create<mlir::func::ReturnOp>(loc, generic.getResults());
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
