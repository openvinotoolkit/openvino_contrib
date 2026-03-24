// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

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
#include "llvm/Support/Casting.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace gfx_plugin {

inline void set_binary_eltwise_input_transform_attrs(mlir::ModuleOp module,
                                                     const std::vector<MlirInputTransformDesc>& input_transforms) {
    if (!module) {
        return;
    }
    auto* ctx = module.getContext();
    mlir::OpBuilder b(ctx);
    for (size_t input_idx = 0; input_idx < input_transforms.size(); ++input_idx) {
        const auto& transform = input_transforms[input_idx];
        if (!transform.has_transpose()) {
            continue;
        }
        llvm::SmallVector<mlir::Attribute> attrs;
        attrs.reserve(transform.transpose_permutation.size());
        for (int64_t axis : transform.transpose_permutation) {
            attrs.push_back(b.getI64IntegerAttr(axis));
        }
        const std::string attr_name = "gfx.absorbed_input" + std::to_string(input_idx) + "_perm";
        module->setAttr(attr_name, b.getArrayAttr(attrs));
    }
}

template <class NodeT, class Emitter>
mlir::ModuleOp build_mlir_binary_eltwise_from_node(const std::shared_ptr<const NodeT>& node,
                                                   mlir::MLIRContext& ctx,
                                                   const std::vector<MlirInputTransformDesc>& input_transforms,
                                                   Emitter&& emit) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();

    OPENVINO_ASSERT(node, "Eltwise MLIR builder: node is null");

    const auto pshape0 = node->get_input_partial_shape(0);
    const auto pshape1 = node->get_input_partial_shape(1);
    const auto pout = node->get_output_partial_shape(0);
    const size_t rank = pout.rank().get_length();

    auto elem_ty = to_mlir_type(node->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true,
                                /*allow_bf16=*/true,
                                /*allow_boolean=*/true,
                                /*signless_integers=*/true);

    auto to_shape = [](const ov::PartialShape& ps) {
        mlir::SmallVector<int64_t> dims;
        for (const auto& d : ps) dims.push_back(d.is_dynamic() ? mlir::ShapedType::kDynamic
                                                              : static_cast<int64_t>(d.get_length()));
        return dims;
    };
    auto consumer_s0 = to_shape(pshape0);
    auto consumer_s1 = to_shape(pshape1);
    auto source_s0 = consumer_s0;
    auto source_s1 = consumer_s1;
    if (input_transforms.size() > 0 && input_transforms[0].has_transpose()) {
        source_s0.assign(input_transforms[0].source_shape.begin(), input_transforms[0].source_shape.end());
    }
    if (input_transforms.size() > 1 && input_transforms[1].has_transpose()) {
        source_s1.assign(input_transforms[1].source_shape.begin(), input_transforms[1].source_shape.end());
    }
    auto sout = to_shape(pout);

    auto ty0 = mlir::RankedTensorType::get(source_s0, elem_ty);
    auto ty1 = mlir::RankedTensorType::get(source_s1, elem_ty);
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
        mlir::Value d = get_dim(func.getArgument(0), source_s0, i);
        if (!d) d = get_dim(func.getArgument(1), source_s1, i);
        if (!d) d = c1;
        out_dyn.push_back(d);
    }

    auto empty = b.create<mlir::tensor::EmptyOp>(loc, sout, elem_ty, out_dyn);

    llvm::SmallVector<mlir::utils::IteratorType> iters(rank, mlir::utils::IteratorType::parallel);
    auto make_map = [&](const mlir::SmallVector<int64_t>& source_shape,
                        const mlir::SmallVector<int64_t>& consumer_shape,
                        const MlirInputTransformDesc* transform) {
        const size_t consumer_rank = consumer_shape.size();
        const size_t source_rank = source_shape.size();
        llvm::SmallVector<mlir::AffineExpr> exprs;
        const size_t start = rank - consumer_rank;
        if (transform && transform->has_transpose()) {
            OPENVINO_ASSERT(consumer_rank == transform->transpose_permutation.size(),
                            "Eltwise MLIR builder: transpose rank mismatch");
            OPENVINO_ASSERT(source_rank == consumer_rank,
                            "Eltwise MLIR builder: absorbed transpose expects rank-preserving layout");
            std::vector<size_t> inverse(consumer_rank, 0);
            std::vector<bool> seen(consumer_rank, false);
            for (size_t axis = 0; axis < consumer_rank; ++axis) {
                const int64_t perm_axis = transform->transpose_permutation[axis];
                OPENVINO_ASSERT(perm_axis >= 0 && static_cast<size_t>(perm_axis) < consumer_rank,
                                "Eltwise MLIR builder: transpose permutation out of range");
                OPENVINO_ASSERT(!seen[static_cast<size_t>(perm_axis)],
                                "Eltwise MLIR builder: transpose permutation must be unique");
                seen[static_cast<size_t>(perm_axis)] = true;
                inverse[static_cast<size_t>(perm_axis)] = axis;
            }
            for (size_t source_axis = 0; source_axis < source_rank; ++source_axis) {
                const size_t consumer_axis = inverse[source_axis];
                const int64_t dim = consumer_shape[consumer_axis];
                if (dim == 1) {
                    exprs.push_back(mlir::getAffineConstantExpr(0, &ctx));
                } else {
                    exprs.push_back(mlir::getAffineDimExpr(start + consumer_axis, &ctx));
                }
            }
            return mlir::AffineMap::get(rank, 0, exprs, &ctx);
        }

        for (size_t i = 0; i < consumer_rank; ++i) {
            if (consumer_shape[i] == 1) exprs.push_back(mlir::getAffineConstantExpr(0, &ctx));
            else exprs.push_back(mlir::getAffineDimExpr(start + i, &ctx));
        }
        return mlir::AffineMap::get(rank, 0, exprs, &ctx);
    };

    const MlirInputTransformDesc* transform0 =
        input_transforms.size() > 0 && input_transforms[0].has_transpose() ? &input_transforms[0] : nullptr;
    const MlirInputTransformDesc* transform1 =
        input_transforms.size() > 1 && input_transforms[1].has_transpose() ? &input_transforms[1] : nullptr;

    auto map0 = make_map(source_s0, consumer_s0, transform0);
    auto map1 = make_map(source_s1, consumer_s1, transform1);
    auto map_out = mlir::AffineMap::getMultiDimIdentityMap(rank, &ctx);

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        ty_out,
        mlir::ValueRange{func.getArgument(0), func.getArgument(1)},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{map0, map1, map_out},
        mlir::ArrayRef<mlir::utils::IteratorType>(iters));
    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty, elem_ty}, {loc, loc, loc});
        mlir::OpBuilder body(block, block->begin());
        mlir::Value res = emit(body, loc, block->getArguments().take_front(2), elem_ty, node);
        body.create<mlir::linalg::YieldOp>(loc, res);
    }

    b.create<mlir::func::ReturnOp>(loc, generic.getResults());
    set_binary_eltwise_input_transform_attrs(module, input_transforms);
    return module;
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
    return build_mlir_binary_eltwise_from_node<NodeT>(node, ctx, {}, std::forward<Emitter>(emit));
}

}  // namespace gfx_plugin
}  // namespace ov
