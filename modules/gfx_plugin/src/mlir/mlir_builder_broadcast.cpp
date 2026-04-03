// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

std::vector<int64_t> get_const_axes(const std::shared_ptr<const ov::Node>& node) {
    auto c = ov::as_type_ptr<const ov::op::v0::Constant>(node);
    OPENVINO_ASSERT(c, "Broadcast MLIR: axes_mapping must be Constant");
    return c->cast_vector<int64_t>();
}

}  // namespace

mlir::ModuleOp build_mlir_broadcast_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::Node> bc_node;
    std::shared_ptr<const ov::op::v3::Broadcast> bc_v3;
    std::shared_ptr<const ov::op::v1::Broadcast> bc_v1;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto b = ov::as_type_ptr<const ov::op::v3::Broadcast>(node)) {
            OPENVINO_ASSERT(!bc_node, "Broadcast MLIR: expected single Broadcast op");
            bc_node = node;
            bc_v3 = b;
            break;
        }
        if (auto b1 = ov::as_type_ptr<const ov::op::v1::Broadcast>(node)) {
            OPENVINO_ASSERT(!bc_node, "Broadcast MLIR: expected single Broadcast op");
            bc_node = node;
            bc_v1 = b1;
            break;
        }
    }
    OPENVINO_ASSERT(bc_node, "Broadcast MLIR: Broadcast op not found");

    const auto in_pshape = bc_node->get_input_partial_shape(0);
    const auto out_pshape = bc_node->get_output_partial_shape(0);
    OPENVINO_ASSERT(out_pshape.is_static(), "Broadcast MLIR: requires static output shape");
    if (bc_node->get_input_size() > 1) {
        auto shape_const = ov::as_type_ptr<const ov::op::v0::Constant>(bc_node->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(shape_const, "Broadcast MLIR: target_shape must be Constant");
    }
    const auto in_shape = to_shape(in_pshape);
    const auto out_shape = to_shape(out_pshape);

    const size_t in_rank = in_shape.size();
    const size_t out_rank = out_shape.size();
    OPENVINO_ASSERT(in_rank <= out_rank, "Broadcast MLIR: input rank must be <= output rank");

    auto elem_ty = to_mlir_type(bc_node->get_output_element_type(0), ctx, /*fallback_f32=*/false,
                                /*allow_unsigned=*/true);
    auto in_ty = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto out_ty = mlir::RankedTensorType::get(out_shape, elem_ty);

    std::vector<int64_t> axes_mapping;
    bool explicit_axes = false;
    if (bc_node->get_input_size() > 2) {
        axes_mapping = get_const_axes(bc_node->get_input_node_shared_ptr(2));
        explicit_axes = true;
    } else if (bc_v3) {
        auto spec = bc_v3->get_broadcast_spec();
        explicit_axes = (spec.m_type == ov::op::BroadcastType::EXPLICIT);
    } else if (bc_v1) {
        auto spec = bc_v1->get_broadcast_spec();
        explicit_axes = (spec.m_type == ov::op::AutoBroadcastType::NONE);
    }

    if (explicit_axes) {
        OPENVINO_ASSERT(axes_mapping.size() == in_rank,
                        "Broadcast MLIR: axes_mapping size mismatch");
    } else {
        axes_mapping.resize(in_rank);
        for (size_t i = 0; i < in_rank; ++i) {
            axes_mapping[i] = static_cast<int64_t>(out_rank - in_rank + i);
        }
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_ty}, {out_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "broadcast_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto empty = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);

    llvm::SmallVector<mlir::utils::IteratorType> iters(out_rank, mlir::utils::IteratorType::parallel);
    llvm::SmallVector<mlir::AffineExpr> in_exprs;
    in_exprs.reserve(in_rank);
    for (size_t i = 0; i < in_rank; ++i) {
        int64_t axis = axes_mapping[i];
        OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < out_rank,
                        "Broadcast MLIR: axes_mapping out of range");
        if (in_shape[i] == 1) {
            in_exprs.push_back(mlir::getAffineConstantExpr(0, &ctx));
        } else {
            in_exprs.push_back(mlir::getAffineDimExpr(static_cast<unsigned>(axis), &ctx));
        }
    }
    auto map_in = mlir::AffineMap::get(static_cast<unsigned>(out_rank), 0, in_exprs, &ctx);
    auto map_out = mlir::AffineMap::getMultiDimIdentityMap(static_cast<unsigned>(out_rank), &ctx);

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        out_ty,
        mlir::ValueRange{func.getArgument(0)},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{map_in, map_out},
        mlir::ArrayRef<mlir::utils::IteratorType>(iters));
    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({elem_ty, elem_ty}, {loc, loc});
        mlir::OpBuilder body(block, block->begin());
        body.create<mlir::linalg::YieldOp>(loc, block->getArgument(0));
    }

    b.create<mlir::func::ReturnOp>(loc, generic.getResults());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
