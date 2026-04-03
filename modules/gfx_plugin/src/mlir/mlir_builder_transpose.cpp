// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_transpose_from_model(const std::shared_ptr<const ov::Model>& model,
                                               mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::op::v1::Transpose> tr;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto t = ov::as_type_ptr<const ov::op::v1::Transpose>(node)) {
            OPENVINO_ASSERT(!tr, "Transpose MLIR builder: expected single Transpose");
            tr = t;
        }
    }
    OPENVINO_ASSERT(tr, "Transpose MLIR builder: Transpose op not found");

    auto in_shape = to_shape(tr->get_input_partial_shape(0));
    auto out_shape = to_shape(tr->get_output_partial_shape(0));
    auto elem_ty = to_mlir_type(tr->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/false,
                                /*allow_small_ints=*/true);

    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_shape, elem_ty);

    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(tr->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(perm_const, "Transpose MLIR: perm must be constant");
    auto perm = perm_const->cast_vector<int64_t>();

    const size_t rank = perm.size();
    OPENVINO_ASSERT(rank == in_shape.size() && rank == out_shape.size(), "Transpose MLIR: rank mismatch");

    std::vector<int64_t> inv_perm(rank, 0);
    for (size_t i = 0; i < rank; ++i) {
        int64_t v = perm[i];
        OPENVINO_ASSERT(v >= 0 && static_cast<size_t>(v) < rank, "Transpose MLIR: perm out of range");
        inv_perm[static_cast<size_t>(v)] = static_cast<int64_t>(i);
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "transpose_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());
    auto empty = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);

    llvm::SmallVector<mlir::AffineExpr> in_exprs;
    in_exprs.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
        in_exprs.push_back(mlir::getAffineDimExpr(static_cast<unsigned>(inv_perm[i]), &ctx));
    }
    auto map_in = mlir::AffineMap::get(static_cast<unsigned>(rank), 0, in_exprs, &ctx);
    auto map_out = mlir::AffineMap::getMultiDimIdentityMap(static_cast<unsigned>(rank), &ctx);
    llvm::SmallVector<mlir::utils::IteratorType> iters(rank, mlir::utils::IteratorType::parallel);

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        out_tensor_ty,
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
