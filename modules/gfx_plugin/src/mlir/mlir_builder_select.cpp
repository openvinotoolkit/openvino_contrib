// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/select.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_select_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect>();

    OPENVINO_ASSERT(model->get_parameters().size() == 3, "Select model must have 3 parameters");
    const auto out_shape = model->get_results()[0]->get_shape();
    const auto out_elems = static_cast<int64_t>(ov::shape_size(out_shape));
    const mlir::SmallVector<int64_t> flat_dims = {out_elems == 0 ? 1 : out_elems};

    mlir::OpBuilder module_builder(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    module_builder.setInsertionPointToStart(module.getBody());

    auto cond_ty = mlir::RankedTensorType::get(flat_dims, mlir::IntegerType::get(&ctx, 1));
    auto to_elem_ty = [&](ov::element::Type et) -> mlir::Type {
        switch (et) {
            case ov::element::f16: return mlir::Float16Type::get(&ctx);
            case ov::element::f32: return mlir::Float32Type::get(&ctx);
            default: return mlir::Float32Type::get(&ctx);
        }
    };
    auto elem_ty = to_elem_ty(model->get_results()[0]->get_output_element_type(0));
    auto data_ty = mlir::RankedTensorType::get(flat_dims, elem_ty);

    auto func_type = module_builder.getFunctionType({cond_ty, data_ty, data_ty}, {data_ty});
    auto func = module_builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "select_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());

    auto empty = b.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&ctx),
                                                 flat_dims,
                                                 elem_ty);

    // Single parallel loop over flattened tensor.
    auto map = mlir::AffineMap::getMultiDimIdentityMap(1, &ctx);
    llvm::SmallVector<mlir::utils::IteratorType> iterators(1, mlir::utils::IteratorType::parallel);

    auto generic = b.create<mlir::linalg::GenericOp>(
        mlir::UnknownLoc::get(&ctx),
        data_ty,
        mlir::ValueRange{func.getArgument(0), func.getArgument(1), func.getArgument(2)},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{map, map, map, map},
        mlir::ArrayRef<mlir::utils::IteratorType>(iterators));
    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({cond_ty.getElementType(), data_ty.getElementType(), data_ty.getElementType(), data_ty.getElementType()},
                            {mlir::UnknownLoc::get(&ctx), mlir::UnknownLoc::get(&ctx),
                             mlir::UnknownLoc::get(&ctx), mlir::UnknownLoc::get(&ctx)});
        mlir::OpBuilder body(block, block->begin());
        auto cond = block->getArgument(0);
        auto tval = block->getArgument(1);
        auto fval = block->getArgument(2);
        auto sel = body.create<mlir::arith::SelectOp>(mlir::UnknownLoc::get(&ctx), cond, tval, fval);
        body.create<mlir::linalg::YieldOp>(mlir::UnknownLoc::get(&ctx), mlir::ValueRange{sel});
    }

    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), generic.getResults());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
