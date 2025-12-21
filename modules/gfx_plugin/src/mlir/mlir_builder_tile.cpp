// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "openvino/core/model.hpp"
#include "openvino/op/tile.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

mlir::ModuleOp build_tile_stub(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    auto& params = model->get_parameters();
    auto& results = model->get_results();
    auto in_shape = params[0]->get_shape();
    auto out_shape = results[0]->get_shape();
    auto et = params[0]->get_element_type();

    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());
    mlir::Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = mlir::Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = mlir::Float32Type::get(&ctx); break;
        case ov::element::i32: elem_ty = mlir::IntegerType::get(&ctx, 32); break;
        case ov::element::i64: elem_ty = mlir::IntegerType::get(&ctx, 64); break;
        default: elem_ty = mlir::Float32Type::get(&ctx); break;
    }

    auto in_ty = mlir::RankedTensorType::get(in_dims, elem_ty);
    auto out_ty = mlir::RankedTensorType::get(out_dims, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "tile_main",
                                              mb.getFunctionType({in_ty}, {out_ty}));
    func.addEntryBlock();
    mb.setInsertionPointToStart(&func.getBody().front());
    mb.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), func.getArguments());
    return module;
}

}  // namespace

mlir::ModuleOp build_mlir_tile_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_tile_stub(model, ctx);
}

}  // namespace gfx_plugin
}  // namespace ov
