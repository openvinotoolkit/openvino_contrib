// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "openvino/core/model.hpp"
#include "openvino/op/range.hpp"

namespace ov {
namespace metal_plugin {

namespace {

mlir::ModuleOp build_range_stub(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    auto& results = model->get_results();
    auto out_shape = results[0]->get_shape();
    auto et = results[0]->get_element_type();

    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());
    mlir::Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = mlir::Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = mlir::Float32Type::get(&ctx); break;
        case ov::element::i32: elem_ty = mlir::IntegerType::get(&ctx, 32); break;
        case ov::element::i64: elem_ty = mlir::IntegerType::get(&ctx, 64); break;
        default: elem_ty = mlir::Float32Type::get(&ctx); break;
    }

    auto out_ty = mlir::RankedTensorType::get(out_dims, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "range_main",
                                              mb.getFunctionType({}, {out_ty}));
    func.addEntryBlock();
    mb.setInsertionPointToStart(&func.getBody().front());
    mb.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), mlir::ValueRange{});
    return module;
}

}  // namespace

mlir::ModuleOp build_mlir_range_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_range_stub(model, ctx);
}

}  // namespace metal_plugin
}  // namespace ov

