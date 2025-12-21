// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/scatter_nd_update.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
mlir::Type to_mlir_type(ov::element::Type et, mlir::MLIRContext& ctx) {
    switch (et) {
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        case ov::element::i32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        case ov::element::i64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
        default: return mlir::Float32Type::get(&ctx);
    }
}
}  // namespace

mlir::ModuleOp build_mlir_scatter_nd_update_from_model(const std::shared_ptr<const ov::Model>& model,
                                                       mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

    std::shared_ptr<const ov::Node> scatter;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v3::ScatterNDUpdate>(node) ||
            ov::as_type_ptr<const ov::op::v15::ScatterNDUpdate>(node)) {
            OPENVINO_ASSERT(!scatter, "ScatterNDUpdate MLIR builder: expected single op");
            scatter = node;
        }
    }
    OPENVINO_ASSERT(scatter, "ScatterNDUpdate MLIR builder: op not found");

    const auto data_shape = scatter->get_input_shape(0);
    const auto idx_shape = scatter->get_input_shape(1);
    const auto upd_shape = scatter->get_input_shape(2);
    const auto out_shape = scatter->get_output_shape(0);

    mlir::SmallVector<int64_t> data_dims(data_shape.begin(), data_shape.end());
    mlir::SmallVector<int64_t> idx_dims(idx_shape.begin(), idx_shape.end());
    mlir::SmallVector<int64_t> upd_dims(upd_shape.begin(), upd_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());

    auto data_ty = to_mlir_type(scatter->get_input_element_type(0), ctx);
    auto idx_ty = to_mlir_type(scatter->get_input_element_type(1), ctx);
    auto upd_ty = to_mlir_type(scatter->get_input_element_type(2), ctx);
    auto out_ty = to_mlir_type(scatter->get_output_element_type(0), ctx);

    auto data_tensor_ty = mlir::RankedTensorType::get(data_dims, data_ty);
    auto idx_tensor_ty = mlir::RankedTensorType::get(idx_dims, idx_ty);
    auto upd_tensor_ty = mlir::RankedTensorType::get(upd_dims, upd_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_dims, out_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({data_tensor_ty, idx_tensor_ty, upd_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "scatter_nd_update_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto out = b.create<mlir::tensor::EmptyOp>(loc, out_dims, out_ty);
    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{out.getResult()});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
