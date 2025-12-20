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
#include "openvino/op/space_to_depth.hpp"

namespace ov {
namespace metal_plugin {

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

mlir::ModuleOp build_mlir_space_to_depth_from_model(const std::shared_ptr<const ov::Model>& model,
                                                    mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

    std::shared_ptr<const ov::op::v0::SpaceToDepth> s2d;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto op = ov::as_type_ptr<const ov::op::v0::SpaceToDepth>(node)) {
            OPENVINO_ASSERT(!s2d, "SpaceToDepth MLIR builder: expected single SpaceToDepth");
            s2d = op;
        }
    }
    OPENVINO_ASSERT(s2d, "SpaceToDepth MLIR builder: SpaceToDepth op not found");

    const auto in_shape = s2d->get_input_shape(0);
    const auto out_shape = s2d->get_output_shape(0);
    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());

    auto in_ty = to_mlir_type(s2d->get_input_element_type(0), ctx);
    auto out_ty = to_mlir_type(s2d->get_output_element_type(0), ctx);

    auto in_tensor_ty = mlir::RankedTensorType::get(in_dims, in_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_dims, out_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "space_to_depth_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto out = b.create<mlir::tensor::EmptyOp>(loc, out_dims, out_ty);
    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{out.getResult()});
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
