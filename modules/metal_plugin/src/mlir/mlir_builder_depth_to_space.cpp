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
#include "openvino/op/depth_to_space.hpp"

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

mlir::ModuleOp build_mlir_depth_to_space_from_model(const std::shared_ptr<const ov::Model>& model,
                                                    mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

    std::shared_ptr<const ov::op::v0::DepthToSpace> d2s;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto op = ov::as_type_ptr<const ov::op::v0::DepthToSpace>(node)) {
            OPENVINO_ASSERT(!d2s, "DepthToSpace MLIR builder: expected single DepthToSpace");
            d2s = op;
        }
    }
    OPENVINO_ASSERT(d2s, "DepthToSpace MLIR builder: DepthToSpace op not found");

    const auto in_shape = d2s->get_input_shape(0);
    const auto out_shape = d2s->get_output_shape(0);
    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());

    auto in_ty = to_mlir_type(d2s->get_input_element_type(0), ctx);
    auto out_ty = to_mlir_type(d2s->get_output_element_type(0), ctx);

    auto in_tensor_ty = mlir::RankedTensorType::get(in_dims, in_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_dims, out_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "depth_to_space_main", func_type);
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
