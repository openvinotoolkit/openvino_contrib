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
#include "openvino/op/topk.hpp"
#include "openvino/op/util/topk_base.hpp"

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

mlir::ModuleOp build_mlir_topk_from_model(const std::shared_ptr<const ov::Model>& model,
                                          mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

    std::shared_ptr<const ov::op::util::TopKBase> topk;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto t = std::dynamic_pointer_cast<const ov::op::util::TopKBase>(node)) {
            OPENVINO_ASSERT(!topk, "TopK MLIR builder: expected single TopK op");
            topk = t;
        }
    }
    OPENVINO_ASSERT(topk, "TopK MLIR builder: TopK op not found");

    const auto in_shape = topk->get_input_shape(0);
    const auto out0_shape = topk->get_output_shape(0);
    const auto out1_shape = topk->get_output_shape(1);

    auto in_elem_ty = to_mlir_type(topk->get_input_element_type(0), ctx);
    auto out_val_ty = to_mlir_type(topk->get_output_element_type(0), ctx);
    auto out_idx_ty = to_mlir_type(topk->get_output_element_type(1), ctx);

    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out0_dims(out0_shape.begin(), out0_shape.end());
    mlir::SmallVector<int64_t> out1_dims(out1_shape.begin(), out1_shape.end());

    auto in_tensor_ty = mlir::RankedTensorType::get(in_dims, in_elem_ty);
    auto out0_tensor_ty = mlir::RankedTensorType::get(out0_dims, out_val_ty);
    auto out1_tensor_ty = mlir::RankedTensorType::get(out1_dims, out_idx_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out0_tensor_ty, out1_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "topk_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto out0 = b.create<mlir::tensor::EmptyOp>(loc, out0_dims, out_val_ty);
    auto out1 = b.create<mlir::tensor::EmptyOp>(loc, out1_dims, out_idx_ty);
    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{out0.getResult(), out1.getResult()});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
