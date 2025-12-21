// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
mlir::Type to_mlir_type(ov::element::Type et, mlir::MLIRContext& ctx, bool fallback_f32 = false) {
    switch (et) {
        case ov::element::i32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        case ov::element::i64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        default:
            if (fallback_f32) return mlir::Float32Type::get(&ctx);
            OPENVINO_THROW("ShapeOf MLIR: unsupported element type");
    }
}
}  // namespace

mlir::ModuleOp build_mlir_shapeof_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::Node> shapeof;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v0::ShapeOf>(node) ||
            ov::as_type_ptr<const ov::op::v3::ShapeOf>(node)) {
            OPENVINO_ASSERT(!shapeof, "ShapeOf MLIR builder: expected single ShapeOf");
            shapeof = node;
        }
    }
    OPENVINO_ASSERT(shapeof, "ShapeOf MLIR builder: ShapeOf op not found");

    const auto in_shape = shapeof->get_input_shape(0);
    auto out_elem_ty = to_mlir_type(shapeof->get_output_element_type(0), ctx);

    mlir::SmallVector<int64_t> in_shape_vec;
    in_shape_vec.reserve(in_shape.size());
    for (auto v : in_shape) in_shape_vec.push_back(static_cast<int64_t>(v));
    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape_vec,
                                                    to_mlir_type(shapeof->get_input_element_type(0), ctx,
                                                                 /*fallback_f32=*/true));
    auto out_tensor_ty =
        mlir::RankedTensorType::get({static_cast<int64_t>(in_shape.size())}, out_elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "shapeof_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    llvm::SmallVector<mlir::Value> dims;
    dims.reserve(in_shape.size());
    auto out_int_ty = mlir::cast<mlir::IntegerType>(out_elem_ty);
    for (size_t i = 0; i < in_shape.size(); ++i) {
        auto dim_idx = b.create<mlir::tensor::DimOp>(loc, func.getArgument(0), i).getResult();
        auto dim_val = b.create<mlir::arith::IndexCastOp>(loc, out_int_ty, dim_idx).getResult();
        dims.push_back(dim_val);
    }
    auto from = b.create<mlir::tensor::FromElementsOp>(loc, out_tensor_ty, dims);
    b.create<mlir::func::ReturnOp>(loc, from.getResult());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
