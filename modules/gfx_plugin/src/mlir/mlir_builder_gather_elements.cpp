// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/gather_elements.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_gather_elements_from_model(const std::shared_ptr<const ov::Model>& model,
                                                     mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

    std::shared_ptr<const ov::op::v6::GatherElements> gather;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto g = ov::as_type_ptr<const ov::op::v6::GatherElements>(node)) {
            OPENVINO_ASSERT(!gather, "GatherElements MLIR builder: expected single GatherElements");
            gather = g;
        }
    }
    OPENVINO_ASSERT(gather, "GatherElements MLIR builder: GatherElements op not found");

    const auto data_shape = gather->get_input_shape(0);
    const auto idx_shape = gather->get_input_shape(1);
    const auto out_shape = gather->get_output_shape(0);

    mlir::SmallVector<int64_t> data_dims(data_shape.begin(), data_shape.end());
    mlir::SmallVector<int64_t> idx_dims(idx_shape.begin(), idx_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());

    auto data_ty = to_mlir_type(gather->get_input_element_type(0), ctx, /*fallback_f32=*/true);
    auto idx_ty = to_mlir_type(gather->get_input_element_type(1), ctx, /*fallback_f32=*/true);
    auto out_ty = to_mlir_type(gather->get_output_element_type(0), ctx, /*fallback_f32=*/true);

    auto data_tensor_ty = mlir::RankedTensorType::get(data_dims, data_ty);
    auto idx_tensor_ty = mlir::RankedTensorType::get(idx_dims, idx_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_dims, out_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({data_tensor_ty, idx_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "gather_elements_main", func_type);
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
