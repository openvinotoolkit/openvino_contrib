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

#include "mlir/gfx_mlir_type_utils.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace gfx_plugin {

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

    const auto in_pshape = shapeof->get_input_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static(), "ShapeOf MLIR builder: input rank must be static");
    auto out_elem_ty = to_mlir_type(shapeof->get_output_element_type(0),
                                    ctx,
                                    /*fallback_f32=*/false,
                                    /*allow_unsigned=*/false,
                                    /*allow_small_ints=*/false,
                                    /*allow_bf16=*/false,
                                    /*allow_boolean=*/false,
                                    /*signless_integers=*/true);
    const auto in_shape_vec = to_shape(in_pshape);
    const auto rank = static_cast<int64_t>(in_pshape.rank().get_length());
    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape_vec,
                                                    to_mlir_type(shapeof->get_input_element_type(0), ctx,
                                                                 /*fallback_f32=*/true));
    auto out_tensor_ty = mlir::RankedTensorType::get({rank}, out_elem_ty);
    auto shape_tensor_ty = mlir::RankedTensorType::get({rank}, out_elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty, shape_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "shapeof_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    (void)loc;
    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), func.getArgument(1));
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
