// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_concat_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::op::v0::Concat> concat;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v0::Concat>(node)) {
            OPENVINO_ASSERT(!concat, "Concat MLIR builder: expected single Concat");
            concat = c;
        }
    }
    OPENVINO_ASSERT(concat, "Concat MLIR builder: Concat op not found");

    const size_t rank = concat->get_output_shape(0).size();
    OPENVINO_ASSERT(rank > 0, "Concat MLIR: output rank must be static");
    const ov::Shape out_shape = concat->get_output_shape(0);

    auto elem_ty = to_mlir_type(concat->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/false,
                                /*allow_small_ints=*/true);

    mlir::SmallVector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto out_tensor_ty = mlir::RankedTensorType::get(out_shape_vec, elem_ty);

    mlir::SmallVector<mlir::Type> input_types;
    input_types.reserve(concat->get_input_size());
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        auto in_shape = to_shape(concat->get_input_partial_shape(i));
        input_types.push_back(mlir::RankedTensorType::get(in_shape, elem_ty));
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType(input_types, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "concat_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    int64_t axis = concat->get_axis();
    if (axis < 0) axis += static_cast<int64_t>(rank);
    OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank, "Concat MLIR: axis out of range");

    mlir::Value result = b.create<mlir::tensor::EmptyOp>(loc, out_shape_vec, elem_ty);

    int64_t axis_offset = 0;
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        auto in_shape = concat->get_input_shape(i);
        mlir::Value src = func.getArgument(static_cast<unsigned>(i));
        mlir::SmallVector<mlir::OpFoldResult> offsets;
        mlir::SmallVector<mlir::OpFoldResult> sizes;
        mlir::SmallVector<mlir::OpFoldResult> strides;
        offsets.reserve(rank);
        sizes.reserve(rank);
        strides.reserve(rank);
        for (size_t dim = 0; dim < rank; ++dim) {
            const int64_t offset = static_cast<int64_t>(dim) == axis ? axis_offset : 0;
            offsets.push_back(b.getIndexAttr(offset));
            sizes.push_back(b.getIndexAttr(static_cast<int64_t>(in_shape[dim])));
            strides.push_back(b.getIndexAttr(1));
        }
        result = b.create<mlir::tensor::InsertSliceOp>(loc, src, result, offsets, sizes, strides).getResult();
        axis_offset += static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
    }

    b.create<mlir::func::ReturnOp>(loc, result);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
