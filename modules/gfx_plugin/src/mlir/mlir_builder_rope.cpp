// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_rope_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();

    std::shared_ptr<const ov::op::internal::RoPE> rope;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto candidate = ov::as_type_ptr<const ov::op::internal::RoPE>(node)) {
            OPENVINO_ASSERT(!rope, "RoPE MLIR: expected single RoPE op");
            rope = candidate;
        }
    }
    OPENVINO_ASSERT(rope, "RoPE MLIR: RoPE op not found");
    OPENVINO_ASSERT(rope->get_input_size() >= 3 && rope->get_input_size() <= 4,
                    "RoPE MLIR: expected data, cos, sin and optional position input");
    OPENVINO_ASSERT(rope->get_output_size() == 1, "RoPE MLIR: expected one output");
    OPENVINO_ASSERT(rope->get_input_partial_shape(0).rank().is_static() &&
                    rope->get_output_partial_shape(0).rank().is_static(),
                    "RoPE MLIR: input/output ranks must be static");

    llvm::SmallVector<mlir::Type> inputs;
    inputs.reserve(rope->get_input_size());
    for (size_t idx = 0; idx < rope->get_input_size(); ++idx) {
        OPENVINO_ASSERT(rope->get_input_partial_shape(idx).rank().is_static(),
                        "RoPE MLIR: all input ranks must be static");
        auto elem_ty = to_mlir_type(rope->get_input_element_type(idx),
                                    ctx,
                                    /*fallback_f32=*/false,
                                    /*allow_unsigned=*/true,
                                    /*allow_small_ints=*/idx == 3,
                                    /*allow_bf16=*/false,
                                    /*allow_boolean=*/false,
                                    /*signless_integers=*/idx == 3);
        inputs.push_back(mlir::RankedTensorType::get(to_shape(rope->get_input_partial_shape(idx)), elem_ty));
    }

    auto out_elem_ty = to_mlir_type(rope->get_output_element_type(0), ctx);
    const auto out_shape = to_shape(rope->get_output_partial_shape(0));
    auto out_ty = mlir::RankedTensorType::get(out_shape, out_elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType(inputs, {out_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "rope_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());
    auto dyn_dims = materialize_dynamic_dims_from_tensor(b, loc, func.getArgument(0), out_shape);
    auto empty = b.create<mlir::tensor::EmptyOp>(loc, out_shape, out_elem_ty, dyn_dims);
    b.create<mlir::func::ReturnOp>(loc, empty.getResult());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
