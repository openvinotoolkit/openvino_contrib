// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/select.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_select_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::op::v1::Select> select;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto s = ov::as_type_ptr<const ov::op::v1::Select>(node)) {
            OPENVINO_ASSERT(!select, "Select MLIR builder: expected single Select");
            select = s;
        }
    }
    OPENVINO_ASSERT(select, "Select MLIR builder: Select op not found");

    const auto out_pshape = select->get_output_partial_shape(0);
    OPENVINO_ASSERT(out_pshape.rank().is_static(), "Select MLIR: output rank must be static");
    const auto out_shape = to_shape(out_pshape);
    const auto out_rank = static_cast<size_t>(out_pshape.rank().get_length());

    mlir::OpBuilder module_builder(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    module_builder.setInsertionPointToStart(module.getBody());

    auto cond_elem_ty = to_mlir_type(select->get_input_element_type(0),
                                     ctx,
                                     /*fallback_f32=*/false,
                                     /*allow_unsigned=*/false,
                                     /*allow_small_ints=*/false,
                                     /*allow_bf16=*/false,
                                     /*allow_boolean=*/true);
    auto elem_ty = to_mlir_type(select->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true,
                                /*allow_bf16=*/false,
                                /*allow_boolean=*/true);
    mlir::SmallVector<mlir::Type> input_types;
    input_types.reserve(3);
    for (size_t i = 0; i < 3; ++i) {
        const auto pshape = select->get_input_partial_shape(i);
        OPENVINO_ASSERT(pshape.rank().is_static(), "Select MLIR: input ranks must be static");
        input_types.push_back(mlir::RankedTensorType::get(to_shape(pshape), i == 0 ? cond_elem_ty : elem_ty));
    }
    auto data_ty = mlir::RankedTensorType::get(out_shape, elem_ty);

    auto func_type = module_builder.getFunctionType(input_types, {data_ty});
    auto func = module_builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "select_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    llvm::SmallVector<mlir::Value> out_dyn_dims;
    out_dyn_dims.reserve(out_rank);
    for (size_t dim = 0; dim < out_rank; ++dim) {
        if (out_shape[dim] != mlir::ShapedType::kDynamic) {
            continue;
        }
        mlir::Value dyn_dim;
        for (size_t input_idx = 0; input_idx < 3 && !dyn_dim; ++input_idx) {
            const auto in_shape = to_shape(select->get_input_partial_shape(input_idx));
            if (in_shape.size() != out_rank || in_shape[dim] != mlir::ShapedType::kDynamic) {
                continue;
            }
            dyn_dim = b.create<mlir::tensor::DimOp>(loc,
                                                    func.getArgument(static_cast<unsigned>(input_idx)),
                                                    static_cast<int64_t>(dim))
                          .getResult();
        }
        OPENVINO_ASSERT(dyn_dim, "Select MLIR: dynamic output dim must map to a runtime input dim");
        out_dyn_dims.push_back(dyn_dim);
    }

    auto map_indices = [&](mlir::OpBuilder& gb,
                           mlir::Location gen_loc,
                           size_t input_idx,
                           mlir::ValueRange out_indices) {
        llvm::SmallVector<mlir::Value> indices;
        const auto in_shape = to_shape(select->get_input_partial_shape(input_idx));
        if (in_shape.empty()) {
            return indices;
        }
        OPENVINO_ASSERT(in_shape.size() <= out_indices.size(), "Select MLIR: input rank must be <= output rank");
        const size_t rank_delta = out_indices.size() - in_shape.size();
        indices.reserve(in_shape.size());
        for (size_t i = 0; i < in_shape.size(); ++i) {
            if (in_shape[i] == 1) {
                indices.push_back(gb.create<mlir::arith::ConstantIndexOp>(gen_loc, 0));
            } else {
                indices.push_back(out_indices[rank_delta + i]);
            }
        }
        return indices;
    };

    auto generated = mlir::tensor::GenerateOp::create(
        b,
        loc,
        data_ty,
        out_dyn_dims,
        [&](mlir::OpBuilder& gb, mlir::Location gen_loc, mlir::ValueRange out_indices) {
            auto cond = gb.create<mlir::tensor::ExtractOp>(gen_loc,
                                                           func.getArgument(0),
                                                           map_indices(gb, gen_loc, 0, out_indices))
                            .getResult();
            auto tval = gb.create<mlir::tensor::ExtractOp>(gen_loc,
                                                           func.getArgument(1),
                                                           map_indices(gb, gen_loc, 1, out_indices))
                            .getResult();
            auto fval = gb.create<mlir::tensor::ExtractOp>(gen_loc,
                                                           func.getArgument(2),
                                                           map_indices(gb, gen_loc, 2, out_indices))
                            .getResult();
            auto sel = gb.create<mlir::arith::SelectOp>(gen_loc, cond, tval, fval).getResult();
            mlir::tensor::YieldOp::create(gb, gen_loc, sel);
        });

    b.create<mlir::func::ReturnOp>(loc, generated.getResult());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
