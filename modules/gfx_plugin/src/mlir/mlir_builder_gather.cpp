// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_gather_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::Node> gather_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v1::Gather>(node) ||
            ov::as_type_ptr<const ov::op::v7::Gather>(node) ||
            ov::as_type_ptr<const ov::op::v8::Gather>(node)) {
            OPENVINO_ASSERT(!gather_node, "Gather MLIR builder: expected single Gather");
            gather_node = node;
        }
    }
    OPENVINO_ASSERT(gather_node, "Gather MLIR builder: Gather op not found");

    int64_t batch_dims = 0;
    if (auto gather_v7 = ov::as_type_ptr<const ov::op::v7::Gather>(gather_node)) {
        batch_dims = gather_v7->get_batch_dims();
    } else if (auto gather_v8 = ov::as_type_ptr<const ov::op::v8::Gather>(gather_node)) {
        batch_dims = gather_v8->get_batch_dims();
    }
    OPENVINO_ASSERT(batch_dims == 0, "Gather MLIR: batch_dims not supported");

    const auto in_pshape = gather_node->get_input_partial_shape(0);
    const auto idx_pshape = gather_node->get_input_partial_shape(1);
    const auto out_pshape = gather_node->get_output_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && idx_pshape.rank().is_static() && out_pshape.rank().is_static(),
                    "Gather MLIR: ranks must be static");

    auto in_shape = to_shape(in_pshape);
    auto idx_shape = to_shape(idx_pshape);
    auto out_shape = to_shape(out_pshape);

    auto elem_ty = to_mlir_type(gather_node->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true,
                                /*allow_bf16=*/false,
                                /*allow_boolean=*/true);
    auto idx_ty = to_mlir_type(gather_node->get_input_element_type(1),
                               ctx,
                               /*fallback_f32=*/true,
                               /*allow_unsigned=*/true,
                               /*allow_small_ints=*/true,
                               /*allow_bf16=*/false,
                               /*allow_boolean=*/false,
                               /*signless_integers=*/true);

    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape, elem_ty);
    auto idx_tensor_ty = mlir::RankedTensorType::get(idx_shape, idx_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_shape, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty, idx_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "gather_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    const auto rank_data = static_cast<size_t>(in_pshape.rank().get_length());
    const auto rank_idx = static_cast<size_t>(idx_pshape.rank().get_length());
    const auto rank_out = static_cast<size_t>(out_pshape.rank().get_length());
    OPENVINO_ASSERT(rank_data > 0, "Gather MLIR: input rank must be positive");

    auto axis_c = ov::as_type_ptr<const ov::op::v0::Constant>(gather_node->get_input_node_shared_ptr(2));
    OPENVINO_ASSERT(axis_c, "Gather MLIR: axis must be constant");
    auto axis_v = axis_c->cast_vector<int64_t>();
    OPENVINO_ASSERT(axis_v.size() == 1, "Gather MLIR: axis must be scalar");
    int64_t axis = axis_v[0];
    if (axis < 0)
        axis += static_cast<int64_t>(rank_data);
    OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank_data, "Gather MLIR: axis out of range");

    llvm::SmallVector<mlir::Value> out_dyn_dims;
    out_dyn_dims.reserve(rank_out);
    for (size_t i = 0; i < rank_out; ++i) {
        if (out_shape[i] != mlir::ShapedType::kDynamic) {
            continue;
        }
        if (i < static_cast<size_t>(axis)) {
            out_dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(0), static_cast<int64_t>(i)).getResult());
        } else if (i < static_cast<size_t>(axis) + rank_idx) {
            out_dyn_dims.push_back(
                b.create<mlir::tensor::DimOp>(loc, func.getArgument(1), static_cast<int64_t>(i - static_cast<size_t>(axis)))
                    .getResult());
        } else {
            out_dyn_dims.push_back(
                b.create<mlir::tensor::DimOp>(loc,
                                              func.getArgument(0),
                                              static_cast<int64_t>(i - rank_idx + 1))
                    .getResult());
        }
    }

    auto generated = mlir::tensor::GenerateOp::create(
        b,
        loc,
        out_tensor_ty,
        out_dyn_dims,
        [&](mlir::OpBuilder& gb, mlir::Location gen_loc, mlir::ValueRange out_indices) {
            llvm::SmallVector<mlir::Value> idx_indices;
            idx_indices.reserve(rank_idx);
            for (size_t i = 0; i < rank_idx; ++i) {
                idx_indices.push_back(out_indices[static_cast<size_t>(axis) + i]);
            }

            auto idx_val = gb.create<mlir::tensor::ExtractOp>(gen_loc, func.getArgument(1), idx_indices).getResult();
            mlir::Value idx_i64 = idx_val;
            if (idx_val.getType().isInteger(32)) {
                idx_i64 = gb.create<mlir::arith::ExtSIOp>(gen_loc, mlir::IntegerType::get(&ctx, 64), idx_val).getResult();
            } else if (!idx_val.getType().isInteger(64)) {
                OPENVINO_THROW("Gather MLIR: only i32/i64 indices are supported");
            }

            auto axis_dim = gb.create<mlir::tensor::DimOp>(gen_loc, func.getArgument(0), axis).getResult();
            auto axis_dim_i64 = gb.create<mlir::arith::IndexCastOp>(gen_loc, mlir::IntegerType::get(&ctx, 64), axis_dim)
                                    .getResult();
            auto zero_i64 = gb.create<mlir::arith::ConstantIntOp>(gen_loc, 0, 64).getResult();
            auto one_i64 = gb.create<mlir::arith::ConstantIntOp>(gen_loc, 1, 64).getResult();
            auto max_i64 = gb.create<mlir::arith::SubIOp>(gen_loc, axis_dim_i64, one_i64).getResult();

            auto neg_pred =
                gb.create<mlir::arith::CmpIOp>(gen_loc, mlir::arith::CmpIPredicate::slt, idx_i64, zero_i64).getResult();
            auto idx_plus = gb.create<mlir::arith::AddIOp>(gen_loc, idx_i64, axis_dim_i64).getResult();
            auto idx_fixed = gb.create<mlir::arith::SelectOp>(gen_loc, neg_pred, idx_plus, idx_i64).getResult();

            auto lt0 = gb.create<mlir::arith::CmpIOp>(gen_loc,
                                                      mlir::arith::CmpIPredicate::slt,
                                                      idx_fixed,
                                                      zero_i64)
                           .getResult();
            auto gtmax = gb.create<mlir::arith::CmpIOp>(gen_loc,
                                                        mlir::arith::CmpIPredicate::sgt,
                                                        idx_fixed,
                                                        max_i64)
                             .getResult();
            auto idx_clamped = gb.create<mlir::arith::SelectOp>(gen_loc, lt0, zero_i64, idx_fixed).getResult();
            idx_clamped = gb.create<mlir::arith::SelectOp>(gen_loc, gtmax, max_i64, idx_clamped).getResult();
            auto idx_index = gb.create<mlir::arith::IndexCastOp>(gen_loc, gb.getIndexType(), idx_clamped).getResult();

            llvm::SmallVector<mlir::Value> data_indices;
            data_indices.reserve(rank_data);
            for (size_t i = 0; i < rank_data; ++i) {
                if (i < static_cast<size_t>(axis)) {
                    data_indices.push_back(out_indices[i]);
                } else if (i == static_cast<size_t>(axis)) {
                    data_indices.push_back(idx_index);
                } else {
                    data_indices.push_back(out_indices[i + rank_idx - 1]);
                }
            }

            auto data_val = gb.create<mlir::tensor::ExtractOp>(gen_loc, func.getArgument(0), data_indices).getResult();
            mlir::tensor::YieldOp::create(gb, gen_loc, data_val);
        });

    b.create<mlir::func::ReturnOp>(loc, generated.getResult());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
