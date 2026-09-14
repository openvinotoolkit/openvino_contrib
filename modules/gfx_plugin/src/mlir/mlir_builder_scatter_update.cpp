// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/scatter_update.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

mlir::Value cast_index_to_i64(mlir::OpBuilder& b, mlir::Location loc, mlir::Value value, mlir::MLIRContext& ctx) {
    auto i64 = mlir::IntegerType::get(&ctx, 64);
    if (value.getType().isIndex()) {
        return b.create<mlir::arith::IndexCastOp>(loc, i64, value).getResult();
    }
    if (value.getType().isInteger(64)) {
        return value;
    }
    if (value.getType().isInteger(32) || value.getType().isInteger(16) || value.getType().isInteger(8)) {
        return b.create<mlir::arith::ExtSIOp>(loc, i64, value).getResult();
    }
    OPENVINO_THROW("ScatterUpdate MLIR: indices must be i8/i16/i32/i64");
}

mlir::Value dim_as_i64(mlir::OpBuilder& b,
                       mlir::Location loc,
                       mlir::Value tensor,
                       size_t dim,
                       mlir::MLIRContext& ctx) {
    auto idx = b.create<mlir::tensor::DimOp>(loc, tensor, static_cast<int64_t>(dim)).getResult();
    return b.create<mlir::arith::IndexCastOp>(loc, mlir::IntegerType::get(&ctx, 64), idx).getResult();
}

}  // namespace

mlir::ModuleOp build_mlir_scatter_update_from_model(const std::shared_ptr<const ov::Model>& model,
                                                    mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect,
                    mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::op::v3::ScatterUpdate> scatter;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto candidate = ov::as_type_ptr<const ov::op::v3::ScatterUpdate>(node)) {
            OPENVINO_ASSERT(!scatter, "ScatterUpdate MLIR builder: expected single op");
            scatter = candidate;
        }
    }
    OPENVINO_ASSERT(scatter, "ScatterUpdate MLIR builder: op not found");
    OPENVINO_ASSERT(scatter->get_input_partial_shape(0).rank().is_static() &&
                        scatter->get_input_partial_shape(1).rank().is_static() &&
                        scatter->get_input_partial_shape(2).rank().is_static() &&
                        scatter->get_output_partial_shape(0).rank().is_static(),
                    "ScatterUpdate MLIR: ranks must be static");

    auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(scatter->input_value(3).get_node_shared_ptr());
    OPENVINO_ASSERT(axis_const, "ScatterUpdate MLIR: axis must be constant");
    auto axis_values = axis_const->cast_vector<int64_t>();
    OPENVINO_ASSERT(axis_values.size() == 1, "ScatterUpdate MLIR: axis must be scalar");

    const size_t data_rank = static_cast<size_t>(scatter->get_input_partial_shape(0).rank().get_length());
    const size_t idx_rank = static_cast<size_t>(scatter->get_input_partial_shape(1).rank().get_length());
    int64_t axis = axis_values[0];
    if (axis < 0) {
        axis += static_cast<int64_t>(data_rank);
    }
    OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < data_rank, "ScatterUpdate MLIR: axis out of range");
    const size_t axis_pos = static_cast<size_t>(axis);
    OPENVINO_ASSERT(scatter->get_input_partial_shape(2).rank().get_length() ==
                        static_cast<int64_t>(data_rank + idx_rank - 1),
                    "ScatterUpdate MLIR: updates rank must be data_rank + indices_rank - 1");

    auto data_dims = to_shape(scatter->get_input_partial_shape(0));
    auto idx_dims = to_shape(scatter->get_input_partial_shape(1));
    auto upd_dims = to_shape(scatter->get_input_partial_shape(2));
    auto out_dims = to_shape(scatter->get_output_partial_shape(0));

    auto data_ty = to_mlir_type(scatter->get_input_element_type(0), ctx, /*fallback_f32=*/true);
    auto idx_ty = to_mlir_type(scatter->get_input_element_type(1), ctx, /*fallback_f32=*/false,
                               /*allow_unsigned=*/true,
                               /*allow_small_ints=*/true,
                               /*allow_bf16=*/false,
                               /*allow_boolean=*/false,
                               /*signless_integers=*/true);
    auto upd_ty = to_mlir_type(scatter->get_input_element_type(2), ctx, /*fallback_f32=*/true);
    auto out_ty = to_mlir_type(scatter->get_output_element_type(0), ctx, /*fallback_f32=*/true);
    OPENVINO_ASSERT(data_ty == upd_ty && data_ty == out_ty, "ScatterUpdate MLIR: data/update/output types must match");

    auto data_tensor_ty = mlir::RankedTensorType::get(data_dims, data_ty);
    auto idx_tensor_ty = mlir::RankedTensorType::get(idx_dims, idx_ty);
    auto upd_tensor_ty = mlir::RankedTensorType::get(upd_dims, upd_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_dims, out_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({data_tensor_ty, idx_tensor_ty, upd_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "scatter_update_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto out_dyn_dims = materialize_dynamic_dims_from_tensor(b, loc, func.getArgument(0), out_dims);
    auto generated = mlir::tensor::GenerateOp::create(
        b,
        loc,
        out_tensor_ty,
        out_dyn_dims,
        [&](mlir::OpBuilder& gb, mlir::Location gen_loc, mlir::ValueRange out_indices) {
            auto initial = gb.create<mlir::tensor::ExtractOp>(gen_loc, func.getArgument(0), out_indices).getResult();
            auto c0 = gb.create<mlir::arith::ConstantIndexOp>(gen_loc, 0).getResult();
            auto c1 = gb.create<mlir::arith::ConstantIndexOp>(gen_loc, 1).getResult();
            auto zero_i64 = gb.create<mlir::arith::ConstantIntOp>(gen_loc, 0, 64).getResult();
            auto idx_total = c1;
            for (size_t i = 0; i < idx_rank; ++i) {
                auto dim = gb.create<mlir::tensor::DimOp>(gen_loc, func.getArgument(1), static_cast<int64_t>(i))
                               .getResult();
                idx_total = gb.create<mlir::arith::MulIOp>(gen_loc, idx_total, dim).getResult();
            }

            auto loop = gb.create<mlir::scf::ForOp>(gen_loc, c0, idx_total, c1, mlir::ValueRange{initial});
            {
                auto* body = loop.getBody();
                mlir::OpBuilder lb(body, body->begin());
                auto linear = loop.getInductionVar();
                llvm::SmallVector<mlir::Value> idx_coords(idx_rank);
                for (size_t rev = 0; rev < idx_rank; ++rev) {
                    const size_t dim_pos = idx_rank - 1 - rev;
                    auto dim = lb.create<mlir::tensor::DimOp>(gen_loc, func.getArgument(1), static_cast<int64_t>(dim_pos))
                                   .getResult();
                    idx_coords[dim_pos] = lb.create<mlir::arith::RemUIOp>(gen_loc, linear, dim).getResult();
                    linear = lb.create<mlir::arith::DivUIOp>(gen_loc, linear, dim).getResult();
                }

                auto raw_idx = lb.create<mlir::tensor::ExtractOp>(gen_loc, func.getArgument(1), idx_coords).getResult();
                auto idx_i64 = cast_index_to_i64(lb, gen_loc, raw_idx, ctx);
                auto axis_dim_i64 = dim_as_i64(lb, gen_loc, func.getArgument(0), axis_pos, ctx);
                auto is_negative =
                    lb.create<mlir::arith::CmpIOp>(gen_loc, mlir::arith::CmpIPredicate::slt, idx_i64, zero_i64)
                        .getResult();
                auto wrapped_idx = lb.create<mlir::arith::AddIOp>(gen_loc, idx_i64, axis_dim_i64).getResult();
                auto normalized_idx =
                    lb.create<mlir::arith::SelectOp>(gen_loc, is_negative, wrapped_idx, idx_i64).getResult();
                auto out_axis_i64 = cast_index_to_i64(lb, gen_loc, out_indices[axis_pos], ctx);
                auto matches =
                    lb.create<mlir::arith::CmpIOp>(gen_loc, mlir::arith::CmpIPredicate::eq, normalized_idx, out_axis_i64)
                        .getResult();

                llvm::SmallVector<mlir::Value> upd_indices;
                upd_indices.reserve(data_rank + idx_rank - 1);
                for (size_t i = 0; i < axis_pos; ++i) {
                    upd_indices.push_back(out_indices[i]);
                }
                for (size_t i = 0; i < idx_rank; ++i) {
                    upd_indices.push_back(idx_coords[i]);
                }
                for (size_t i = axis_pos + 1; i < data_rank; ++i) {
                    upd_indices.push_back(out_indices[i]);
                }
                auto update = lb.create<mlir::tensor::ExtractOp>(gen_loc, func.getArgument(2), upd_indices).getResult();
                auto selected =
                    lb.create<mlir::arith::SelectOp>(gen_loc, matches, update, loop.getRegionIterArgs()[0]).getResult();
                lb.create<mlir::scf::YieldOp>(gen_loc, selected);
            }
            gb.create<mlir::tensor::YieldOp>(gen_loc, loop.getResult(0));
        });

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{generated.getResult()});
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
