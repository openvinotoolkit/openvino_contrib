// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/util/topk_base.hpp"

#include <algorithm>
#include <limits>

namespace ov {
namespace gfx_plugin {

namespace {
mlir::Value make_init_value(mlir::OpBuilder& b,
                            mlir::Location loc,
                            mlir::Type elem_ty,
                            ov::op::TopKMode mode) {
    if (mlir::isa<mlir::FloatType>(elem_ty)) {
        double init = (mode == ov::op::TopKMode::MAX) ? -std::numeric_limits<double>::infinity()
                                                      : std::numeric_limits<double>::infinity();
        return b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(elem_ty, init)).getResult();
    }
    auto ity = mlir::cast<mlir::IntegerType>(elem_ty);
    int64_t init = 0;
    if (mode == ov::op::TopKMode::MAX) {
        init = ity.isUnsigned() ? 0 : std::numeric_limits<int64_t>::min();
    } else {
        init = ity.isUnsigned() ? -1 : std::numeric_limits<int64_t>::max();
    }
    return b.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(elem_ty, init)).getResult();
}
}  // namespace

mlir::ModuleOp build_mlir_topk_from_model(const std::shared_ptr<const ov::Model>& model,
                                          mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::scf::SCFDialect>();

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
    OPENVINO_ASSERT(!in_shape.empty(), "TopK MLIR: input shape must be static");
    OPENVINO_ASSERT(!out0_shape.empty() && !out1_shape.empty(), "TopK MLIR: output shapes must be static");

    int64_t axis_signed = topk->get_axis();
    if (axis_signed < 0) {
        axis_signed += static_cast<int64_t>(in_shape.size());
    }
    OPENVINO_ASSERT(axis_signed >= 0 && static_cast<size_t>(axis_signed) < in_shape.size(),
                    "TopK MLIR: axis out of range");
    const auto axis = static_cast<size_t>(axis_signed);
    const auto axis_dim = static_cast<int64_t>(in_shape.at(axis));
    const auto k = static_cast<int64_t>(out0_shape.at(axis));
    OPENVINO_ASSERT(k > 0, "TopK MLIR: k must be > 0");
    OPENVINO_ASSERT(k <= axis_dim, "TopK MLIR: k exceeds axis dimension");

    OPENVINO_ASSERT(topk->get_sort_type() == ov::op::TopKSortType::NONE ||
                        topk->get_sort_type() == ov::op::TopKSortType::SORT_VALUES,
                    "TopK MLIR: only NONE or SORT_VALUES supported");

    auto in_elem_ty = to_mlir_type(topk->get_input_element_type(0), ctx, /*fallback_f32=*/false,
                                   /*allow_unsigned=*/true);
    auto out_val_ty = to_mlir_type(topk->get_output_element_type(0), ctx, /*fallback_f32=*/false,
                                   /*allow_unsigned=*/true);
    auto out_idx_ty = to_mlir_type(topk->get_output_element_type(1), ctx, /*fallback_f32=*/false,
                                   /*allow_unsigned=*/true, /*allow_small_ints=*/false,
                                   /*allow_bf16=*/false, /*allow_boolean=*/false,
                                   /*signless_integers=*/true);
    const bool emulate_i64_indices =
        mlir::isa<mlir::IntegerType>(out_idx_ty) &&
        mlir::cast<mlir::IntegerType>(out_idx_ty).getWidth() == 64;
    auto kernel_out_idx_ty = emulate_i64_indices ? mlir::IntegerType::get(&ctx, 32) : out_idx_ty;

    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out0_dims(out0_shape.begin(), out0_shape.end());
    mlir::SmallVector<int64_t> out1_dims(out1_shape.begin(), out1_shape.end());
    ov::Shape out1_kernel_shape = out1_shape;
    if (emulate_i64_indices) {
        out1_kernel_shape.push_back(2);
        out1_dims.assign(out1_kernel_shape.begin(), out1_kernel_shape.end());
    }

    auto in_memref_ty = mlir::MemRefType::get(in_dims, in_elem_ty);
    auto out0_memref_ty = mlir::MemRefType::get(out0_dims, out_val_ty);
    auto out1_memref_ty = mlir::MemRefType::get(out1_dims, kernel_out_idx_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_memref_ty, out0_memref_ty, out1_memref_ty}, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "topk_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    module->setAttr("gfx.parallel_dispatch", mlir::BoolAttr::get(&ctx, false));
    module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(&ctx, false));
    module->setAttr("gfx.force_single_dispatch", mlir::BoolAttr::get(&ctx, true));

    const int64_t inner = static_cast<int64_t>(
        ov::shape_size(ov::Shape(in_shape.begin() + axis + 1, in_shape.end())));
    const int64_t outer = static_cast<int64_t>(
        ov::shape_size(ov::Shape(in_shape.begin(), in_shape.begin() + axis)));
    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto c_inner = b.create<mlir::arith::ConstantIndexOp>(loc, inner);
    auto c_k = b.create<mlir::arith::ConstantIndexOp>(loc, k);
    auto c_axis = b.create<mlir::arith::ConstantIndexOp>(loc, axis_dim);
    auto c_rows = b.create<mlir::arith::ConstantIndexOp>(loc, outer * inner);
    auto c_topk_i32_zero = b.create<mlir::arith::ConstantIntOp>(
        loc,
        0,
        mlir::cast<mlir::IntegerType>(kernel_out_idx_ty).getWidth()).getResult();

    auto make_indices = [&](mlir::OpBuilder& ib,
                            mlir::Location iloc,
                            const ov::Shape& shape,
                            mlir::Value outer_linear_value,
                            mlir::Value axis_value,
                            mlir::Value inner_linear_value) {
        mlir::SmallVector<mlir::Value, 6> indices(shape.size(), c0.getResult());

        mlir::Value outer_linear = outer_linear_value;
        for (size_t rev = axis; rev > 0; --rev) {
            const size_t dim_idx = rev - 1;
            auto dim = ib.create<mlir::arith::ConstantIndexOp>(
                iloc, static_cast<int64_t>(shape[dim_idx]));
            indices[dim_idx] = ib.create<mlir::arith::RemUIOp>(iloc, outer_linear, dim).getResult();
            outer_linear = ib.create<mlir::arith::DivUIOp>(iloc, outer_linear, dim).getResult();
        }

        indices[axis] = axis_value;

        mlir::Value inner_linear = inner_linear_value;
        for (size_t rev = shape.size(); rev > axis + 1; --rev) {
            const size_t dim_idx = rev - 1;
            auto dim = ib.create<mlir::arith::ConstantIndexOp>(
                iloc, static_cast<int64_t>(shape[dim_idx]));
            indices[dim_idx] = ib.create<mlir::arith::RemUIOp>(iloc, inner_linear, dim).getResult();
            inner_linear = ib.create<mlir::arith::DivUIOp>(iloc, inner_linear, dim).getResult();
        }

        return indices;
    };

    auto make_out1_indices = [&](mlir::OpBuilder& ib,
                                 mlir::Location iloc,
                                 mlir::Value outer_linear_value,
                                 mlir::Value axis_value,
                                 mlir::Value inner_linear_value,
                                 mlir::Value lane_value) {
        auto indices = make_indices(ib, iloc, out1_shape, outer_linear_value, axis_value, inner_linear_value);
        if (emulate_i64_indices) {
            indices.push_back(lane_value);
        }
        return indices;
    };

    auto make_value_better = [&](mlir::OpBuilder& ib,
                                 mlir::Location iloc,
                                 mlir::Value candidate_value,
                                 mlir::Value candidate_index,
                                 mlir::Value current_value,
                                 mlir::Value current_index) -> mlir::Value {
        auto candidate_has_lower_index =
            ib.create<mlir::arith::CmpIOp>(iloc,
                                           mlir::arith::CmpIPredicate::ult,
                                           candidate_index,
                                           current_index)
                .getResult();
        mlir::Value value_better;
        mlir::Value values_equal;
        if (mlir::isa<mlir::FloatType>(out_val_ty)) {
            auto pred = topk->get_mode() == ov::op::TopKMode::MAX
                            ? mlir::arith::CmpFPredicate::OGT
                            : mlir::arith::CmpFPredicate::OLT;
            value_better = ib.create<mlir::arith::CmpFOp>(iloc, pred, candidate_value, current_value).getResult();
            values_equal =
                ib.create<mlir::arith::CmpFOp>(iloc, mlir::arith::CmpFPredicate::OEQ, candidate_value, current_value)
                    .getResult();
        } else {
            auto ity = mlir::cast<mlir::IntegerType>(out_val_ty);
            auto pred = topk->get_mode() == ov::op::TopKMode::MAX
                            ? (ity.isUnsigned() ? mlir::arith::CmpIPredicate::ugt
                                                : mlir::arith::CmpIPredicate::sgt)
                            : (ity.isUnsigned() ? mlir::arith::CmpIPredicate::ult
                                                : mlir::arith::CmpIPredicate::slt);
            value_better = ib.create<mlir::arith::CmpIOp>(iloc, pred, candidate_value, current_value).getResult();
            values_equal =
                ib.create<mlir::arith::CmpIOp>(iloc, mlir::arith::CmpIPredicate::eq, candidate_value, current_value)
                    .getResult();
        }
        auto equal_with_lower_index =
            ib.create<mlir::arith::AndIOp>(iloc, values_equal, candidate_has_lower_index).getResult();
        return ib.create<mlir::arith::OrIOp>(iloc, value_better, equal_with_lower_index).getResult();
    };

    auto row_loop = b.create<mlir::scf::ParallelOp>(
        loc,
        mlir::ValueRange{c0},
        mlir::ValueRange{c_rows},
        mlir::ValueRange{c1});
    {
        auto* body = row_loop.getBody();
        mlir::OpBuilder rb(body, body->begin());
        auto row_iv = row_loop.getInductionVars()[0];
        auto outer_iv = rb.create<mlir::arith::DivUIOp>(loc, row_iv, c_inner).getResult();
        auto inner_iv = rb.create<mlir::arith::RemUIOp>(loc, row_iv, c_inner).getResult();

        auto init_loop = rb.create<mlir::scf::ForOp>(loc, c0, c_k, c1);
        {
            auto* ibody = init_loop.getBody();
            mlir::OpBuilder ib(ibody, ibody->begin());
            auto i = init_loop.getInductionVar();
            auto out0_indices = make_indices(ib, loc, out0_shape, outer_iv, i, inner_iv);
            ib.create<mlir::memref::StoreOp>(loc,
                                             make_init_value(ib, loc, out_val_ty, topk->get_mode()),
                                             func.getArgument(1),
                                             out0_indices);
            auto out1_indices = make_out1_indices(ib, loc, outer_iv, i, inner_iv, c0.getResult());
            ib.create<mlir::memref::StoreOp>(loc, c_topk_i32_zero, func.getArgument(2), out1_indices);
            if (emulate_i64_indices) {
                auto out1_hi_indices = make_out1_indices(ib, loc, outer_iv, i, inner_iv, c1.getResult());
                ib.create<mlir::memref::StoreOp>(loc, c_topk_i32_zero, func.getArgument(2), out1_hi_indices);
            }
        }

        auto axis_loop = rb.create<mlir::scf::ForOp>(
            loc, c0, c_axis, c1, mlir::ValueRange{c0.getResult(), c0.getResult()});
        {
            auto* abody = axis_loop.getBody();
            mlir::OpBuilder ab(abody, abody->begin());
            auto a = axis_loop.getInductionVar();
            auto filled_count = axis_loop.getRegionIterArgs()[0];
            auto worst_pos = axis_loop.getRegionIterArgs()[1];
            auto in_indices = make_indices(ab, loc, in_shape, outer_iv, a, inner_iv);
            auto raw_val = ab.create<mlir::memref::LoadOp>(loc, func.getArgument(0), in_indices).getResult();
            mlir::Value val = raw_val;
            if (val.getType() != out_val_ty) {
                if (mlir::isa<mlir::FloatType>(val.getType()) && mlir::isa<mlir::FloatType>(out_val_ty)) {
                    const auto src_width = mlir::cast<mlir::FloatType>(val.getType()).getWidth();
                    const auto dst_width = mlir::cast<mlir::FloatType>(out_val_ty).getWidth();
                    val = dst_width > src_width
                              ? ab.create<mlir::arith::ExtFOp>(loc, out_val_ty, val).getResult()
                              : ab.create<mlir::arith::TruncFOp>(loc, out_val_ty, val).getResult();
                }
            }

            auto worst_val_indices = make_indices(ab, loc, out0_shape, outer_iv, worst_pos, inner_iv);
            auto worst_val = ab.create<mlir::memref::LoadOp>(loc, func.getArgument(1), worst_val_indices).getResult();
            auto worst_id_indices = make_out1_indices(ab, loc, outer_iv, worst_pos, inner_iv, c0.getResult());
            auto worst_id = ab.create<mlir::memref::LoadOp>(loc, func.getArgument(2), worst_id_indices).getResult();
            auto worst_id_index = ab.create<mlir::arith::IndexCastOp>(loc, ab.getIndexType(), worst_id).getResult();
            auto has_room =
                ab.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, filled_count, c_k).getResult();
            auto better_than_worst = make_value_better(ab, loc, val, a, worst_val, worst_id_index);
            auto should_insert = ab.create<mlir::arith::OrIOp>(loc, has_room, better_than_worst).getResult();
            auto insert_pos = ab.create<mlir::arith::SelectOp>(loc, has_room, filled_count, worst_pos).getResult();
            auto insert_if = ab.create<mlir::scf::IfOp>(
                loc, mlir::TypeRange{ab.getIndexType(), ab.getIndexType()}, should_insert, /*withElseRegion=*/true);
            {
                auto then_builder = insert_if.getThenBodyBuilder();
                auto candidate_idx =
                    then_builder.create<mlir::arith::IndexCastOp>(loc, kernel_out_idx_ty, a).getResult();
                auto insert_val_indices = make_indices(then_builder, loc, out0_shape, outer_iv, insert_pos, inner_iv);
                then_builder.create<mlir::memref::StoreOp>(loc, val, func.getArgument(1), insert_val_indices);
                auto insert_id_indices =
                    make_out1_indices(then_builder, loc, outer_iv, insert_pos, inner_iv, c0.getResult());
                then_builder.create<mlir::memref::StoreOp>(loc,
                                                           candidate_idx,
                                                           func.getArgument(2),
                                                           insert_id_indices);
                if (emulate_i64_indices) {
                    auto insert_hi_indices =
                        make_out1_indices(then_builder, loc, outer_iv, insert_pos, inner_iv, c1.getResult());
                    then_builder.create<mlir::memref::StoreOp>(loc,
                                                               c_topk_i32_zero,
                                                               func.getArgument(2),
                                                               insert_hi_indices);
                }
                auto next_filled_candidate = then_builder.create<mlir::arith::AddIOp>(loc, filled_count, c1).getResult();
                auto next_filled =
                    then_builder.create<mlir::arith::SelectOp>(loc, has_room, next_filled_candidate, filled_count)
                        .getResult();
                auto worst_scan = then_builder.create<mlir::scf::ForOp>(
                    loc, c1, c_k, c1, mlir::ValueRange{c0.getResult()});
                {
                    auto* sbody = worst_scan.getBody();
                    mlir::OpBuilder sb(sbody, sbody->begin());
                    auto i = worst_scan.getInductionVar();
                    auto worst_acc = worst_scan.getRegionIterArgs()[0];
                    auto acc_val_indices = make_indices(sb, loc, out0_shape, outer_iv, worst_acc, inner_iv);
                    auto acc_val =
                        sb.create<mlir::memref::LoadOp>(loc, func.getArgument(1), acc_val_indices).getResult();
                    auto acc_id_indices = make_out1_indices(sb, loc, outer_iv, worst_acc, inner_iv, c0.getResult());
                    auto acc_id = sb.create<mlir::memref::LoadOp>(loc, func.getArgument(2), acc_id_indices).getResult();
                    auto acc_id_index =
                        sb.create<mlir::arith::IndexCastOp>(loc, sb.getIndexType(), acc_id).getResult();
                    auto cur_val_indices = make_indices(sb, loc, out0_shape, outer_iv, i, inner_iv);
                    auto cur_val =
                        sb.create<mlir::memref::LoadOp>(loc, func.getArgument(1), cur_val_indices).getResult();
                    auto cur_id_indices = make_out1_indices(sb, loc, outer_iv, i, inner_iv, c0.getResult());
                    auto cur_id = sb.create<mlir::memref::LoadOp>(loc, func.getArgument(2), cur_id_indices).getResult();
                    auto cur_id_index =
                        sb.create<mlir::arith::IndexCastOp>(loc, sb.getIndexType(), cur_id).getResult();
                    auto acc_better_than_cur =
                        make_value_better(sb, loc, acc_val, acc_id_index, cur_val, cur_id_index);
                    auto next_worst = sb.create<mlir::arith::SelectOp>(loc, acc_better_than_cur, i, worst_acc)
                                          .getResult();
                    sb.create<mlir::scf::YieldOp>(loc, next_worst);
                }
                then_builder.create<mlir::scf::YieldOp>(loc,
                                                        mlir::ValueRange{next_filled, worst_scan.getResult(0)});
                auto else_builder = insert_if.getElseBodyBuilder();
                else_builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{filled_count, worst_pos});
            }
            ab.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{insert_if.getResult(0), insert_if.getResult(1)});
        }

        auto sort_outer = rb.create<mlir::scf::ForOp>(loc, c0, c_k, c1);
        {
            auto* obody = sort_outer.getBody();
            mlir::OpBuilder ob(obody, obody->begin());
            auto i = sort_outer.getInductionVar();
            auto j_begin = ob.create<mlir::arith::AddIOp>(loc, i, c1).getResult();
            auto sort_inner = ob.create<mlir::scf::ForOp>(loc, j_begin, c_k, c1);
            {
                auto* ibody = sort_inner.getBody();
                mlir::OpBuilder sb(ibody, ibody->begin());
                auto j = sort_inner.getInductionVar();
                auto i_val_indices = make_indices(sb, loc, out0_shape, outer_iv, i, inner_iv);
                auto j_val_indices = make_indices(sb, loc, out0_shape, outer_iv, j, inner_iv);
                auto i_val = sb.create<mlir::memref::LoadOp>(loc, func.getArgument(1), i_val_indices).getResult();
                auto j_val = sb.create<mlir::memref::LoadOp>(loc, func.getArgument(1), j_val_indices).getResult();
                auto i_id_indices = make_out1_indices(sb, loc, outer_iv, i, inner_iv, c0.getResult());
                auto j_id_indices = make_out1_indices(sb, loc, outer_iv, j, inner_iv, c0.getResult());
                auto i_id = sb.create<mlir::memref::LoadOp>(loc, func.getArgument(2), i_id_indices).getResult();
                auto j_id = sb.create<mlir::memref::LoadOp>(loc, func.getArgument(2), j_id_indices).getResult();
                auto i_id_index = sb.create<mlir::arith::IndexCastOp>(loc, sb.getIndexType(), i_id).getResult();
                auto j_id_index = sb.create<mlir::arith::IndexCastOp>(loc, sb.getIndexType(), j_id).getResult();
                auto swap = make_value_better(sb, loc, j_val, j_id_index, i_val, i_id_index);
                auto swap_if = sb.create<mlir::scf::IfOp>(loc, swap, /*withElseRegion=*/false);
                auto sib = swap_if.getThenBodyBuilder();
                sib.create<mlir::memref::StoreOp>(loc, j_val, func.getArgument(1), i_val_indices);
                sib.create<mlir::memref::StoreOp>(loc, i_val, func.getArgument(1), j_val_indices);
                sib.create<mlir::memref::StoreOp>(loc, j_id, func.getArgument(2), i_id_indices);
                sib.create<mlir::memref::StoreOp>(loc, i_id, func.getArgument(2), j_id_indices);
                if (emulate_i64_indices) {
                    auto i_hi_indices = make_out1_indices(sib, loc, outer_iv, i, inner_iv, c1.getResult());
                    auto j_hi_indices = make_out1_indices(sib, loc, outer_iv, j, inner_iv, c1.getResult());
                    sib.create<mlir::memref::StoreOp>(loc, c_topk_i32_zero, func.getArgument(2), i_hi_indices);
                    sib.create<mlir::memref::StoreOp>(loc, c_topk_i32_zero, func.getArgument(2), j_hi_indices);
                }
            }
        }
    }

    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
