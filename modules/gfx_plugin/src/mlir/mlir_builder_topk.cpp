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
        out1_kernel_shape.at(axis) *= 2;
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
            mlir::Value out1_axis = i;
            mlir::Value out1_axis_hi;
            if (emulate_i64_indices) {
                out1_axis = ib.create<mlir::arith::AddIOp>(loc, i, i).getResult();
                out1_axis_hi = ib.create<mlir::arith::AddIOp>(loc, out1_axis, c1).getResult();
            }
            auto out1_indices = make_indices(ib, loc, out1_kernel_shape, outer_iv, out1_axis, inner_iv);
            ib.create<mlir::memref::StoreOp>(loc, c_topk_i32_zero, func.getArgument(2), out1_indices);
            if (emulate_i64_indices) {
                auto out1_hi_indices = make_indices(ib, loc, out1_kernel_shape, outer_iv, out1_axis_hi, inner_iv);
                ib.create<mlir::memref::StoreOp>(loc, c_topk_i32_zero, func.getArgument(2), out1_hi_indices);
            }
        }

        auto axis_loop = rb.create<mlir::scf::ForOp>(loc, c0, c_axis, c1);
        {
            auto* abody = axis_loop.getBody();
            mlir::OpBuilder ab(abody, abody->begin());
            auto a = axis_loop.getInductionVar();
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

            auto insert_init = ab.create<mlir::arith::ConstantIndexOp>(loc, k).getResult();
            auto find_loop = ab.create<mlir::scf::ForOp>(loc, c0, c_k, c1, mlir::ValueRange{insert_init});
            {
                auto* fbody = find_loop.getBody();
                mlir::OpBuilder fb(fbody, fbody->begin());
                auto i = find_loop.getInductionVar();
                auto insert_acc = find_loop.getRegionIterArgs()[0];
                auto already_found =
                    fb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, insert_acc, c_k).getResult();
                auto current_indices = make_indices(fb, loc, out0_shape, outer_iv, i, inner_iv);
                auto current = fb.create<mlir::memref::LoadOp>(loc, func.getArgument(1), current_indices).getResult();
                mlir::Value current_out1_axis = i;
                if (emulate_i64_indices) {
                    current_out1_axis = fb.create<mlir::arith::AddIOp>(loc, i, i).getResult();
                }
                auto current_id_indices =
                    make_indices(fb, loc, out1_kernel_shape, outer_iv, current_out1_axis, inner_iv);
                auto current_id =
                    fb.create<mlir::memref::LoadOp>(loc, func.getArgument(2), current_id_indices).getResult();
                auto current_id_index =
                    fb.create<mlir::arith::IndexCastOp>(loc, fb.getIndexType(), current_id).getResult();
                auto candidate_has_lower_index =
                    fb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, a, current_id_index)
                        .getResult();
                mlir::Value value_better;
                mlir::Value values_equal;
                if (mlir::isa<mlir::FloatType>(out_val_ty)) {
                    auto pred = topk->get_mode() == ov::op::TopKMode::MAX
                                    ? mlir::arith::CmpFPredicate::OGT
                                    : mlir::arith::CmpFPredicate::OLT;
                    value_better = fb.create<mlir::arith::CmpFOp>(loc, pred, val, current).getResult();
                    values_equal =
                        fb.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, val, current).getResult();
                } else {
                    auto ity = mlir::cast<mlir::IntegerType>(out_val_ty);
                    auto pred = topk->get_mode() == ov::op::TopKMode::MAX
                                    ? (ity.isUnsigned() ? mlir::arith::CmpIPredicate::ugt
                                                        : mlir::arith::CmpIPredicate::sgt)
                                    : (ity.isUnsigned() ? mlir::arith::CmpIPredicate::ult
                                                        : mlir::arith::CmpIPredicate::slt);
                    value_better = fb.create<mlir::arith::CmpIOp>(loc, pred, val, current).getResult();
                    values_equal =
                        fb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, val, current).getResult();
                }
                auto equal_with_lower_index =
                    fb.create<mlir::arith::AndIOp>(loc, values_equal, candidate_has_lower_index).getResult();
                auto better =
                    fb.create<mlir::arith::OrIOp>(loc, value_better, equal_with_lower_index).getResult();
                auto take = fb.create<mlir::arith::AndIOp>(
                    loc,
                    better,
                    fb.create<mlir::arith::XOrIOp>(
                          loc,
                          already_found,
                          fb.create<mlir::arith::ConstantIntOp>(loc, 1, 1))
                        .getResult())
                                .getResult();
                auto next_insert = fb.create<mlir::arith::SelectOp>(loc, take, i, insert_acc).getResult();
                fb.create<mlir::scf::YieldOp>(loc, next_insert);
            }
            auto insert_pos = find_loop.getResult(0);
            auto has_insert =
                ab.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, insert_pos, c_k).getResult();
            auto insert_if = ab.create<mlir::scf::IfOp>(loc, has_insert, /*withElseRegion=*/false);
            {
                auto then_builder = insert_if.getThenBodyBuilder();
                auto shift_loop = then_builder.create<mlir::scf::ForOp>(loc, c0, c_k, c1);
                {
                    auto* sbody = shift_loop.getBody();
                    mlir::OpBuilder sb(sbody, sbody->begin());
                    auto shift = shift_loop.getInductionVar();
                    auto last = sb.create<mlir::arith::SubIOp>(loc, c_k, c1).getResult();
                    auto j = sb.create<mlir::arith::SubIOp>(loc, last, shift).getResult();
                    auto do_shift =
                        sb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, j, insert_pos).getResult();
                    auto shift_if = sb.create<mlir::scf::IfOp>(loc, do_shift, /*withElseRegion=*/false);
                    auto sib = shift_if.getThenBodyBuilder();
                    auto from = sib.create<mlir::arith::SubIOp>(loc, j, c1).getResult();
                    auto prev_val_indices = make_indices(sib, loc, out0_shape, outer_iv, from, inner_iv);
                    auto dst_val_indices = make_indices(sib, loc, out0_shape, outer_iv, j, inner_iv);
                    auto prev_val =
                        sib.create<mlir::memref::LoadOp>(loc, func.getArgument(1), prev_val_indices).getResult();
                    sib.create<mlir::memref::StoreOp>(loc, prev_val, func.getArgument(1), dst_val_indices);

                    mlir::Value prev_out1_axis = from;
                    mlir::Value dst_out1_axis = j;
                    mlir::Value dst_out1_axis_hi;
                    if (emulate_i64_indices) {
                        prev_out1_axis = sib.create<mlir::arith::AddIOp>(loc, from, from).getResult();
                        dst_out1_axis = sib.create<mlir::arith::AddIOp>(loc, j, j).getResult();
                        dst_out1_axis_hi = sib.create<mlir::arith::AddIOp>(loc, dst_out1_axis, c1).getResult();
                    }
                    auto prev_id_indices =
                        make_indices(sib, loc, out1_kernel_shape, outer_iv, prev_out1_axis, inner_iv);
                    auto dst_id_indices =
                        make_indices(sib, loc, out1_kernel_shape, outer_iv, dst_out1_axis, inner_iv);
                    auto prev_id =
                        sib.create<mlir::memref::LoadOp>(loc, func.getArgument(2), prev_id_indices).getResult();
                    sib.create<mlir::memref::StoreOp>(loc, prev_id, func.getArgument(2), dst_id_indices);
                    if (emulate_i64_indices) {
                        auto dst_hi_indices =
                            make_indices(sib, loc, out1_kernel_shape, outer_iv, dst_out1_axis_hi, inner_iv);
                        sib.create<mlir::memref::StoreOp>(loc, c_topk_i32_zero, func.getArgument(2), dst_hi_indices);
                    }
                }
                auto candidate_idx =
                    then_builder.create<mlir::arith::IndexCastOp>(loc, kernel_out_idx_ty, a).getResult();
                auto insert_val_indices = make_indices(then_builder, loc, out0_shape, outer_iv, insert_pos, inner_iv);
                then_builder.create<mlir::memref::StoreOp>(loc, val, func.getArgument(1), insert_val_indices);
                mlir::Value insert_out1_axis = insert_pos;
                mlir::Value insert_out1_axis_hi;
                if (emulate_i64_indices) {
                    insert_out1_axis =
                        then_builder.create<mlir::arith::AddIOp>(loc, insert_pos, insert_pos).getResult();
                    insert_out1_axis_hi =
                        then_builder.create<mlir::arith::AddIOp>(loc, insert_out1_axis, c1).getResult();
                }
                auto insert_id_indices =
                    make_indices(then_builder, loc, out1_kernel_shape, outer_iv, insert_out1_axis, inner_iv);
                then_builder.create<mlir::memref::StoreOp>(loc,
                                                           candidate_idx,
                                                           func.getArgument(2),
                                                           insert_id_indices);
                if (emulate_i64_indices) {
                    auto insert_hi_indices =
                        make_indices(then_builder, loc, out1_kernel_shape, outer_iv, insert_out1_axis_hi, inner_iv);
                    then_builder.create<mlir::memref::StoreOp>(loc,
                                                               c_topk_i32_zero,
                                                               func.getArgument(2),
                                                               insert_hi_indices);
                }
            }
        }
    }

    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
