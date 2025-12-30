// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "openvino/op/softmax.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/core/model.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <limits>

namespace ov {
namespace gfx_plugin {

namespace {
std::shared_ptr<const ov::Node> find_single_softmax(const std::shared_ptr<const ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<const ov::op::v1::Softmax>(node) || ov::is_type<const ov::op::v8::Softmax>(node))
            return node;
    }
    OPENVINO_THROW("Softmax builder: Softmax op not found");
}

std::shared_ptr<const ov::Node> find_single_logsoftmax(const std::shared_ptr<const ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<const ov::op::v5::LogSoftmax>(node))
            return node;
    }
    OPENVINO_THROW("Softmax builder: LogSoftmax op not found");
}

mlir::ModuleOp build_softmax_like_from_node(const std::shared_ptr<const ov::Node>& sm,
                                            mlir::MLIRContext& ctx,
                                            const ov::Shape* input_shape_override) {
    ctx.loadDialect<mlir::func::FuncDialect,
                    mlir::scf::SCFDialect,
                    mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect,
                    mlir::math::MathDialect>();
    const ov::Shape shape = (input_shape_override && !input_shape_override->empty())
                                ? *input_shape_override
                                : sm->get_input_shape(0);
    auto to_elem_ty = [&](ov::element::Type et) -> mlir::Type {
        switch (et) {
            case ov::element::f16: return mlir::Float16Type::get(&ctx);
            case ov::element::f32: return mlir::Float32Type::get(&ctx);
            default: return mlir::Float32Type::get(&ctx);
        }
    };
    auto elem_ty = to_elem_ty(sm->get_output_element_type(0));
    auto compute_ty = elem_ty;
    if (mlir::isa<mlir::Float16Type>(elem_ty)) {
        compute_ty = mlir::Float32Type::get(&ctx);
    }

    int64_t axis = -1;
    const bool log_softmax = ov::is_type<const ov::op::v5::LogSoftmax>(sm);
    if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(sm)) axis = s1->get_axis();
    else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(sm)) axis = s8->get_axis();
    else if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(sm)) axis = ls->get_axis();
    else OPENVINO_THROW("Softmax builder: unsupported op kind");
    if (axis < 0) axis += static_cast<int64_t>(shape.size());
    int64_t rows = 1, cols = shape.at(axis), inner = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) rows *= shape[i];
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= shape[i];

    mlir::SmallVector<int64_t> dims(shape.begin(), shape.end());
    auto ty = mlir::MemRefType::get(dims, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({ty, ty}, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "softmax_main", func_type);
    func.addEntryBlock();
    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto cast_to_compute = [&](mlir::OpBuilder& bld, mlir::Value v) -> mlir::Value {
        if (v.getType() == compute_ty) {
            return v;
        }
        return bld.create<mlir::arith::ExtFOp>(loc, compute_ty, v);
    };
    auto cast_to_output = [&](mlir::OpBuilder& bld, mlir::Value v) -> mlir::Value {
        if (v.getType() == elem_ty) {
            return v;
        }
        return bld.create<mlir::arith::TruncFOp>(loc, elem_ty, v);
    };
    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto rows_c = b.create<mlir::arith::ConstantIndexOp>(loc, rows);
    auto cols_c = b.create<mlir::arith::ConstantIndexOp>(loc, cols);
    auto inner_c = b.create<mlir::arith::ConstantIndexOp>(loc, inner);

    auto flat_dim = rows * cols * inner;
    auto flat_ty = mlir::MemRefType::get({flat_dim}, elem_ty);
    auto offset = b.getI64IntegerAttr(0);
    mlir::SmallVector<mlir::OpFoldResult, 1> sizes{b.getI64IntegerAttr(flat_dim)};
    mlir::SmallVector<mlir::OpFoldResult, 1> strides{b.getI64IntegerAttr(1)};
    auto flat_in = b.create<mlir::memref::ReinterpretCastOp>(loc,
                                                            flat_ty,
                                                            func.getArgument(0),
                                                            offset,
                                                            sizes,
                                                            strides);
    auto flat_out = b.create<mlir::memref::ReinterpretCastOp>(loc,
                                                             flat_ty,
                                                             func.getArgument(1),
                                                             offset,
                                                             sizes,
                                                             strides);

    auto for_row = b.create<mlir::scf::ForOp>(loc, c0, rows_c, c1);
    auto brow = mlir::OpBuilder::atBlockBegin(for_row.getBody());
    mlir::scf::ForOp for_inner;
    if (inner > 1) {
        for_inner = brow.create<mlir::scf::ForOp>(loc, c0, inner_c, c1);
    }
    auto binner = inner > 1 ? mlir::OpBuilder::atBlockBegin(for_inner.getBody()) : brow;
    mlir::Value inner_idx = inner > 1 ? for_inner.getInductionVar() : c0;

    auto compute_flat = [&](mlir::OpBuilder& bld, mlir::Value row, mlir::Value col, mlir::Value in_idx) {
        auto mul1 = bld.create<mlir::arith::MulIOp>(loc, row, cols_c);
        auto add1 = bld.create<mlir::arith::AddIOp>(loc, mul1, col);
        auto mul2 = bld.create<mlir::arith::MulIOp>(loc, add1, inner_c);
        return bld.create<mlir::arith::AddIOp>(loc, mul2, in_idx);
    };

    // Compute max for this (row, inner) slice.
    auto neg_inf = binner.create<mlir::arith::ConstantOp>(
        loc,
        mlir::FloatAttr::get(compute_ty, -std::numeric_limits<float>::infinity()));
    auto for_col_max = binner.create<mlir::scf::ForOp>(loc, c0, cols_c, c1, neg_inf.getResult());
    auto bmax = mlir::OpBuilder::atBlockBegin(for_col_max.getBody());
    auto flat_max = compute_flat(bmax, for_row.getInductionVar(), for_col_max.getInductionVar(), inner_idx);
    auto val_max_raw = bmax.create<mlir::memref::LoadOp>(loc, flat_in, mlir::ValueRange{flat_max});
    auto val_max = cast_to_compute(bmax, val_max_raw);
    auto cur_max = for_col_max.getRegionIterArgs()[0];
    auto cmp_max = bmax.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, val_max, cur_max);
    auto sel_max = bmax.create<mlir::arith::SelectOp>(loc, cmp_max, val_max, cur_max);
    bmax.create<mlir::scf::YieldOp>(loc, sel_max.getResult());
    auto max_val = for_col_max.getResult(0);

    // Compute sum of exp(x - max).
    auto zero = binner.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(compute_ty, 0.0f));
    auto for_col_sum = binner.create<mlir::scf::ForOp>(loc, c0, cols_c, c1, zero.getResult());
    auto bsum = mlir::OpBuilder::atBlockBegin(for_col_sum.getBody());
    auto flat_sum = compute_flat(bsum, for_row.getInductionVar(), for_col_sum.getInductionVar(), inner_idx);
    auto val_sum_raw = bsum.create<mlir::memref::LoadOp>(loc, flat_in, mlir::ValueRange{flat_sum});
    auto val_sum = cast_to_compute(bsum, val_sum_raw);
    auto diff = bsum.create<mlir::arith::SubFOp>(loc, val_sum, max_val);
    auto expv = bsum.create<mlir::math::ExpOp>(loc, diff);
    auto cur_sum = for_col_sum.getRegionIterArgs()[0];
    auto new_sum = bsum.create<mlir::arith::AddFOp>(loc, cur_sum, expv);
    bsum.create<mlir::scf::YieldOp>(loc, new_sum.getResult());
    auto sum_val = for_col_sum.getResult(0);

    // Write output.
    auto for_col = binner.create<mlir::scf::ForOp>(loc, c0, cols_c, c1);
    auto bcol = mlir::OpBuilder::atBlockBegin(for_col.getBody());
    auto flat_out_idx = compute_flat(bcol, for_row.getInductionVar(), for_col.getInductionVar(), inner_idx);
    auto val_raw = bcol.create<mlir::memref::LoadOp>(loc, flat_in, mlir::ValueRange{flat_out_idx});
    auto val = cast_to_compute(bcol, val_raw);
    auto diff_out = bcol.create<mlir::arith::SubFOp>(loc, val, max_val);
    mlir::Value out_val;
    if (log_softmax) {
        auto logsum = bcol.create<mlir::math::LogOp>(loc, sum_val);
        out_val = bcol.create<mlir::arith::SubFOp>(loc, diff_out, logsum);
    } else {
        auto exp_out = bcol.create<mlir::math::ExpOp>(loc, diff_out);
        out_val = bcol.create<mlir::arith::DivFOp>(loc, exp_out, sum_val);
    }
    auto out_cast = cast_to_output(bcol, out_val);
    bcol.create<mlir::memref::StoreOp>(loc, out_cast, flat_out, mlir::ValueRange{flat_out_idx});

    b.create<mlir::func::ReturnOp>(loc);
    return module;
}

mlir::ModuleOp build_softmax_like_from_node_tiled(const std::shared_ptr<const ov::Node>& sm,
                                                  mlir::MLIRContext& ctx,
                                                  const ov::Shape* input_shape_override) {
    ctx.loadDialect<mlir::func::FuncDialect,
                    mlir::scf::SCFDialect,
                    mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect,
                    mlir::math::MathDialect>();
    const ov::Shape shape = (input_shape_override && !input_shape_override->empty())
                                ? *input_shape_override
                                : sm->get_input_shape(0);
    auto to_elem_ty = [&](ov::element::Type et) -> mlir::Type {
        switch (et) {
            case ov::element::f16: return mlir::Float16Type::get(&ctx);
            case ov::element::f32: return mlir::Float32Type::get(&ctx);
            default: return mlir::Float32Type::get(&ctx);
        }
    };
    auto elem_ty = to_elem_ty(sm->get_output_element_type(0));
    auto compute_ty = elem_ty;
    if (mlir::isa<mlir::Float16Type>(elem_ty)) {
        compute_ty = mlir::Float32Type::get(&ctx);
    }

    int64_t axis = -1;
    const bool log_softmax = ov::is_type<const ov::op::v5::LogSoftmax>(sm);
    if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(sm)) axis = s1->get_axis();
    else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(sm)) axis = s8->get_axis();
    else if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(sm)) axis = ls->get_axis();
    else OPENVINO_THROW("Softmax builder: unsupported op kind");
    if (axis < 0) axis += static_cast<int64_t>(shape.size());
    int64_t rows = 1, cols = shape.at(axis), inner = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) rows *= shape[i];
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= shape[i];

    mlir::SmallVector<int64_t> dims(shape.begin(), shape.end());
    auto ty = mlir::MemRefType::get(dims, elem_ty);
    auto param_ty = mlir::MemRefType::get({2}, mlir::IntegerType::get(&ctx, 32));

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({ty, param_ty, ty}, {});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "softmax_main", func_type);
    func.addEntryBlock();
    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto cast_to_compute = [&](mlir::OpBuilder& bld, mlir::Value v) -> mlir::Value {
        if (v.getType() == compute_ty) {
            return v;
        }
        return bld.create<mlir::arith::ExtFOp>(loc, compute_ty, v);
    };
    auto cast_to_output = [&](mlir::OpBuilder& bld, mlir::Value v) -> mlir::Value {
        if (v.getType() == elem_ty) {
            return v;
        }
        return bld.create<mlir::arith::TruncFOp>(loc, elem_ty, v);
    };

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto rows_c = b.create<mlir::arith::ConstantIndexOp>(loc, rows);
    auto cols_c = b.create<mlir::arith::ConstantIndexOp>(loc, cols);
    auto inner_c = b.create<mlir::arith::ConstantIndexOp>(loc, inner);

    auto flat_dim = rows * cols * inner;
    auto flat_ty = mlir::MemRefType::get({flat_dim}, elem_ty);
    auto offset = b.getI64IntegerAttr(0);
    mlir::SmallVector<mlir::OpFoldResult, 1> sizes{b.getI64IntegerAttr(flat_dim)};
    mlir::SmallVector<mlir::OpFoldResult, 1> strides{b.getI64IntegerAttr(1)};
    auto flat_in = b.create<mlir::memref::ReinterpretCastOp>(loc,
                                                            flat_ty,
                                                            func.getArgument(0),
                                                            offset,
                                                            sizes,
                                                            strides);
    auto flat_out = b.create<mlir::memref::ReinterpretCastOp>(loc,
                                                             flat_ty,
                                                             func.getArgument(2),
                                                             offset,
                                                             sizes,
                                                             strides);

    auto params = func.getArgument(1);
    auto off_i32 = b.create<mlir::memref::LoadOp>(loc, params, mlir::ValueRange{c0});
    auto count_i32 = b.create<mlir::memref::LoadOp>(loc, params, mlir::ValueRange{c1});
    auto offset_idx = b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), off_i32);
    auto count_idx = b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), count_i32);

    auto for_idx = b.create<mlir::scf::ForOp>(loc, c0, count_idx, c1);
    auto bidx = mlir::OpBuilder::atBlockBegin(for_idx.getBody());
    auto global_idx = bidx.create<mlir::arith::AddIOp>(loc, offset_idx, for_idx.getInductionVar());
    auto global_i32 = bidx.create<mlir::arith::IndexCastOp>(loc, bidx.getI32Type(), global_idx);
    auto inner_i32 = bidx.create<mlir::arith::ConstantIntOp>(loc, inner, 32);
    auto row_i32 = bidx.create<mlir::arith::DivUIOp>(loc, global_i32, inner_i32);
    auto inner_i32_val = bidx.create<mlir::arith::RemUIOp>(loc, global_i32, inner_i32);
    auto row = bidx.create<mlir::arith::IndexCastOp>(loc, bidx.getIndexType(), row_i32);
    auto inner_idx = bidx.create<mlir::arith::IndexCastOp>(loc, bidx.getIndexType(), inner_i32_val);

    auto compute_flat = [&](mlir::OpBuilder& bld, mlir::Value row_v, mlir::Value col_v, mlir::Value in_idx) {
        auto mul1 = bld.create<mlir::arith::MulIOp>(loc, row_v, cols_c);
        auto add1 = bld.create<mlir::arith::AddIOp>(loc, mul1, col_v);
        auto mul2 = bld.create<mlir::arith::MulIOp>(loc, add1, inner_c);
        return bld.create<mlir::arith::AddIOp>(loc, mul2, in_idx);
    };

    // Compute max for this (row, inner) slice.
    auto neg_inf = bidx.create<mlir::arith::ConstantOp>(
        loc,
        mlir::FloatAttr::get(compute_ty, -std::numeric_limits<float>::infinity()));
    auto for_col_max = bidx.create<mlir::scf::ForOp>(loc, c0, cols_c, c1, neg_inf.getResult());
    auto bmax = mlir::OpBuilder::atBlockBegin(for_col_max.getBody());
    auto flat_max = compute_flat(bmax, row, for_col_max.getInductionVar(), inner_idx);
    auto val_max_raw = bmax.create<mlir::memref::LoadOp>(loc, flat_in, mlir::ValueRange{flat_max});
    auto val_max = cast_to_compute(bmax, val_max_raw);
    auto cur_max = for_col_max.getRegionIterArgs()[0];
    auto cmp_max = bmax.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, val_max, cur_max);
    auto sel_max = bmax.create<mlir::arith::SelectOp>(loc, cmp_max, val_max, cur_max);
    bmax.create<mlir::scf::YieldOp>(loc, sel_max.getResult());
    auto max_val = for_col_max.getResult(0);

    // Compute sum of exp(x - max).
    auto zero = bidx.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(compute_ty, 0.0f));
    auto for_col_sum = bidx.create<mlir::scf::ForOp>(loc, c0, cols_c, c1, zero.getResult());
    auto bsum = mlir::OpBuilder::atBlockBegin(for_col_sum.getBody());
    auto flat_sum = compute_flat(bsum, row, for_col_sum.getInductionVar(), inner_idx);
    auto val_sum_raw = bsum.create<mlir::memref::LoadOp>(loc, flat_in, mlir::ValueRange{flat_sum});
    auto val_sum = cast_to_compute(bsum, val_sum_raw);
    auto diff = bsum.create<mlir::arith::SubFOp>(loc, val_sum, max_val);
    auto expv = bsum.create<mlir::math::ExpOp>(loc, diff);
    auto cur_sum = for_col_sum.getRegionIterArgs()[0];
    auto new_sum = bsum.create<mlir::arith::AddFOp>(loc, cur_sum, expv);
    bsum.create<mlir::scf::YieldOp>(loc, new_sum.getResult());
    auto sum_val = for_col_sum.getResult(0);

    // Write output.
    auto for_col = bidx.create<mlir::scf::ForOp>(loc, c0, cols_c, c1);
    auto bcol = mlir::OpBuilder::atBlockBegin(for_col.getBody());
    auto flat_out_idx = compute_flat(bcol, row, for_col.getInductionVar(), inner_idx);
    auto val_raw = bcol.create<mlir::memref::LoadOp>(loc, flat_in, mlir::ValueRange{flat_out_idx});
    auto val = cast_to_compute(bcol, val_raw);
    auto diff_out = bcol.create<mlir::arith::SubFOp>(loc, val, max_val);
    mlir::Value out_val;
    if (log_softmax) {
        auto logsum = bcol.create<mlir::math::LogOp>(loc, sum_val);
        out_val = bcol.create<mlir::arith::SubFOp>(loc, diff_out, logsum);
    } else {
        auto exp_out = bcol.create<mlir::math::ExpOp>(loc, diff_out);
        out_val = bcol.create<mlir::arith::DivFOp>(loc, exp_out, sum_val);
    }
    auto out_cast = cast_to_output(bcol, out_val);
    bcol.create<mlir::memref::StoreOp>(loc, out_cast, flat_out, mlir::ValueRange{flat_out_idx});

    b.create<mlir::func::ReturnOp>(loc);
    return module;
}
}  // namespace

mlir::ModuleOp build_mlir_softmax_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx) {
    auto sm = find_single_softmax(model);
    return build_softmax_like_from_node(sm, ctx, nullptr);
}

mlir::ModuleOp build_mlir_logsoftmax_from_model(const std::shared_ptr<const ov::Model>& model,
                                                mlir::MLIRContext& ctx) {
    auto sm = find_single_logsoftmax(model);
    return build_softmax_like_from_node(sm, ctx, nullptr);
}

mlir::ModuleOp build_mlir_softmax_from_node(const std::shared_ptr<const ov::Node>& node,
                                            mlir::MLIRContext& ctx,
                                            const ov::Shape& input_shape) {
    OPENVINO_ASSERT(node, "Softmax builder: node is null");
    return build_softmax_like_from_node(node, ctx, &input_shape);
}

mlir::ModuleOp build_mlir_logsoftmax_from_node(const std::shared_ptr<const ov::Node>& node,
                                               mlir::MLIRContext& ctx,
                                               const ov::Shape& input_shape) {
    OPENVINO_ASSERT(node, "LogSoftmax builder: node is null");
    return build_softmax_like_from_node(node, ctx, &input_shape);
}

mlir::ModuleOp build_mlir_softmax_tiled_from_node(const std::shared_ptr<const ov::Node>& node,
                                                  mlir::MLIRContext& ctx,
                                                  const ov::Shape& input_shape) {
    OPENVINO_ASSERT(node, "Softmax builder: node is null");
    return build_softmax_like_from_node_tiled(node, ctx, &input_shape);
}

mlir::ModuleOp build_mlir_logsoftmax_tiled_from_node(const std::shared_ptr<const ov::Node>& node,
                                                     mlir::MLIRContext& ctx,
                                                     const ov::Shape& input_shape) {
    OPENVINO_ASSERT(node, "LogSoftmax builder: node is null");
    return build_softmax_like_from_node_tiled(node, ctx, &input_shape);
}

}  // namespace gfx_plugin
}  // namespace ov
