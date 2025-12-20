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

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"

namespace ov {
namespace metal_plugin {

namespace {
mlir::Type to_mlir_type(ov::element::Type et, mlir::MLIRContext& ctx, bool fallback_f32 = false) {
    switch (et) {
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        case ov::element::i32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        case ov::element::i64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
        default:
            if (fallback_f32) return mlir::Float32Type::get(&ctx);
            OPENVINO_THROW("Gather MLIR: unsupported element type");
    }
}

mlir::SmallVector<int64_t> to_shape(const ov::PartialShape& ps) {
    mlir::SmallVector<int64_t> dims;
    dims.reserve(ps.rank().get_length());
    for (const auto& d : ps) {
        dims.push_back(d.is_dynamic() ? mlir::ShapedType::kDynamic
                                      : static_cast<int64_t>(d.get_length()));
    }
    return dims;
}
}  // namespace

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

    auto in_shape = to_shape(gather_node->get_input_partial_shape(0));
    auto idx_shape = to_shape(gather_node->get_input_partial_shape(1));
    auto out_shape = to_shape(gather_node->get_output_partial_shape(0));

    auto elem_ty = to_mlir_type(gather_node->get_output_element_type(0), ctx);
    auto idx_ty = to_mlir_type(gather_node->get_input_element_type(1), ctx, /*fallback_f32=*/true);

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

    auto empty_out = b.create<mlir::tensor::EmptyOp>(loc, out_shape, elem_ty);

    // Use a flattened implementation for gather:
    // out_flat[gid] = data_flat[outer*axis_dim*inner + idx*inner + inner_idx]
    auto rank_data = gather_node->get_input_shape(0).size();
    OPENVINO_ASSERT(rank_data > 0, "Gather MLIR: input rank must be static");
    auto data_shape = gather_node->get_input_shape(0);
    auto idx_shape_static = gather_node->get_input_shape(1);
    auto out_shape_static = gather_node->get_output_shape(0);
    OPENVINO_ASSERT(!data_shape.empty() && !idx_shape_static.empty() && !out_shape_static.empty(),
                    "Gather MLIR: requires static shapes");

    auto axis_c = ov::as_type_ptr<const ov::op::v0::Constant>(gather_node->get_input_node_shared_ptr(2));
    OPENVINO_ASSERT(axis_c, "Gather MLIR: axis must be constant");
    auto axis_v = axis_c->cast_vector<int64_t>();
    OPENVINO_ASSERT(axis_v.size() == 1, "Gather MLIR: axis must be scalar");
    int64_t axis = axis_v[0];
    if (axis < 0)
        axis += static_cast<int64_t>(data_shape.size());
    OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < data_shape.size(), "Gather MLIR: axis out of range");

    auto product = [](const ov::Shape& s, size_t start, size_t end) {
        uint64_t prod = 1;
        for (size_t i = start; i < end; ++i) prod *= s[i];
        return prod;
    };
    uint64_t outer = product(data_shape, 0, static_cast<size_t>(axis));
    uint64_t inner = product(data_shape, static_cast<size_t>(axis) + 1, data_shape.size());
    uint64_t axis_dim = data_shape[static_cast<size_t>(axis)];
    uint64_t indices_count = ov::shape_size(idx_shape_static);
    uint64_t total = outer * indices_count * inner;

    auto idx_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(indices_count)}, idx_ty);
    auto data_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(outer * axis_dim * inner)}, elem_ty);
    auto out_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(total)}, elem_ty);

    auto collapse_reassoc = [&](size_t rank) {
        mlir::SmallVector<mlir::ReassociationIndices> reassoc;
        mlir::ReassociationIndices group;
        for (size_t i = 0; i < rank; ++i) group.push_back(static_cast<int64_t>(i));
        reassoc.push_back(group);
        return reassoc;
    };

    mlir::Value data_flat = func.getArgument(0);
    if (rank_data > 1) {
        data_flat = b.create<mlir::tensor::CollapseShapeOp>(loc, data_flat_ty, func.getArgument(0),
                                                            collapse_reassoc(rank_data));
    }
    mlir::Value idx_flat = func.getArgument(1);
    if (idx_shape_static.size() > 1) {
        idx_flat = b.create<mlir::tensor::CollapseShapeOp>(loc, idx_flat_ty, func.getArgument(1),
                                                           collapse_reassoc(idx_shape_static.size()));
    }
    mlir::Value out_flat = b.create<mlir::tensor::EmptyOp>(loc, mlir::ArrayRef<int64_t>({static_cast<int64_t>(total)}),
                                                           elem_ty);

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto c_total = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(total));
    auto c_inner = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(inner));
    auto c_indices = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(indices_count));
    auto c_axis_dim = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(axis_dim));

    auto loop = b.create<mlir::scf::ForOp>(loc, c0, c_total, c1, mlir::ValueRange{out_flat});
    {
        auto* body = loop.getBody();
        mlir::OpBuilder lb(body, body->begin());
        auto iv = loop.getInductionVar();
        auto acc = loop.getRegionIterArgs()[0];

        auto inner_idx = lb.create<mlir::arith::RemUIOp>(loc, iv, c_inner).getResult();
        auto tmp = lb.create<mlir::arith::DivUIOp>(loc, iv, c_inner).getResult();
        auto idx_idx = lb.create<mlir::arith::RemUIOp>(loc, tmp, c_indices).getResult();
        auto outer_idx = lb.create<mlir::arith::DivUIOp>(loc, tmp, c_indices).getResult();

        auto idx_val = lb.create<mlir::tensor::ExtractOp>(loc, idx_flat, mlir::ValueRange{idx_idx}).getResult();
        mlir::Value idx_i64 = idx_val;
        if (idx_val.getType().isInteger(32)) {
            idx_i64 = lb.create<mlir::arith::ExtSIOp>(loc, mlir::IntegerType::get(&ctx, 64), idx_val).getResult();
        }

        auto axis_dim_i64 =
            lb.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(axis_dim), 64).getResult();
        auto zero_i64 = lb.create<mlir::arith::ConstantIntOp>(loc, 0, 64).getResult();
        auto max_i64 =
            lb.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(axis_dim - 1), 64).getResult();

        auto neg_pred =
            lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, idx_i64, zero_i64).getResult();
        auto idx_plus = lb.create<mlir::arith::AddIOp>(loc, idx_i64, axis_dim_i64).getResult();
        auto idx_fixed = lb.create<mlir::arith::SelectOp>(loc, neg_pred, idx_plus, idx_i64).getResult();

        auto lt0 =
            lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, idx_fixed, zero_i64).getResult();
        auto gtmax =
            lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, idx_fixed, max_i64).getResult();
        auto idx_clamped = lb.create<mlir::arith::SelectOp>(loc, lt0, zero_i64, idx_fixed).getResult();
        idx_clamped = lb.create<mlir::arith::SelectOp>(loc, gtmax, max_i64, idx_clamped).getResult();

        auto idx_index =
            lb.create<mlir::arith::IndexCastOp>(loc, lb.getIndexType(), idx_clamped).getResult();

        auto outer_mul =
            lb.create<mlir::arith::MulIOp>(loc, outer_idx,
                                           lb.create<mlir::arith::MulIOp>(loc, c_axis_dim, c_inner).getResult())
                .getResult();
        auto axis_mul = lb.create<mlir::arith::MulIOp>(loc, idx_index, c_inner).getResult();
        auto in_index =
            lb.create<mlir::arith::AddIOp>(loc, outer_mul,
                                           lb.create<mlir::arith::AddIOp>(loc, axis_mul, inner_idx).getResult())
                .getResult();

        auto data_val =
            lb.create<mlir::tensor::ExtractOp>(loc, data_flat, mlir::ValueRange{in_index}).getResult();
        auto updated =
            lb.create<mlir::tensor::InsertOp>(loc, data_val, acc, mlir::ValueRange{iv}).getResult();
        lb.create<mlir::scf::YieldOp>(loc, updated);
    }
    out_flat = loop.getResults()[0];

    mlir::Value final_out = out_flat;
    if (out_shape_static.size() > 1) {
        auto reassoc = collapse_reassoc(out_shape_static.size());
        final_out = b.create<mlir::tensor::ExpandShapeOp>(loc, out_tensor_ty, out_flat, reassoc);
    }

    b.create<mlir::func::ReturnOp>(loc, final_out);
    return module;
}

}  // namespace metal_plugin
}  // namespace ov
