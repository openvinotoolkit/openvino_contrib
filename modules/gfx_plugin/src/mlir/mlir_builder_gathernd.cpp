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

#include "mlir/gfx_mlir_type_utils.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather_nd.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
uint64_t product(const ov::Shape& s, size_t start, size_t end) {
    uint64_t prod = 1;
    for (size_t i = start; i < end; ++i) prod *= s[i];
    return prod;
}
}  // namespace

mlir::ModuleOp build_mlir_gathernd_from_model(const std::shared_ptr<const ov::Model>& model,
                                              mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::Node> gather_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v5::GatherND>(node) ||
            ov::as_type_ptr<const ov::op::v8::GatherND>(node)) {
            OPENVINO_ASSERT(!gather_node, "GatherND MLIR builder: expected single GatherND");
            gather_node = node;
        }
    }
    OPENVINO_ASSERT(gather_node, "GatherND MLIR builder: GatherND op not found");

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
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "gathernd_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto data_shape = gather_node->get_input_shape(0);
    auto indices_shape = gather_node->get_input_shape(1);
    auto out_shape_static = gather_node->get_output_shape(0);

    OPENVINO_ASSERT(!data_shape.empty(), "GatherND MLIR: data shape must be static");
    OPENVINO_ASSERT(!indices_shape.empty(), "GatherND MLIR: indices shape must be static");

    const size_t data_rank = data_shape.size();
    const size_t idx_rank = indices_shape.size();
    const size_t k = indices_shape.back();
    OPENVINO_ASSERT(k >= 1 && k <= data_rank, "GatherND MLIR: invalid indices last dim");

    auto base = ov::as_type_ptr<const ov::op::util::GatherNDBase>(gather_node);
    OPENVINO_ASSERT(base, "GatherND MLIR: expected GatherNDBase");
    OPENVINO_ASSERT(base->get_batch_dims() == 0, "GatherND MLIR: batch_dims not supported");

    const uint64_t num_indices = (idx_rank > 1) ? product(indices_shape, 0, idx_rank - 1) : 1;
    const uint64_t inner = product(data_shape, k, data_rank);
    const uint64_t total = num_indices * inner;

    auto data_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(ov::shape_size(data_shape))}, elem_ty);
    mlir::Value data_flat = func.getArgument(0);
    if (data_rank > 1) {
        mlir::SmallVector<mlir::ReassociationIndices> reassoc;
        mlir::ReassociationIndices group;
        for (size_t i = 0; i < data_rank; ++i) group.push_back(static_cast<int64_t>(i));
        reassoc.push_back(group);
        data_flat = b.create<mlir::tensor::CollapseShapeOp>(loc, data_flat_ty, func.getArgument(0), reassoc);
    }

    mlir::Value indices_flat = func.getArgument(1);
    mlir::Value indices_2d = {};
    mlir::RankedTensorType idx2d_ty;
    if (idx_rank > 1) {
        idx2d_ty = mlir::RankedTensorType::get({static_cast<int64_t>(num_indices), static_cast<int64_t>(k)}, idx_ty);
        if (idx_rank == 2) {
            indices_2d = func.getArgument(1);
        } else {
            mlir::SmallVector<mlir::ReassociationIndices> reassoc;
            mlir::ReassociationIndices group0;
            for (size_t i = 0; i + 1 < idx_rank; ++i) group0.push_back(static_cast<int64_t>(i));
            mlir::ReassociationIndices group1{static_cast<int64_t>(idx_rank - 1)};
            reassoc.push_back(group0);
            reassoc.push_back(group1);
            indices_2d = b.create<mlir::tensor::CollapseShapeOp>(loc, idx2d_ty, func.getArgument(1), reassoc);
        }
    }

    auto out_flat = b.create<mlir::tensor::EmptyOp>(loc, mlir::ArrayRef<int64_t>({static_cast<int64_t>(total)}),
                                                    elem_ty);

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto c_total = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(total));
    auto c_inner = b.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(inner));

    auto loop = b.create<mlir::scf::ForOp>(loc, c0, c_total, c1, mlir::ValueRange{out_flat});
    {
        auto* body = loop.getBody();
        mlir::OpBuilder lb(body, body->begin());
        auto iv = loop.getInductionVar();
        auto acc = loop.getRegionIterArgs()[0];

        auto inner_idx = lb.create<mlir::arith::RemUIOp>(loc, iv, c_inner).getResult();
        auto idx_pos = lb.create<mlir::arith::DivUIOp>(loc, iv, c_inner).getResult();

        mlir::Value base_i64 = lb.create<mlir::arith::ConstantIntOp>(loc, 0, 64).getResult();
        for (size_t i = 0; i < k; ++i) {
            mlir::Value idx_val;
            if (idx_rank == 1) {
                auto idx_i = lb.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(i)).getResult();
                idx_val = lb.create<mlir::tensor::ExtractOp>(loc, indices_flat, mlir::ValueRange{idx_i}).getResult();
            } else {
                auto idx_i = lb.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(i)).getResult();
                idx_val = lb.create<mlir::tensor::ExtractOp>(loc, indices_2d,
                                                            mlir::ValueRange{idx_pos, idx_i}).getResult();
            }
            mlir::Value idx_i64 = idx_val;
            if (idx_val.getType().isInteger(32)) {
                idx_i64 = lb.create<mlir::arith::ExtSIOp>(loc, mlir::IntegerType::get(&ctx, 64), idx_val).getResult();
            }

            auto dim_i64 =
                lb.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(data_shape[i]), 64).getResult();
            auto zero_i64 = lb.create<mlir::arith::ConstantIntOp>(loc, 0, 64).getResult();
            auto max_i64 =
                lb.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(data_shape[i] - 1), 64).getResult();

            auto neg_pred =
                lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, idx_i64, zero_i64).getResult();
            auto idx_plus = lb.create<mlir::arith::AddIOp>(loc, idx_i64, dim_i64).getResult();
            auto idx_fixed = lb.create<mlir::arith::SelectOp>(loc, neg_pred, idx_plus, idx_i64).getResult();

            auto lt0 =
                lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, idx_fixed, zero_i64).getResult();
            auto gtmax =
                lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, idx_fixed, max_i64).getResult();
            auto idx_clamped = lb.create<mlir::arith::SelectOp>(loc, lt0, zero_i64, idx_fixed).getResult();
            idx_clamped = lb.create<mlir::arith::SelectOp>(loc, gtmax, max_i64, idx_clamped).getResult();

            uint64_t stride = product(data_shape, i + 1, data_rank);
            auto stride_i64 =
                lb.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(stride), 64).getResult();
            auto mul = lb.create<mlir::arith::MulIOp>(loc, idx_clamped, stride_i64).getResult();
            base_i64 = lb.create<mlir::arith::AddIOp>(loc, base_i64, mul).getResult();
        }

        auto base_idx = lb.create<mlir::arith::IndexCastOp>(loc, lb.getIndexType(), base_i64).getResult();
        auto in_index = lb.create<mlir::arith::AddIOp>(loc, base_idx, inner_idx).getResult();
        auto data_val =
            lb.create<mlir::tensor::ExtractOp>(loc, data_flat, mlir::ValueRange{in_index}).getResult();
        auto updated =
            lb.create<mlir::tensor::InsertOp>(loc, data_val, acc, mlir::ValueRange{iv}).getResult();
        lb.create<mlir::scf::YieldOp>(loc, updated);
    }
    auto out_flat_val = loop.getResults()[0];

    mlir::Value final_out = out_flat_val;
    if (out_shape_static.size() > 1) {
        mlir::SmallVector<mlir::ReassociationIndices> reassoc;
        mlir::ReassociationIndices group;
        for (size_t i = 0; i < out_shape_static.size(); ++i) group.push_back(static_cast<int64_t>(i));
        reassoc.push_back(group);
        final_out = b.create<mlir::tensor::ExpandShapeOp>(loc, out_tensor_ty, out_flat_val, reassoc);
    }

    b.create<mlir::func::ReturnOp>(loc, final_out);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
