// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "openvino/op/softmax.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/core/model.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace ov {
namespace metal_plugin {

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

mlir::ModuleOp build_softmax_like_from_node(const std::shared_ptr<const ov::Node>& sm, mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect>();
    const auto shape = sm->get_input_shape(0);
    auto f32 = mlir::Float32Type::get(&ctx);
    mlir::SmallVector<int64_t> dims(shape.begin(), shape.end());
    auto ty = mlir::MemRefType::get(dims, f32);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func_type = mb.getFunctionType({ty}, {ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "softmax_main", func_type);
    func.addEntryBlock();
    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);

    int64_t axis = -1;
    if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(sm)) axis = s1->get_axis();
    else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(sm)) axis = s8->get_axis();
    else if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(sm)) axis = ls->get_axis();
    else OPENVINO_THROW("Softmax builder: unsupported op kind");
    if (axis < 0) axis += static_cast<int64_t>(shape.size());
    int64_t rows = 1, cols = shape.at(axis), inner = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) rows *= shape[i];
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= shape[i];

    auto out_alloc = b.create<mlir::memref::AllocOp>(loc, ty);
    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto rows_c = b.create<mlir::arith::ConstantIndexOp>(loc, rows);
    auto cols_c = b.create<mlir::arith::ConstantIndexOp>(loc, cols);
    auto inner_c = b.create<mlir::arith::ConstantIndexOp>(loc, inner);

    auto for_row = b.create<mlir::scf::ForOp>(loc, c0, rows_c, c1);
    auto brow = mlir::OpBuilder::atBlockBegin(for_row.getBody());
    mlir::scf::ForOp for_inner;
    if (inner > 1) {
        for_inner = brow.create<mlir::scf::ForOp>(loc, c0, inner_c, c1);
    }
    auto binner = inner > 1 ? mlir::OpBuilder::atBlockBegin(for_inner.getBody()) : brow;
    auto for_col = binner.create<mlir::scf::ForOp>(loc, c0, cols_c, c1);
    auto bcol = mlir::OpBuilder::atBlockBegin(for_col.getBody());

    mlir::Value inner_idx = inner > 1 ? for_inner.getInductionVar() : c0;
    auto mul1 = bcol.create<mlir::arith::MulIOp>(loc, for_row.getInductionVar(), cols_c);
    auto add1 = bcol.create<mlir::arith::AddIOp>(loc, mul1, for_col.getInductionVar());
    auto mul2 = bcol.create<mlir::arith::MulIOp>(loc, add1, inner_c);
    auto flat = bcol.create<mlir::arith::AddIOp>(loc, mul2, inner_idx);

    auto val = bcol.create<mlir::memref::LoadOp>(loc, func.getArgument(0), mlir::ValueRange{flat});
    bcol.create<mlir::memref::StoreOp>(loc, val, out_alloc, mlir::ValueRange{flat});
    bcol.create<mlir::scf::YieldOp>(loc);
    if (inner > 1) mlir::OpBuilder::atBlockEnd(for_inner.getBody()).create<mlir::scf::YieldOp>(loc);
    mlir::OpBuilder::atBlockEnd(for_row.getBody()).create<mlir::scf::YieldOp>(loc);

    b.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{out_alloc});
    return module;
}
}  // namespace

mlir::ModuleOp build_mlir_softmax_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx) {
    auto sm = find_single_softmax(model);
    return build_softmax_like_from_node(sm, ctx);
}

mlir::ModuleOp build_mlir_logsoftmax_from_model(const std::shared_ptr<const ov::Model>& model,
                                                mlir::MLIRContext& ctx) {
    auto sm = find_single_logsoftmax(model);
    return build_softmax_like_from_node(sm, ctx);
}

}  // namespace metal_plugin
}  // namespace ov
