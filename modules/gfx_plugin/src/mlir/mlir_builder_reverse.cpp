// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reverse.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

mlir::Type to_mlir_type(ov::element::Type et, mlir::MLIRContext& ctx) {
    switch (et) {
        case ov::element::f16: return mlir::Float16Type::get(&ctx);
        case ov::element::f32: return mlir::Float32Type::get(&ctx);
        case ov::element::i32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        case ov::element::i64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
        case ov::element::u32: return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned);
        case ov::element::u64: return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Unsigned);
        default: OPENVINO_THROW("Reverse MLIR: unsupported element type");
    }
}

std::vector<int64_t> get_const_axes(const std::shared_ptr<const ov::Node>& node) {
    auto c = ov::as_type_ptr<const ov::op::v0::Constant>(node);
    OPENVINO_ASSERT(c, "Reverse MLIR: axes input must be Constant");
    return c->cast_vector<int64_t>();
}

std::vector<int64_t> compute_strides(const ov::Shape& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    int64_t acc = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = acc;
        acc *= static_cast<int64_t>(shape[static_cast<size_t>(i)]);
    }
    return strides;
}

}  // namespace

mlir::ModuleOp build_mlir_reverse_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::op::v1::Reverse> rev;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto r = ov::as_type_ptr<const ov::op::v1::Reverse>(node)) {
            OPENVINO_ASSERT(!rev, "Reverse MLIR builder: expected single Reverse op");
            rev = r;
        }
    }
    OPENVINO_ASSERT(rev, "Reverse MLIR builder: Reverse op not found");
    OPENVINO_ASSERT(rev->get_mode() == ov::op::v1::Reverse::Mode::INDEX,
                    "Reverse MLIR: only INDEX mode supported");

    auto in_shape = rev->get_input_shape(0);
    auto out_shape = rev->get_output_shape(0);
    auto axes = get_const_axes(rev->get_input_node_shared_ptr(1));

    auto elem_ty = to_mlir_type(rev->get_output_element_type(0), ctx);
    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());
    auto in_ty = mlir::RankedTensorType::get(in_dims, elem_ty);
    auto out_ty = mlir::RankedTensorType::get(out_dims, elem_ty);

    const auto in_strides = compute_strides(in_shape);
    const auto out_strides = compute_strides(out_shape);
    const int64_t out_total = static_cast<int64_t>(ov::shape_size(out_shape));

    std::vector<int64_t> reverse_flags(in_shape.size(), 0);
    for (auto axis : axes) {
        if (axis < 0)
            axis += static_cast<int64_t>(in_shape.size());
        OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < in_shape.size(),
                        "Reverse MLIR: axis out of range");
        reverse_flags[static_cast<size_t>(axis)] = 1;
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "reverse_main",
                                              mb.getFunctionType({in_ty}, {out_ty}));
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto out_flat_ty = mlir::RankedTensorType::get({out_total}, elem_ty);
    auto in_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(ov::shape_size(in_shape))}, elem_ty);

    auto collapse = [&](size_t rank) {
        mlir::SmallVector<mlir::ReassociationIndices> reassoc;
        mlir::ReassociationIndices group;
        for (size_t i = 0; i < rank; ++i)
            group.push_back(static_cast<int64_t>(i));
        reassoc.push_back(group);
        return reassoc;
    };

    mlir::Value in_flat = func.getArgument(0);
    if (in_shape.size() > 1) {
        in_flat = b.create<mlir::tensor::CollapseShapeOp>(loc, in_flat_ty, in_flat, collapse(in_shape.size()));
    }
    mlir::Value out_flat = b.create<mlir::tensor::EmptyOp>(loc, mlir::ArrayRef<int64_t>{out_total}, elem_ty);

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto c_out = b.create<mlir::arith::ConstantIndexOp>(loc, out_total);

    auto loop = b.create<mlir::scf::ForOp>(loc, c0, c_out, c1, mlir::ValueRange{out_flat});
    {
        auto* body = loop.getBody();
        mlir::OpBuilder lb(body, body->begin());
        auto iv = loop.getInductionVar();
        auto acc = loop.getRegionIterArgs()[0];

        mlir::Value idx = iv;
        mlir::Value in_linear = lb.create<mlir::arith::ConstantIndexOp>(loc, 0);

        for (size_t d = 0; d < out_shape.size(); ++d) {
            auto stride = lb.create<mlir::arith::ConstantIndexOp>(loc, out_strides[d]);
            auto out_i = lb.create<mlir::arith::DivUIOp>(loc, idx, stride).getResult();
            idx = lb.create<mlir::arith::RemUIOp>(loc, idx, stride).getResult();

            mlir::Value in_i = out_i;
            if (reverse_flags[d]) {
                auto dim = lb.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape[d]));
                auto one = lb.create<mlir::arith::ConstantIndexOp>(loc, 1);
                auto last = lb.create<mlir::arith::SubIOp>(loc, dim, one).getResult();
                in_i = lb.create<mlir::arith::SubIOp>(loc, last, out_i).getResult();
            }

            auto in_stride = lb.create<mlir::arith::ConstantIndexOp>(loc, in_strides[d]);
            auto scaled = lb.create<mlir::arith::MulIOp>(loc, in_i, in_stride).getResult();
            in_linear = lb.create<mlir::arith::AddIOp>(loc, in_linear, scaled).getResult();
        }

        auto val = lb.create<mlir::tensor::ExtractOp>(loc, in_flat, mlir::ValueRange{in_linear}).getResult();
        auto updated = lb.create<mlir::tensor::InsertOp>(loc, val, acc, mlir::ValueRange{iv}).getResult();
        lb.create<mlir::scf::YieldOp>(loc, updated);
    }
    out_flat = loop.getResults()[0];

    mlir::Value out_val = out_flat;
    if (out_shape.size() > 1) {
        out_val = b.create<mlir::tensor::ExpandShapeOp>(loc, out_ty, out_flat, collapse(out_shape.size()));
    }

    b.create<mlir::func::ReturnOp>(loc, out_val);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
