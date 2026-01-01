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
#include "openvino/op/pad.hpp"

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
        default: OPENVINO_THROW("Pad MLIR: unsupported element type");
    }
}

std::vector<int64_t> get_const_i64(const std::shared_ptr<const ov::Node>& node) {
    auto c = ov::as_type_ptr<const ov::op::v0::Constant>(node);
    OPENVINO_ASSERT(c, "Pad MLIR: pads must be Constant");
    return c->cast_vector<int64_t>();
}

mlir::Value make_pad_value(mlir::OpBuilder& b, mlir::Location loc, mlir::Type elem_ty, const ov::Tensor& tensor) {
    if (auto fty = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
        double v = 0.0;
        if (tensor.get_element_type() == ov::element::f16 || tensor.get_element_type() == ov::element::bf16) {
            v = static_cast<double>(*tensor.data<const ov::float16>());
        } else {
            v = static_cast<double>(*tensor.data<const float>());
        }
        return b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(fty, v)).getResult();
    }
    if (auto ity = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
        int64_t v = 0;
        if (tensor.get_element_type().is_integral_number()) {
            switch (tensor.get_element_type()) {
                case ov::element::i32: v = *tensor.data<const int32_t>(); break;
                case ov::element::i64: v = *tensor.data<const int64_t>(); break;
                case ov::element::u32: v = static_cast<int64_t>(*tensor.data<const uint32_t>()); break;
                case ov::element::u64: v = static_cast<int64_t>(*tensor.data<const uint64_t>()); break;
                default: v = 0; break;
            }
        }
        return b.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(ity, v)).getResult();
    }
    OPENVINO_THROW("Pad MLIR: unsupported pad value type");
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

mlir::ModuleOp build_mlir_pad_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::Node> pad_node;
    std::shared_ptr<const ov::op::util::PadBase> pad_base;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto p = ov::as_type_ptr<const ov::op::v1::Pad>(node)) {
            OPENVINO_ASSERT(!pad_node, "Pad MLIR builder: expected single Pad");
            pad_node = node;
            pad_base = p;
        } else if (auto p = ov::as_type_ptr<const ov::op::v12::Pad>(node)) {
            OPENVINO_ASSERT(!pad_node, "Pad MLIR builder: expected single Pad");
            pad_node = node;
            pad_base = p;
        }
    }
    OPENVINO_ASSERT(pad_node && pad_base, "Pad MLIR builder: Pad op not found");
    OPENVINO_ASSERT(pad_base->get_pad_mode() == ov::op::PadMode::CONSTANT,
                    "Pad MLIR: only CONSTANT mode supported");

    auto pads_begin = get_const_i64(pad_node->get_input_node_shared_ptr(1));
    auto pads_end = get_const_i64(pad_node->get_input_node_shared_ptr(2));
    OPENVINO_ASSERT(pads_begin.size() == pads_end.size(),
                    "Pad MLIR: pads_begin/pads_end size mismatch");

    auto in_shape = pad_node->get_input_shape(0);
    auto out_shape = pad_node->get_output_shape(0);
    OPENVINO_ASSERT(in_shape.size() == pads_begin.size(), "Pad MLIR: pads rank mismatch");

    for (size_t i = 0; i < pads_begin.size(); ++i) {
        OPENVINO_ASSERT(pads_begin[i] >= 0 && pads_end[i] >= 0,
                        "Pad MLIR: negative pads are not supported yet");
    }

    auto elem_ty = to_mlir_type(pad_node->get_output_element_type(0), ctx);
    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());
    auto in_ty = mlir::RankedTensorType::get(in_dims, elem_ty);
    auto out_ty = mlir::RankedTensorType::get(out_dims, elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "pad_main",
                                              mb.getFunctionType({in_ty}, {out_ty}));
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    mlir::Value pad_val;
    if (pad_node->get_input_size() > 3) {
        auto pad_const = ov::as_type_ptr<const ov::op::v0::Constant>(pad_node->get_input_node_shared_ptr(3));
        OPENVINO_ASSERT(pad_const, "Pad MLIR: pad value must be Constant");
        pad_val = make_pad_value(b, loc, elem_ty, pad_const->get_tensor_view());
    } else {
        if (auto fty = mlir::dyn_cast<mlir::FloatType>(elem_ty)) {
            pad_val = b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(fty, 0.0)).getResult();
        } else if (auto ity = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
            pad_val = b.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(ity, 0)).getResult();
        } else {
            OPENVINO_THROW("Pad MLIR: unsupported pad value type");
        }
    }

    auto out_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(ov::shape_size(out_shape))}, elem_ty);
    auto in_flat_ty = mlir::RankedTensorType::get({static_cast<int64_t>(ov::shape_size(in_shape))}, elem_ty);

    auto collapse = [&](size_t rank) {
        mlir::SmallVector<mlir::ReassociationIndices> reassoc;
        mlir::ReassociationIndices group;
        for (size_t i = 0; i < rank; ++i)
            group.push_back(static_cast<int64_t>(i));
        reassoc.push_back(group);
        return reassoc;
    };

    auto in_strides = compute_strides(in_shape);
    auto out_strides = compute_strides(out_shape);
    const int64_t out_total = static_cast<int64_t>(ov::shape_size(out_shape));

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
        mlir::Value in_bounds = lb.create<mlir::arith::ConstantIntOp>(loc, 1, 1);

        for (size_t d = 0; d < out_shape.size(); ++d) {
            auto stride = lb.create<mlir::arith::ConstantIndexOp>(loc, out_strides[d]);
            auto out_i = lb.create<mlir::arith::DivUIOp>(loc, idx, stride).getResult();
            idx = lb.create<mlir::arith::RemUIOp>(loc, idx, stride).getResult();

            auto pad_b = lb.create<mlir::arith::ConstantIndexOp>(loc, pads_begin[d]);
            auto pad_e = lb.create<mlir::arith::ConstantIndexOp>(loc, pads_end[d]);
            auto in_dim = lb.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape[d]));
            auto start_ok = lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, out_i, pad_b);
            auto end_limit = lb.create<mlir::arith::AddIOp>(loc, pad_b, in_dim);
            auto end_ok = lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, out_i, end_limit);
            in_bounds = lb.create<mlir::arith::AndIOp>(loc, in_bounds, start_ok);
            in_bounds = lb.create<mlir::arith::AndIOp>(loc, in_bounds, end_ok);

            auto zero = lb.create<mlir::arith::ConstantIndexOp>(loc, 0);
            auto max_idx = lb.create<mlir::arith::SubIOp>(
                loc, in_dim, lb.create<mlir::arith::ConstantIndexOp>(loc, 1)).getResult();
            auto in_i_raw = lb.create<mlir::arith::SubIOp>(loc, out_i, pad_b).getResult();
            auto lt = lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, out_i, pad_b);
            auto ge = lb.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, out_i, end_limit);
            auto in_i = lb.create<mlir::arith::SelectOp>(loc, lt, zero, in_i_raw).getResult();
            in_i = lb.create<mlir::arith::SelectOp>(loc, ge, max_idx, in_i).getResult();
            auto in_stride = lb.create<mlir::arith::ConstantIndexOp>(loc, in_strides[d]);
            auto scaled = lb.create<mlir::arith::MulIOp>(loc, in_i, in_stride).getResult();
            in_linear = lb.create<mlir::arith::AddIOp>(loc, in_linear, scaled).getResult();
        }

        auto val = lb.create<mlir::tensor::ExtractOp>(loc, in_flat, mlir::ValueRange{in_linear}).getResult();
        auto selected = lb.create<mlir::arith::SelectOp>(loc, in_bounds, val, pad_val).getResult();
        auto updated = lb.create<mlir::tensor::InsertOp>(loc, selected, acc, mlir::ValueRange{iv}).getResult();
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
