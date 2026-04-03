// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/util/reduction_base.hpp"

#include <algorithm>
#include <limits>

namespace ov {
namespace gfx_plugin {

namespace {

enum class ReduceKind { Sum, Mean, Max, Min, Prod, L1, L2 };

std::vector<int64_t> compute_strides(const ov::Shape& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    int64_t acc = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = acc;
        acc *= static_cast<int64_t>(shape[static_cast<size_t>(i)]);
    }
    return strides;
}

mlir::Value make_init_value(mlir::OpBuilder& b, mlir::Location loc, mlir::Type elem_ty, ReduceKind kind) {
    if (mlir::isa<mlir::FloatType>(elem_ty)) {
        double init = 0.0;
        switch (kind) {
            case ReduceKind::Sum:
            case ReduceKind::Mean:
            case ReduceKind::L1:
            case ReduceKind::L2: init = 0.0; break;
            case ReduceKind::Prod: init = 1.0; break;
            case ReduceKind::Max: init = -std::numeric_limits<double>::infinity(); break;
            case ReduceKind::Min: init = std::numeric_limits<double>::infinity(); break;
        }
        return b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(elem_ty, init)).getResult();
    }
    auto ity = mlir::cast<mlir::IntegerType>(elem_ty);
    int64_t init = 0;
    switch (kind) {
        case ReduceKind::Sum:
        case ReduceKind::Mean:
        case ReduceKind::L1:
        case ReduceKind::L2: init = 0; break;
        case ReduceKind::Prod: init = 1; break;
        case ReduceKind::Max:
            init = ity.isUnsigned() ? 0 : std::numeric_limits<int64_t>::min();
            break;
        case ReduceKind::Min:
            init = ity.isUnsigned() ? static_cast<int64_t>(std::numeric_limits<uint64_t>::max())
                                    : std::numeric_limits<int64_t>::max();
            break;
    }
    return b.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(elem_ty, init)).getResult();
}

mlir::Value reduce_update(mlir::OpBuilder& b,
                          mlir::Location loc,
                          mlir::Type elem_ty,
                          ReduceKind kind,
                          mlir::Value acc,
                          mlir::Value val) {
    const bool is_float = mlir::isa<mlir::FloatType>(elem_ty);
    auto zero = [&]() {
        if (is_float) {
            return b.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(elem_ty, 0.0)).getResult();
        }
        return b.create<mlir::arith::ConstantOp>(loc, mlir::IntegerAttr::get(elem_ty, 0)).getResult();
    };

    auto abs_val = [&](mlir::Value v) {
        auto z = zero();
        if (is_float) {
            auto neg = b.create<mlir::arith::SubFOp>(loc, z, v).getResult();
            auto lt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, v, z).getResult();
            return b.create<mlir::arith::SelectOp>(loc, lt, neg, v).getResult();
        }
        auto neg = b.create<mlir::arith::SubIOp>(loc, z, v).getResult();
        auto lt = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, v, z).getResult();
        return b.create<mlir::arith::SelectOp>(loc, lt, neg, v).getResult();
    };

    switch (kind) {
        case ReduceKind::Sum:
        case ReduceKind::Mean:
            return is_float ? b.create<mlir::arith::AddFOp>(loc, acc, val).getResult()
                            : b.create<mlir::arith::AddIOp>(loc, acc, val).getResult();
        case ReduceKind::Prod:
            return is_float ? b.create<mlir::arith::MulFOp>(loc, acc, val).getResult()
                            : b.create<mlir::arith::MulIOp>(loc, acc, val).getResult();
        case ReduceKind::Max: {
            if (is_float) {
                auto lt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, acc, val).getResult();
                return b.create<mlir::arith::SelectOp>(loc, lt, val, acc).getResult();
            }
            auto lt = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, acc, val).getResult();
            return b.create<mlir::arith::SelectOp>(loc, lt, val, acc).getResult();
        }
        case ReduceKind::Min: {
            if (is_float) {
                auto lt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, val, acc).getResult();
                return b.create<mlir::arith::SelectOp>(loc, lt, val, acc).getResult();
            }
            auto lt = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, val, acc).getResult();
            return b.create<mlir::arith::SelectOp>(loc, lt, val, acc).getResult();
        }
        case ReduceKind::L1: {
            auto a = abs_val(val);
            return is_float ? b.create<mlir::arith::AddFOp>(loc, acc, a).getResult()
                            : b.create<mlir::arith::AddIOp>(loc, acc, a).getResult();
        }
        case ReduceKind::L2: {
            auto mul = is_float ? b.create<mlir::arith::MulFOp>(loc, val, val).getResult()
                                : b.create<mlir::arith::MulIOp>(loc, val, val).getResult();
            return is_float ? b.create<mlir::arith::AddFOp>(loc, acc, mul).getResult()
                            : b.create<mlir::arith::AddIOp>(loc, acc, mul).getResult();
        }
    }
    return acc;
}

mlir::ModuleOp build_reduce_impl(const std::shared_ptr<const ov::Model>& model,
                                 mlir::MLIRContext& ctx,
                                 ReduceKind kind) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect, mlir::math::MathDialect>();

    std::shared_ptr<const ov::Node> reduce_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v1::ReduceSum>(node) ||
            ov::as_type_ptr<const ov::op::v1::ReduceMean>(node) ||
            ov::as_type_ptr<const ov::op::v1::ReduceMax>(node) ||
            ov::as_type_ptr<const ov::op::v1::ReduceMin>(node) ||
            ov::as_type_ptr<const ov::op::v1::ReduceProd>(node) ||
            ov::as_type_ptr<const ov::op::v4::ReduceL1>(node) ||
            ov::as_type_ptr<const ov::op::v4::ReduceL2>(node)) {
            OPENVINO_ASSERT(!reduce_node, "Reduce MLIR builder: expected single Reduce op");
            reduce_node = node;
        }
    }
    OPENVINO_ASSERT(reduce_node, "Reduce MLIR builder: Reduce op not found");

    auto base = ov::as_type_ptr<const ov::op::util::ReductionBase>(reduce_node);
    OPENVINO_ASSERT(base, "Reduce MLIR: invalid reduce op");
    OPENVINO_ASSERT(base->reduction_axes_constant(), "Reduce MLIR: reduction axes must be constant");
    const auto axes_set = base->get_reduction_axes();
    const bool keep_dims = base->get_keep_dims();

    auto in_shape = reduce_node->get_input_shape(0);
    auto out_shape = reduce_node->get_output_shape(0);
    OPENVINO_ASSERT(!in_shape.empty(), "Reduce MLIR: input shape must be static");

    std::vector<size_t> axes(axes_set.begin(), axes_set.end());
    std::sort(axes.begin(), axes.end());

    const size_t in_rank = in_shape.size();
    for (auto axis : axes) {
        OPENVINO_ASSERT(axis < in_rank, "Reduce MLIR: axis out of range");
    }

    auto elem_ty = to_mlir_type(reduce_node->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/true);
    mlir::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    mlir::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());
    auto in_ty = mlir::RankedTensorType::get(in_dims, elem_ty);
    auto out_ty = mlir::RankedTensorType::get(out_dims, elem_ty);

    auto in_strides = compute_strides(in_shape);
    auto out_strides = compute_strides(out_shape);

    std::vector<size_t> non_reduced_axes;
    for (size_t i = 0; i < in_rank; ++i) {
        if (!std::binary_search(axes.begin(), axes.end(), i)) {
            non_reduced_axes.push_back(i);
        }
    }

    std::vector<int64_t> reduce_shape;
    reduce_shape.reserve(axes.size());
    for (auto axis : axes) {
        reduce_shape.push_back(static_cast<int64_t>(in_shape[axis]));
    }
    auto reduce_strides = compute_strides(ov::Shape(reduce_shape.begin(), reduce_shape.end()));
    const int64_t reduce_total = reduce_shape.empty() ? 1 : static_cast<int64_t>(ov::shape_size(reduce_shape));
    const int64_t out_total = static_cast<int64_t>(ov::shape_size(out_shape));

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "reduce_main",
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
        auto acc_tensor = loop.getRegionIterArgs()[0];

        mlir::Value idx = iv;
        std::vector<mlir::Value> out_indices;
        out_indices.reserve(out_shape.size());
        for (size_t d = 0; d < out_shape.size(); ++d) {
            auto stride = lb.create<mlir::arith::ConstantIndexOp>(loc, out_strides[d]);
            auto out_i = lb.create<mlir::arith::DivUIOp>(loc, idx, stride).getResult();
            idx = lb.create<mlir::arith::RemUIOp>(loc, idx, stride).getResult();
            out_indices.push_back(out_i);
        }

        mlir::Value base = lb.create<mlir::arith::ConstantIndexOp>(loc, 0);
        size_t out_pos = 0;
        for (size_t i = 0; i < in_rank; ++i) {
            if (std::binary_search(axes.begin(), axes.end(), i)) {
                continue;
            }
            size_t od = keep_dims ? i : out_pos++;
            auto in_stride = lb.create<mlir::arith::ConstantIndexOp>(loc, in_strides[i]);
            auto scaled = lb.create<mlir::arith::MulIOp>(loc, out_indices[od], in_stride).getResult();
            base = lb.create<mlir::arith::AddIOp>(loc, base, scaled).getResult();
        }

        auto init = make_init_value(lb, loc, elem_ty, kind);
        auto c_red = lb.create<mlir::arith::ConstantIndexOp>(loc, reduce_total);
        auto red_loop = lb.create<mlir::scf::ForOp>(loc, c0, c_red, c1, mlir::ValueRange{init});
        {
            auto* rbody = red_loop.getBody();
            mlir::OpBuilder rb(rbody, rbody->begin());
            auto r_iv = red_loop.getInductionVar();
            auto r_acc = red_loop.getRegionIterArgs()[0];

            mlir::Value r_idx = r_iv;
            mlir::Value in_linear = base;
            for (size_t j = 0; j < axes.size(); ++j) {
                auto stride = rb.create<mlir::arith::ConstantIndexOp>(loc, reduce_strides[j]);
                auto r_i = rb.create<mlir::arith::DivUIOp>(loc, r_idx, stride).getResult();
                r_idx = rb.create<mlir::arith::RemUIOp>(loc, r_idx, stride).getResult();
                auto in_stride = rb.create<mlir::arith::ConstantIndexOp>(loc, in_strides[axes[j]]);
                auto scaled = rb.create<mlir::arith::MulIOp>(loc, r_i, in_stride).getResult();
                in_linear = rb.create<mlir::arith::AddIOp>(loc, in_linear, scaled).getResult();
            }

            auto val = rb.create<mlir::tensor::ExtractOp>(loc, in_flat, mlir::ValueRange{in_linear}).getResult();
            auto updated = reduce_update(rb, loc, elem_ty, kind, r_acc, val);
            rb.create<mlir::scf::YieldOp>(loc, updated);
        }
        mlir::Value reduced = red_loop.getResults()[0];

        if (kind == ReduceKind::Mean) {
            OPENVINO_ASSERT(mlir::isa<mlir::FloatType>(elem_ty), "ReduceMean MLIR: only float supported");
            auto denom = lb.create<mlir::arith::ConstantOp>(loc, mlir::FloatAttr::get(elem_ty, static_cast<double>(reduce_total)));
            reduced = lb.create<mlir::arith::DivFOp>(loc, reduced, denom).getResult();
        } else if (kind == ReduceKind::L2) {
            OPENVINO_ASSERT(mlir::isa<mlir::FloatType>(elem_ty), "ReduceL2 MLIR: only float supported");
            reduced = lb.create<mlir::math::SqrtOp>(loc, reduced).getResult();
        }

        auto updated_out = lb.create<mlir::tensor::InsertOp>(loc, reduced, acc_tensor, mlir::ValueRange{iv}).getResult();
        lb.create<mlir::scf::YieldOp>(loc, updated_out);
    }
    out_flat = loop.getResults()[0];

    mlir::Value out_val = out_flat;
    if (out_shape.size() > 1) {
        out_val = b.create<mlir::tensor::ExpandShapeOp>(loc, out_ty, out_flat, collapse(out_shape.size()));
    }

    b.create<mlir::func::ReturnOp>(loc, out_val);
    return module;
}

}  // namespace

mlir::ModuleOp build_mlir_reducesum_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_reduce_impl(model, ctx, ReduceKind::Sum);
}

mlir::ModuleOp build_mlir_reducemean_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_reduce_impl(model, ctx, ReduceKind::Mean);
}

mlir::ModuleOp build_mlir_reducemax_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_reduce_impl(model, ctx, ReduceKind::Max);
}

mlir::ModuleOp build_mlir_reducemin_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_reduce_impl(model, ctx, ReduceKind::Min);
}

mlir::ModuleOp build_mlir_reduceprod_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_reduce_impl(model, ctx, ReduceKind::Prod);
}

mlir::ModuleOp build_mlir_reducel1_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_reduce_impl(model, ctx, ReduceKind::L1);
}

mlir::ModuleOp build_mlir_reducel2_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_reduce_impl(model, ctx, ReduceKind::L2);
}

}  // namespace gfx_plugin
}  // namespace ov
