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

struct ReduceOpInfo {
    ov::AxisSet axes;
    bool keep_dims = false;
};

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

    auto get_reduce_info = [&](const std::shared_ptr<const ov::Node>& node) -> std::optional<ReduceOpInfo> {
        if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceSum>(node)) {
            OPENVINO_ASSERT(reduce->reduction_axes_constant(), "Reduce MLIR: reduction axes must be constant");
            return ReduceOpInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
        }
        if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMean>(node)) {
            OPENVINO_ASSERT(reduce->reduction_axes_constant(), "Reduce MLIR: reduction axes must be constant");
            return ReduceOpInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
        }
        if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMax>(node)) {
            OPENVINO_ASSERT(reduce->reduction_axes_constant(), "Reduce MLIR: reduction axes must be constant");
            return ReduceOpInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
        }
        if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMin>(node)) {
            OPENVINO_ASSERT(reduce->reduction_axes_constant(), "Reduce MLIR: reduction axes must be constant");
            return ReduceOpInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
        }
        if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceProd>(node)) {
            OPENVINO_ASSERT(reduce->reduction_axes_constant(), "Reduce MLIR: reduction axes must be constant");
            return ReduceOpInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
        }
        if (auto reduce = ov::as_type_ptr<const ov::op::v4::ReduceL1>(node)) {
            OPENVINO_ASSERT(reduce->reduction_axes_constant(), "Reduce MLIR: reduction axes must be constant");
            return ReduceOpInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
        }
        if (auto reduce = ov::as_type_ptr<const ov::op::v4::ReduceL2>(node)) {
            OPENVINO_ASSERT(reduce->reduction_axes_constant(), "Reduce MLIR: reduction axes must be constant");
            return ReduceOpInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
        }
        return std::nullopt;
    };

    const auto info = get_reduce_info(reduce_node);
    OPENVINO_ASSERT(info.has_value(), "Reduce MLIR: invalid reduce op");
    const auto axes_set = info->axes;
    const bool keep_dims = info->keep_dims;

    const auto in_pshape = reduce_node->get_input_partial_shape(0);
    const auto out_pshape = reduce_node->get_output_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && out_pshape.rank().is_static(),
                    "Reduce MLIR: ranks must be static");

    std::vector<size_t> axes(axes_set.begin(), axes_set.end());
    std::sort(axes.begin(), axes.end());

    const size_t in_rank = static_cast<size_t>(in_pshape.rank().get_length());
    for (auto axis : axes) {
        OPENVINO_ASSERT(axis < in_rank, "Reduce MLIR: axis out of range");
    }

    auto elem_ty = to_mlir_type(reduce_node->get_output_element_type(0),
                                ctx,
                                /*fallback_f32=*/false,
                                /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true,
                                /*allow_bf16=*/false,
                                /*allow_boolean=*/false,
                                /*signless_integers=*/true);
    mlir::SmallVector<int64_t> in_dims = to_shape(in_pshape);
    mlir::SmallVector<int64_t> out_dims = to_shape(out_pshape);
    auto in_ty = mlir::RankedTensorType::get(in_dims, elem_ty);
    auto out_ty = mlir::RankedTensorType::get(out_dims, elem_ty);

    std::vector<size_t> non_reduced_axes;
    for (size_t i = 0; i < in_rank; ++i) {
        if (!std::binary_search(axes.begin(), axes.end(), i)) {
            non_reduced_axes.push_back(i);
        }
    }

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "reduce_main",
                                              mb.getFunctionType({in_ty}, {out_ty}));
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());
    llvm::SmallVector<mlir::Value> out_dyn_dims;
    out_dyn_dims.reserve(out_dims.size());
    size_t out_pos = 0;
    for (size_t i = 0; i < in_rank; ++i) {
        if (std::binary_search(axes.begin(), axes.end(), i)) {
            continue;
        }
        size_t od = keep_dims ? i : out_pos++;
        if (out_dims[od] == mlir::ShapedType::kDynamic) {
            out_dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(0), static_cast<int64_t>(i)).getResult());
        }
    }

    auto generated = mlir::tensor::GenerateOp::create(
        b,
        loc,
        out_ty,
        out_dyn_dims,
        [&](mlir::OpBuilder& gb, mlir::Location gen_loc, mlir::ValueRange out_indices) {
            auto c0 = gb.create<mlir::arith::ConstantIndexOp>(gen_loc, 0).getResult();
            auto c1 = gb.create<mlir::arith::ConstantIndexOp>(gen_loc, 1).getResult();

            llvm::SmallVector<mlir::Value> base_indices(in_rank, c0);
            size_t output_pos = 0;
            for (size_t i = 0; i < in_rank; ++i) {
                if (std::binary_search(axes.begin(), axes.end(), i)) {
                    continue;
                }
                size_t od = keep_dims ? i : output_pos++;
                base_indices[i] = out_indices[od];
            }

            auto reduce_recursive =
                [&](auto&& self, mlir::OpBuilder& rb, size_t axis_pos, llvm::SmallVector<mlir::Value> indices, mlir::Value acc)
                    -> mlir::Value {
                if (axis_pos == axes.size()) {
                    auto val = rb.create<mlir::tensor::ExtractOp>(gen_loc, func.getArgument(0), indices).getResult();
                    return reduce_update(rb, gen_loc, elem_ty, kind, acc, val);
                }

                auto upper =
                    rb.create<mlir::tensor::DimOp>(gen_loc, func.getArgument(0), static_cast<int64_t>(axes[axis_pos]))
                        .getResult();
                auto loop = rb.create<mlir::scf::ForOp>(gen_loc, c0, upper, c1, mlir::ValueRange{acc});
                {
                    auto* body = loop.getBody();
                    mlir::OpBuilder lb(body, body->begin());
                    auto next_indices = indices;
                    next_indices[axes[axis_pos]] = loop.getInductionVar();
                    auto updated =
                        self(self, lb, axis_pos + 1, next_indices, loop.getRegionIterArgs()[0]);
                    lb.create<mlir::scf::YieldOp>(gen_loc, updated);
                }
                return loop.getResult(0);
            };

            auto reduced = reduce_recursive(reduce_recursive,
                                            gb,
                                            0,
                                            base_indices,
                                            make_init_value(gb, gen_loc, elem_ty, kind));

            if (kind == ReduceKind::Mean) {
                OPENVINO_ASSERT(mlir::isa<mlir::FloatType>(elem_ty), "ReduceMean MLIR: only float supported");
                mlir::Value denom_idx = c1;
                for (auto axis : axes) {
                    auto dim = gb.create<mlir::tensor::DimOp>(gen_loc, func.getArgument(0), static_cast<int64_t>(axis))
                                   .getResult();
                    denom_idx = gb.create<mlir::arith::MulIOp>(gen_loc, denom_idx, dim).getResult();
                }
                auto denom_i64 =
                    gb.create<mlir::arith::IndexCastOp>(gen_loc, mlir::IntegerType::get(&ctx, 64), denom_idx).getResult();
                auto denom =
                    gb.create<mlir::arith::SIToFPOp>(gen_loc, elem_ty, denom_i64).getResult();
                reduced = gb.create<mlir::arith::DivFOp>(gen_loc, reduced, denom).getResult();
            } else if (kind == ReduceKind::L2) {
                OPENVINO_ASSERT(mlir::isa<mlir::FloatType>(elem_ty), "ReduceL2 MLIR: only float supported");
                reduced = gb.create<mlir::math::SqrtOp>(gen_loc, reduced).getResult();
            }

            mlir::tensor::YieldOp::create(gb, gen_loc, reduced);
        });

    b.create<mlir::func::ReturnOp>(loc, generated.getResult());
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
