// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "ov_ops/rms.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

mlir::Value cast_float(mlir::OpBuilder& b,
                       mlir::Location loc,
                       mlir::Value value,
                       mlir::Type dst_ty) {
    if (value.getType() == dst_ty) {
        return value;
    }
    auto src_ty = value.getType();
    OPENVINO_ASSERT(mlir::isa<mlir::FloatType>(src_ty) && mlir::isa<mlir::FloatType>(dst_ty),
                    "RMS MLIR: only floating-point casts are supported");
    const unsigned src_width = mlir::cast<mlir::FloatType>(src_ty).getWidth();
    const unsigned dst_width = mlir::cast<mlir::FloatType>(dst_ty).getWidth();
    return src_width < dst_width ? b.create<mlir::arith::ExtFOp>(loc, dst_ty, value).getResult()
                                 : b.create<mlir::arith::TruncFOp>(loc, dst_ty, value).getResult();
}

mlir::Value gamma_index_for_output(mlir::OpBuilder& b,
                                   mlir::Location loc,
                                   mlir::Value gamma,
                                   mlir::RankedTensorType gamma_ty,
                                   mlir::ValueRange out_indices) {
    const int64_t gamma_rank = gamma_ty.getRank();
    if (gamma_rank == 0) {
        return b.create<mlir::tensor::ExtractOp>(loc, gamma).getResult();
    }

    llvm::SmallVector<mlir::Value> gamma_indices;
    gamma_indices.reserve(static_cast<size_t>(gamma_rank));
    auto zero = b.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
    const int64_t out_rank = static_cast<int64_t>(out_indices.size());
    const int64_t offset = out_rank - gamma_rank;
    for (int64_t i = 0; i < gamma_rank; ++i) {
        const int64_t dim = gamma_ty.getDimSize(i);
        if (dim == 1) {
            gamma_indices.push_back(zero);
            continue;
        }
        const int64_t out_axis = offset + i;
        OPENVINO_ASSERT(out_axis >= 0 && out_axis < out_rank,
                        "RMS MLIR: gamma rank is not broadcastable to data rank");
        gamma_indices.push_back(out_indices[static_cast<size_t>(out_axis)]);
    }
    return b.create<mlir::tensor::ExtractOp>(loc, gamma, gamma_indices).getResult();
}

}  // namespace

mlir::ModuleOp build_mlir_rms_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect,
                    mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect,
                    mlir::math::MathDialect>();

    std::shared_ptr<const ov::op::internal::RMS> rms_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto rms = ov::as_type_ptr<const ov::op::internal::RMS>(node)) {
            OPENVINO_ASSERT(!rms_node, "RMS MLIR builder: expected single RMS op");
            rms_node = rms;
        }
    }
    OPENVINO_ASSERT(rms_node, "RMS MLIR builder: RMS op not found");
    OPENVINO_ASSERT(rms_node->get_input_size() == 2 && rms_node->get_output_size() == 1,
                    "RMS MLIR: expected data, gamma and one output");

    const auto data_pshape = rms_node->get_input_partial_shape(0);
    const auto gamma_pshape = rms_node->get_input_partial_shape(1);
    const auto out_pshape = rms_node->get_output_partial_shape(0);
    OPENVINO_ASSERT(data_pshape.rank().is_static() && gamma_pshape.rank().is_static() && out_pshape.rank().is_static(),
                    "RMS MLIR: ranks must be static");
    OPENVINO_ASSERT(data_pshape.rank().get_length() >= 2, "RMS MLIR: data rank must be at least 2");

    auto data_elem_ty = to_mlir_type(rms_node->get_input_element_type(0),
                                     ctx,
                                     /*fallback_f32=*/false,
                                     /*allow_unsigned=*/false,
                                     /*allow_small_ints=*/false,
                                     /*allow_bf16=*/false,
                                     /*allow_boolean=*/false,
                                     /*signless_integers=*/true);
    auto gamma_elem_ty = to_mlir_type(rms_node->get_input_element_type(1),
                                      ctx,
                                      /*fallback_f32=*/false,
                                      /*allow_unsigned=*/false,
                                      /*allow_small_ints=*/false,
                                      /*allow_bf16=*/false,
                                      /*allow_boolean=*/false,
                                      /*signless_integers=*/true);
    auto out_elem_ty = to_mlir_type(rms_node->get_output_element_type(0),
                                    ctx,
                                    /*fallback_f32=*/false,
                                    /*allow_unsigned=*/false,
                                    /*allow_small_ints=*/false,
                                    /*allow_bf16=*/false,
                                    /*allow_boolean=*/false,
                                    /*signless_integers=*/true);
    OPENVINO_ASSERT(mlir::isa<mlir::FloatType>(data_elem_ty) &&
                    mlir::isa<mlir::FloatType>(gamma_elem_ty) &&
                    mlir::isa<mlir::FloatType>(out_elem_ty),
                    "RMS MLIR: only floating-point tensors are supported");

    mlir::SmallVector<int64_t> data_dims = to_shape(data_pshape);
    mlir::SmallVector<int64_t> gamma_dims = to_shape(gamma_pshape);
    mlir::SmallVector<int64_t> out_dims = to_shape(out_pshape);
    const int64_t rank = static_cast<int64_t>(data_dims.size());
    OPENVINO_ASSERT(static_cast<size_t>(rank) == out_dims.size(),
                    "RMS MLIR: output rank must match data rank");

    auto data_ty = mlir::RankedTensorType::get(data_dims, data_elem_ty);
    auto gamma_ty = mlir::RankedTensorType::get(gamma_dims, gamma_elem_ty);
    auto out_ty = mlir::RankedTensorType::get(out_dims, out_elem_ty);
    auto compute_ty = mlir::Float32Type::get(&ctx);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx),
                                              "rms_main",
                                              mb.getFunctionType({data_ty, gamma_ty}, {out_ty}));
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    llvm::SmallVector<mlir::Value> out_dyn_dims;
    out_dyn_dims.reserve(out_dims.size());
    for (int64_t i = 0; i < rank; ++i) {
        if (out_dims[static_cast<size_t>(i)] == mlir::ShapedType::kDynamic) {
            out_dyn_dims.push_back(b.create<mlir::tensor::DimOp>(loc, func.getArgument(0), i).getResult());
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
            auto last_dim = gb.create<mlir::tensor::DimOp>(gen_loc, func.getArgument(0), rank - 1).getResult();
            auto zero_f = gb.create<mlir::arith::ConstantOp>(gen_loc, mlir::FloatAttr::get(compute_ty, 0.0)).getResult();

            auto loop = gb.create<mlir::scf::ForOp>(gen_loc, c0, last_dim, c1, mlir::ValueRange{zero_f});
            {
                auto* body = loop.getBody();
                mlir::OpBuilder lb(body, body->begin());
                llvm::SmallVector<mlir::Value> reduce_indices(out_indices.begin(), out_indices.end());
                reduce_indices[static_cast<size_t>(rank - 1)] = loop.getInductionVar();
                auto x = lb.create<mlir::tensor::ExtractOp>(gen_loc, func.getArgument(0), reduce_indices).getResult();
                auto xf = cast_float(lb, gen_loc, x, compute_ty);
                auto sq = lb.create<mlir::arith::MulFOp>(gen_loc, xf, xf).getResult();
                auto sum = lb.create<mlir::arith::AddFOp>(gen_loc, loop.getRegionIterArgs()[0], sq).getResult();
                lb.create<mlir::scf::YieldOp>(gen_loc, sum);
            }

            auto denom_i64 = gb.create<mlir::arith::IndexCastOp>(gen_loc,
                                                                 mlir::IntegerType::get(&ctx, 64),
                                                                 last_dim).getResult();
            auto denom = gb.create<mlir::arith::SIToFPOp>(gen_loc, compute_ty, denom_i64).getResult();
            auto mean = gb.create<mlir::arith::DivFOp>(gen_loc, loop.getResult(0), denom).getResult();
            auto eps = gb.create<mlir::arith::ConstantOp>(gen_loc,
                                                          mlir::FloatAttr::get(compute_ty,
                                                                               rms_node->get_epsilon())).getResult();
            auto mean_eps = gb.create<mlir::arith::AddFOp>(gen_loc, mean, eps).getResult();
            auto sqrt = gb.create<mlir::math::SqrtOp>(gen_loc, mean_eps).getResult();
            auto one = gb.create<mlir::arith::ConstantOp>(gen_loc, mlir::FloatAttr::get(compute_ty, 1.0)).getResult();
            auto inv = gb.create<mlir::arith::DivFOp>(gen_loc, one, sqrt).getResult();
            auto x = gb.create<mlir::tensor::ExtractOp>(gen_loc, func.getArgument(0), out_indices).getResult();
            auto gamma = gamma_index_for_output(gb, gen_loc, func.getArgument(1), gamma_ty, out_indices);
            auto xf = cast_float(gb, gen_loc, x, compute_ty);
            auto gf = cast_float(gb, gen_loc, gamma, compute_ty);
            auto scaled = gb.create<mlir::arith::MulFOp>(gen_loc,
                                                         gb.create<mlir::arith::MulFOp>(gen_loc, xf, inv).getResult(),
                                                         gf).getResult();
            mlir::tensor::YieldOp::create(gb, gen_loc, cast_float(gb, gen_loc, scaled, out_elem_ty));
        });

    b.create<mlir::func::ReturnOp>(loc, generated.getResult());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
