// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

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
#include "openvino/op/range.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

mlir::Value extract_scalar(mlir::OpBuilder& b, mlir::Location loc, mlir::Value tensor) {
    auto type = mlir::cast<mlir::RankedTensorType>(tensor.getType());
    if (type.getRank() == 0) {
        return b.create<mlir::tensor::ExtractOp>(loc, tensor, mlir::ValueRange{}).getResult();
    }
    auto zero = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    return b.create<mlir::tensor::ExtractOp>(loc, tensor, mlir::ValueRange{zero}).getResult();
}

mlir::Value cast_scalar_to_i64(mlir::OpBuilder& b, mlir::Location loc, mlir::Value value) {
    auto src_ty = value.getType();
    auto i64_ty = b.getI64Type();
    if (src_ty == i64_ty) {
        return value;
    }
    auto int_ty = mlir::dyn_cast<mlir::IntegerType>(src_ty);
    OPENVINO_ASSERT(int_ty, "Range MLIR: dynamic output length requires integer bounds");
    if (int_ty.getWidth() < 64) {
        return int_ty.isUnsigned() ? b.create<mlir::arith::ExtUIOp>(loc, i64_ty, value).getResult()
                                   : b.create<mlir::arith::ExtSIOp>(loc, i64_ty, value).getResult();
    }
    if (int_ty.getWidth() > 64) {
        return b.create<mlir::arith::TruncIOp>(loc, i64_ty, value).getResult();
    }
    return value;
}

}  // namespace

mlir::ModuleOp build_mlir_range_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();

    std::shared_ptr<const ov::Node> range_node;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v4::Range>(node) ||
            ov::as_type_ptr<const ov::op::v0::Range>(node)) {
            OPENVINO_ASSERT(!range_node, "Range MLIR builder: expected single Range");
            range_node = node;
        }
    }
    OPENVINO_ASSERT(range_node, "Range MLIR builder: Range op not found");

    const auto out_pshape = range_node->get_output_partial_shape(0);
    OPENVINO_ASSERT(out_pshape.rank().is_static() && out_pshape.rank().get_length() == 1,
                    "Range MLIR: output rank must be 1");
    const bool dynamic_output = !out_pshape.is_static();
    const int64_t out_total = dynamic_output ? mlir::ShapedType::kDynamic
                                             : static_cast<int64_t>(ov::shape_size(out_pshape.to_shape()));
    auto elem_ty = to_mlir_type(range_node->get_output_element_type(0), ctx, /*fallback_f32=*/false,
                                /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true,
                                /*allow_bf16=*/false,
                                /*allow_boolean=*/false,
                                /*signless_integers=*/true);
    auto out_ty = mlir::RankedTensorType::get({out_total}, elem_ty);

    mlir::SmallVector<int64_t> start_shape = to_shape(range_node->get_input_partial_shape(0));
    mlir::SmallVector<int64_t> stop_shape = to_shape(range_node->get_input_partial_shape(1));
    mlir::SmallVector<int64_t> step_shape = to_shape(range_node->get_input_partial_shape(2));

    auto start_elem_ty = to_mlir_type(range_node->get_input_element_type(0), ctx, /*fallback_f32=*/false,
                                      /*allow_unsigned=*/true,
                                      /*allow_small_ints=*/true,
                                      /*allow_bf16=*/false,
                                      /*allow_boolean=*/false,
                                      /*signless_integers=*/true);
    auto stop_elem_ty = to_mlir_type(range_node->get_input_element_type(1), ctx, /*fallback_f32=*/false,
                                     /*allow_unsigned=*/true,
                                     /*allow_small_ints=*/true,
                                     /*allow_bf16=*/false,
                                     /*allow_boolean=*/false,
                                     /*signless_integers=*/true);
    auto step_elem_ty = to_mlir_type(range_node->get_input_element_type(2), ctx, /*fallback_f32=*/false,
                                     /*allow_unsigned=*/true,
                                     /*allow_small_ints=*/true,
                                     /*allow_bf16=*/false,
                                     /*allow_boolean=*/false,
                                     /*signless_integers=*/true);

    auto start_ty = mlir::RankedTensorType::get(start_shape, start_elem_ty);
    auto stop_ty = mlir::RankedTensorType::get(stop_shape, stop_elem_ty);
    auto step_ty = mlir::RankedTensorType::get(step_shape, step_elem_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "range_main",
                                              mb.getFunctionType({start_ty, stop_ty, step_ty}, {out_ty}));
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto start_raw = extract_scalar(b, loc, func.getArgument(0));
    auto stop_raw = extract_scalar(b, loc, func.getArgument(1));
    auto step_raw = extract_scalar(b, loc, func.getArgument(2));
    auto start_val = start_raw;
    auto step_val = step_raw;
    auto cast_to_elem = [&](mlir::Value v) -> mlir::Value {
        auto src_ty = v.getType();
        if (src_ty == elem_ty) {
            return v;
        }
        if (mlir::isa<mlir::FloatType>(elem_ty) && mlir::isa<mlir::IntegerType>(src_ty)) {
            auto it = mlir::cast<mlir::IntegerType>(src_ty);
            return it.isUnsigned() ? b.create<mlir::arith::UIToFPOp>(loc, elem_ty, v).getResult()
                                   : b.create<mlir::arith::SIToFPOp>(loc, elem_ty, v).getResult();
        }
        if (mlir::isa<mlir::IntegerType>(elem_ty) && mlir::isa<mlir::FloatType>(src_ty)) {
            auto it = mlir::cast<mlir::IntegerType>(elem_ty);
            return it.isUnsigned() ? b.create<mlir::arith::FPToUIOp>(loc, elem_ty, v).getResult()
                                   : b.create<mlir::arith::FPToSIOp>(loc, elem_ty, v).getResult();
        }
        if (mlir::isa<mlir::IntegerType>(elem_ty) && mlir::isa<mlir::IntegerType>(src_ty)) {
            auto dst = mlir::cast<mlir::IntegerType>(elem_ty);
            auto src = mlir::cast<mlir::IntegerType>(src_ty);
            if (dst.getWidth() > src.getWidth()) {
                return dst.isUnsigned() ? b.create<mlir::arith::ExtUIOp>(loc, elem_ty, v).getResult()
                                        : b.create<mlir::arith::ExtSIOp>(loc, elem_ty, v).getResult();
            }
            return b.create<mlir::arith::TruncIOp>(loc, elem_ty, v).getResult();
        }
        OPENVINO_THROW("Range MLIR: unsupported type cast");
    };
    start_val = cast_to_elem(start_val);
    step_val = cast_to_elem(step_val);

    llvm::SmallVector<mlir::Value> out_dyn_dims;
    if (dynamic_output) {
        auto start_i64 = cast_scalar_to_i64(b, loc, start_raw);
        auto stop_i64 = cast_scalar_to_i64(b, loc, stop_raw);
        auto step_i64 = cast_scalar_to_i64(b, loc, step_raw);
        auto diff = b.create<mlir::arith::SubIOp>(loc, stop_i64, start_i64).getResult();
        auto len_i64 = b.create<mlir::arith::CeilDivSIOp>(loc, diff, step_i64).getResult();
        out_dyn_dims.push_back(b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), len_i64).getResult());
    }
    auto out_flat = b.create<mlir::tensor::EmptyOp>(loc, mlir::ArrayRef<int64_t>{out_total}, elem_ty, out_dyn_dims);

    auto c0 = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value c_out = dynamic_output ? out_dyn_dims.front()
                                       : b.create<mlir::arith::ConstantIndexOp>(loc, out_total).getResult();

    auto loop = b.create<mlir::scf::ForOp>(loc, c0, c_out, c1, mlir::ValueRange{out_flat});
    {
        auto* body = loop.getBody();
        mlir::OpBuilder lb(body, body->begin());
        auto iv = loop.getInductionVar();
        auto acc = loop.getRegionIterArgs()[0];

        mlir::Value idx_val;
        if (mlir::isa<mlir::FloatType>(elem_ty)) {
            auto idx_i64 = lb.create<mlir::arith::IndexCastOp>(loc, lb.getI64Type(), iv).getResult();
            idx_val = lb.create<mlir::arith::SIToFPOp>(loc, elem_ty, idx_i64).getResult();
            auto step_mul = lb.create<mlir::arith::MulFOp>(loc, step_val, idx_val).getResult();
            idx_val = lb.create<mlir::arith::AddFOp>(loc, start_val, step_mul).getResult();
        } else {
            auto idx_int = lb.create<mlir::arith::IndexCastOp>(loc, elem_ty, iv).getResult();
            auto step_mul = lb.create<mlir::arith::MulIOp>(loc, step_val, idx_int).getResult();
            idx_val = lb.create<mlir::arith::AddIOp>(loc, start_val, step_mul).getResult();
        }

        auto updated = lb.create<mlir::tensor::InsertOp>(loc, idx_val, acc, mlir::ValueRange{iv}).getResult();
        lb.create<mlir::scf::YieldOp>(loc, updated);
    }

    b.create<mlir::func::ReturnOp>(loc, loop.getResults()[0]);
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
