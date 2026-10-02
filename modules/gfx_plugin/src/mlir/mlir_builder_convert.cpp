// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/gfx_mlir_type_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/convert.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
bool is_unsigned(ov::element::Type et) {
    return et == ov::element::boolean || et == ov::element::u8 || et == ov::element::u16 ||
           et == ov::element::u32 || et == ov::element::u64;
}

bool is_signed(ov::element::Type et) {
    return et == ov::element::i8 || et == ov::element::i32 || et == ov::element::i64;
}

}  // namespace

mlir::ModuleOp build_mlir_convert_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect>();

    std::shared_ptr<const ov::op::v0::Convert> cvt;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
            OPENVINO_ASSERT(!cvt, "Convert MLIR builder: expected single Convert");
            cvt = c;
        }
    }
    OPENVINO_ASSERT(cvt, "Convert MLIR builder: Convert op not found");

    auto in_shape = to_shape(cvt->get_input_partial_shape(0));
    auto out_shape = to_shape(cvt->get_output_partial_shape(0));
    const auto in_et = cvt->get_input_element_type(0);
    const auto out_et = cvt->get_output_element_type(0);
    auto in_ty = to_mlir_type(in_et,
                              ctx,
                              /*fallback_f32=*/false,
                              /*allow_unsigned=*/true,
                              /*allow_small_ints=*/true,
                              /*allow_bf16=*/false,
                              /*allow_boolean=*/true,
                              /*signless_integers=*/true);
    auto out_ty = to_mlir_type(out_et,
                               ctx,
                               /*fallback_f32=*/false,
                               /*allow_unsigned=*/true,
                               /*allow_small_ints=*/true,
                               /*allow_bf16=*/false,
                               /*allow_boolean=*/true,
                               /*signless_integers=*/true);

    auto in_tensor_ty = mlir::RankedTensorType::get(in_shape, in_ty);
    auto out_tensor_ty = mlir::RankedTensorType::get(out_shape, out_ty);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_tensor_ty}, {out_tensor_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "convert_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    auto loc = mlir::UnknownLoc::get(&ctx);
    b.setInsertionPointToStart(&func.getBody().front());

    auto out_dyn_dims = materialize_dynamic_dims_from_tensor(b, loc, func.getArgument(0), out_shape);
    auto empty = b.create<mlir::tensor::EmptyOp>(loc, out_shape, out_ty, out_dyn_dims);
    auto map = mlir::AffineMap::getMultiDimIdentityMap(out_shape.size(), &ctx);
    llvm::SmallVector<mlir::utils::IteratorType> iters(out_shape.size(), mlir::utils::IteratorType::parallel);

    auto generic = b.create<mlir::linalg::GenericOp>(
        loc,
        out_tensor_ty,
        mlir::ValueRange{func.getArgument(0)},
        mlir::ValueRange{empty},
        mlir::ArrayRef<mlir::AffineMap>{map, map},
        mlir::ArrayRef<mlir::utils::IteratorType>(iters));
    {
        auto& region = generic.getRegion();
        region.getBlocks().clear();
        auto* block = &region.emplaceBlock();
        block->addArguments({in_ty, out_ty}, {loc, loc});
        mlir::OpBuilder body(block, block->begin());
        auto x = block->getArgument(0);
        mlir::Value y;
        if (in_ty == out_ty) {
            y = x;
        } else if (mlir::isa<mlir::FloatType>(in_ty) && mlir::isa<mlir::FloatType>(out_ty)) {
            auto in_bits = mlir::cast<mlir::FloatType>(in_ty).getWidth();
            auto out_bits = mlir::cast<mlir::FloatType>(out_ty).getWidth();
            if (out_bits > in_bits) {
                y = body.create<mlir::arith::ExtFOp>(loc, out_ty, x);
            } else {
                y = body.create<mlir::arith::TruncFOp>(loc, out_ty, x);
            }
        } else if (mlir::isa<mlir::IntegerType>(in_ty) && mlir::isa<mlir::IntegerType>(out_ty)) {
            auto in_bits = mlir::cast<mlir::IntegerType>(in_ty).getWidth();
            auto out_bits = mlir::cast<mlir::IntegerType>(out_ty).getWidth();
            if (out_bits > in_bits) {
                if (is_unsigned(in_et)) {
                    y = body.create<mlir::arith::ExtUIOp>(loc, out_ty, x);
                } else {
                    y = body.create<mlir::arith::ExtSIOp>(loc, out_ty, x);
                }
            } else {
                y = body.create<mlir::arith::TruncIOp>(loc, out_ty, x);
            }
        } else if (mlir::isa<mlir::IntegerType>(in_ty) && mlir::isa<mlir::FloatType>(out_ty)) {
            if (is_unsigned(in_et)) {
                y = body.create<mlir::arith::UIToFPOp>(loc, out_ty, x);
            } else {
                y = body.create<mlir::arith::SIToFPOp>(loc, out_ty, x);
            }
        } else if (mlir::isa<mlir::FloatType>(in_ty) && mlir::isa<mlir::IntegerType>(out_ty)) {
            if (is_unsigned(out_et)) {
                y = body.create<mlir::arith::FPToUIOp>(loc, out_ty, x);
            } else {
                y = body.create<mlir::arith::FPToSIOp>(loc, out_ty, x);
            }
        } else {
            OPENVINO_THROW("Convert MLIR: unsupported type conversion");
        }
        body.create<mlir::linalg::YieldOp>(loc, y);
    }

    b.create<mlir::func::ReturnOp>(loc, generic.getResults());
    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
