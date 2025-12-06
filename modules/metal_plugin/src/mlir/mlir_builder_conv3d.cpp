// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/core/model.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_conv3d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect,
                    mlir::math::MathDialect>();
    std::shared_ptr<const ov::op::v1::Convolution> conv;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            if (c->get_input_partial_shape(0).rank().is_static() &&
                c->get_input_partial_shape(0).rank().get_length() == 5) {
                conv = c;
                break;
            }
        }
    }
    OPENVINO_ASSERT(conv, "Conv3D builder: Convolution op (rank-5) not found");

    const auto in_shape = conv->get_input_shape(0);   // NCDHW
    const auto w_shape  = conv->get_input_shape(1);   // OIDHW
    OPENVINO_ASSERT(in_shape.size() == 5 && w_shape.size() == 5, "Conv3D: rank-5 expected");

    auto f32 = mlir::Float32Type::get(&ctx);
    llvm::SmallVector<int64_t> in_dims(in_shape.begin(), in_shape.end());
    llvm::SmallVector<int64_t> w_dims(w_shape.begin(), w_shape.end());
    auto in_ty = mlir::RankedTensorType::get(in_dims, f32);
    auto w_ty  = mlir::RankedTensorType::get(w_dims, f32);

    auto out_shape = conv->get_output_shape(0);
    llvm::SmallVector<int64_t> out_dims(out_shape.begin(), out_shape.end());
    auto out_ty = mlir::RankedTensorType::get(out_dims, f32);

    mlir::OpBuilder mb(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mb.setInsertionPointToStart(module.getBody());

    auto func_type = mb.getFunctionType({in_ty, w_ty}, {out_ty});
    auto func = mb.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "conv3d_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto loc = mlir::UnknownLoc::get(&ctx);
    auto empty = b.create<mlir::tensor::EmptyOp>(loc, out_dims, f32);

    const auto pads_begin = conv->get_pads_begin();  // {front, top, left}
    const auto pads_end   = conv->get_pads_end();    // {back, bottom, right}
    bool has_pad = (pads_begin[0] || pads_begin[1] || pads_begin[2] ||
                    pads_end[0]   || pads_end[1]   || pads_end[2]);

    mlir::Value input_val = func.getArgument(0);
    if (has_pad) {
        mlir::SmallVector<int64_t> low_static = {
            0,
            0,
            static_cast<int64_t>(pads_begin[0]),
            static_cast<int64_t>(pads_begin[1]),
            static_cast<int64_t>(pads_begin[2])
        };
        mlir::SmallVector<int64_t> high_static = {
            0,
            0,
            static_cast<int64_t>(pads_end[0]),
            static_cast<int64_t>(pads_end[1]),
            static_cast<int64_t>(pads_end[2])
        };
        mlir::SmallVector<mlir::OpFoldResult> low_ofr, high_ofr;
        for (auto v : low_static) low_ofr.push_back(b.getI64IntegerAttr(v));
        for (auto v : high_static) high_ofr.push_back(b.getI64IntegerAttr(v));
        auto zero = b.create<mlir::arith::ConstantOp>(loc, b.getF32FloatAttr(0.0f));
        input_val = b.create<mlir::tensor::PadOp>(
            loc,
            mlir::RankedTensorType::get(
                {static_cast<int64_t>(in_shape[0]),
                 static_cast<int64_t>(in_shape[1]),
                 static_cast<int64_t>(in_shape[2] + pads_begin[0] + pads_end[0]),
                 static_cast<int64_t>(in_shape[3] + pads_begin[1] + pads_end[1]),
                 static_cast<int64_t>(in_shape[4] + pads_begin[2] + pads_end[2])},
                f32),
            func.getArgument(0),
            low_ofr,
            high_ofr,
            zero,
            false);
    }

    auto conv_op = b.create<mlir::linalg::Conv3DNcdhwFcdhwOp>(
        loc,
        out_ty,
        mlir::ValueRange{input_val, func.getArgument(1)},
        mlir::ValueRange{empty});

    b.create<mlir::func::ReturnOp>(loc, conv_op.getResult(0));

    return module;
}

}  // namespace metal_plugin
}  // namespace ov
