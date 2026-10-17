// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/divide.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_div_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::Divide>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (elem_ty.isIntOrIndex())
                return static_cast<mlir::Value>(body.create<mlir::arith::DivSIOp>(l, args[0], args[1]));
            if (mlir::isa<mlir::Float16Type>(elem_ty)) {
                auto f32 = mlir::Float32Type::get(body.getContext());
                auto lhs = body.create<mlir::arith::ExtFOp>(l, f32, args[0]);
                auto rhs = body.create<mlir::arith::ExtFOp>(l, f32, args[1]);
                auto div = body.create<mlir::arith::DivFOp>(l, lhs, rhs);
                return static_cast<mlir::Value>(body.create<mlir::arith::TruncFOp>(l, elem_ty, div));
            }
            return static_cast<mlir::Value>(body.create<mlir::arith::DivFOp>(l, args[0], args[1]));
        });
}

}  // namespace gfx_plugin
}  // namespace ov
