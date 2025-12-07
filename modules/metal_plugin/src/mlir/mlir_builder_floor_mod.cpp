// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/floor_mod.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_floor_mod_from_model(const std::shared_ptr<const ov::Model>& model,
                                               mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::FloorMod>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            auto rem = elem_ty.isIntOrIndex()
                           ? static_cast<mlir::Value>(body.create<mlir::arith::RemSIOp>(l, args[0], args[1]))
                           : static_cast<mlir::Value>(body.create<mlir::arith::RemFOp>(l, args[0], args[1]));
            if (elem_ty.isIntOrIndex()) {
                auto int_ty = mlir::dyn_cast<mlir::IntegerType>(elem_ty);
                OPENVINO_ASSERT(int_ty, "Eltwise MLIR: expected integer type for FloorMod path");
                auto zero = body.create<mlir::arith::ConstantIntOp>(l, 0, int_ty.getWidth());
                auto rhs_neg = body.create<mlir::arith::CmpIOp>(l, mlir::arith::CmpIPredicate::slt, args[1], zero);
                auto rem_neg = body.create<mlir::arith::CmpIOp>(l, mlir::arith::CmpIPredicate::slt, rem, zero);
                auto cond = body.create<mlir::arith::XOrIOp>(l, rhs_neg, rem_neg);
                auto add = body.create<mlir::arith::AddIOp>(l, rem, args[1]);
                return static_cast<mlir::Value>(body.create<mlir::arith::SelectOp>(l, cond, add, rem));
            }
            auto zero = body.create<mlir::arith::ConstantOp>(l, body.getFloatAttr(elem_ty, 0.0));
            auto rhs_neg = body.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OLT, args[1], zero);
            auto rem_neg = body.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OLT, rem, zero);
            auto cond = body.create<mlir::arith::XOrIOp>(l, rhs_neg, rem_neg);
            auto add = body.create<mlir::arith::AddFOp>(l, rem, args[1]);
            return static_cast<mlir::Value>(body.create<mlir::arith::SelectOp>(l, cond, add, rem));
        });
}

}  // namespace metal_plugin
}  // namespace ov
