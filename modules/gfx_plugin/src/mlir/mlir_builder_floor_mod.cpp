// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/floor_mod.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_floor_mod_from_model(const std::shared_ptr<const ov::Model>& model,
                                               mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::FloorMod>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (elem_ty.isIntOrIndex()) {
                auto rem = body.create<mlir::arith::RemSIOp>(l, args[0], args[1]);
                auto int_ty = mlir::dyn_cast<mlir::IntegerType>(elem_ty);
                OPENVINO_ASSERT(int_ty, "Eltwise MLIR: expected integer type for FloorMod path");
                auto zero = body.create<mlir::arith::ConstantIntOp>(l, 0, int_ty.getWidth());
                auto rhs_neg = body.create<mlir::arith::CmpIOp>(l, mlir::arith::CmpIPredicate::slt, args[1], zero);
                auto rem_neg = body.create<mlir::arith::CmpIOp>(l, mlir::arith::CmpIPredicate::slt, rem, zero);
                auto cond = body.create<mlir::arith::XOrIOp>(l, rhs_neg, rem_neg);
                auto add = body.create<mlir::arith::AddIOp>(l, rem, args[1]);
                return static_cast<mlir::Value>(body.create<mlir::arith::SelectOp>(l, cond, add, rem));
            }
            if (mlir::isa<mlir::Float16Type>(elem_ty)) {
                auto f32 = mlir::Float32Type::get(body.getContext());
                auto lhs = body.create<mlir::arith::ExtFOp>(l, f32, args[0]);
                auto rhs = body.create<mlir::arith::ExtFOp>(l, f32, args[1]);
                auto div = body.create<mlir::arith::DivFOp>(l, lhs, rhs);
                auto floor = body.create<mlir::math::FloorOp>(l, div);
                auto mul = body.create<mlir::arith::MulFOp>(l, floor, rhs);
                auto sub = body.create<mlir::arith::SubFOp>(l, lhs, mul);
                auto abs_sub = body.create<mlir::math::AbsFOp>(l, sub);
                auto abs_rhs = body.create<mlir::math::AbsFOp>(l, rhs);
                auto ge = body.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OGE, abs_sub, abs_rhs);
                auto zero = body.create<mlir::arith::ConstantOp>(l, body.getFloatAttr(f32, 0.0));
                auto fixed = body.create<mlir::arith::SelectOp>(l, ge, zero, sub);
                return static_cast<mlir::Value>(body.create<mlir::arith::TruncFOp>(l, elem_ty, fixed));
            }
            auto div = body.create<mlir::arith::DivFOp>(l, args[0], args[1]);
            auto floor = body.create<mlir::math::FloorOp>(l, div);
            auto mul = body.create<mlir::arith::MulFOp>(l, floor, args[1]);
            auto sub = body.create<mlir::arith::SubFOp>(l, args[0], mul);
            auto abs_sub = body.create<mlir::math::AbsFOp>(l, sub);
            auto abs_rhs = body.create<mlir::math::AbsFOp>(l, args[1]);
            auto ge = body.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OGE, abs_sub, abs_rhs);
            auto zero = body.create<mlir::arith::ConstantOp>(l, body.getFloatAttr(elem_ty, 0.0));
            return static_cast<mlir::Value>(body.create<mlir::arith::SelectOp>(l, ge, zero, sub));
        });
}

}  // namespace gfx_plugin
}  // namespace ov
