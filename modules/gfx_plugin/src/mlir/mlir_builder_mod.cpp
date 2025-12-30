// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/mod.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_mod_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::Mod>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (elem_ty.isIntOrIndex())
                return static_cast<mlir::Value>(body.create<mlir::arith::RemSIOp>(l, args[0], args[1]));
            if (mlir::isa<mlir::Float16Type>(elem_ty)) {
                auto f32 = mlir::Float32Type::get(body.getContext());
                auto lhs = body.create<mlir::arith::ExtFOp>(l, f32, args[0]);
                auto rhs = body.create<mlir::arith::ExtFOp>(l, f32, args[1]);
                auto rem = body.create<mlir::arith::RemFOp>(l, lhs, rhs);
                auto abs_rem = body.create<mlir::math::AbsFOp>(l, rem);
                auto abs_rhs = body.create<mlir::math::AbsFOp>(l, rhs);
                auto ge = body.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OGE, abs_rem, abs_rhs);
                auto zero = body.create<mlir::arith::ConstantOp>(l, body.getFloatAttr(f32, 0.0));
                auto fixed = body.create<mlir::arith::SelectOp>(l, ge, zero, rem);
                return static_cast<mlir::Value>(body.create<mlir::arith::TruncFOp>(l, elem_ty, fixed));
            }
            auto rem = body.create<mlir::arith::RemFOp>(l, args[0], args[1]);
            auto abs_rem = body.create<mlir::math::AbsFOp>(l, rem);
            auto abs_rhs = body.create<mlir::math::AbsFOp>(l, args[1]);
            auto ge = body.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OGE, abs_rem, abs_rhs);
            auto zero = body.create<mlir::arith::ConstantOp>(l, body.getFloatAttr(elem_ty, 0.0));
            return static_cast<mlir::Value>(body.create<mlir::arith::SelectOp>(l, ge, zero, rem));
        });
}

}  // namespace gfx_plugin
}  // namespace ov
