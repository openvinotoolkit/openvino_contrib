// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/power.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_pow_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::Power>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto node) {
            if (elem_ty.isIntOrIndex()) {
                auto et = node->get_output_element_type(0);
                bool is_unsigned = et == ov::element::u8 || et == ov::element::u16 ||
                                   et == ov::element::u32 || et == ov::element::u64;
                auto f32_ty = mlir::Float32Type::get(body.getContext());
                auto lhs_f = is_unsigned
                                 ? body.create<mlir::arith::UIToFPOp>(l, f32_ty, args[0]).getResult()
                                 : body.create<mlir::arith::SIToFPOp>(l, f32_ty, args[0]).getResult();
                auto rhs_f = is_unsigned
                                 ? body.create<mlir::arith::UIToFPOp>(l, f32_ty, args[1]).getResult()
                                 : body.create<mlir::arith::SIToFPOp>(l, f32_ty, args[1]).getResult();
                auto pow_f = body.create<mlir::math::PowFOp>(l, lhs_f, rhs_f);
                auto rounded = body.create<mlir::math::RoundOp>(l, pow_f);
                if (is_unsigned) {
                    return static_cast<mlir::Value>(
                        body.create<mlir::arith::FPToUIOp>(l, elem_ty, rounded).getResult());
                }
                return static_cast<mlir::Value>(
                    body.create<mlir::arith::FPToSIOp>(l, elem_ty, rounded).getResult());
            }
            return static_cast<mlir::Value>(body.create<mlir::math::PowFOp>(l, args[0], args[1]));
        });
}

}  // namespace gfx_plugin
}  // namespace ov
