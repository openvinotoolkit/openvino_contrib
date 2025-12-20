// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/prelu.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_prelu_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v0::PRelu>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (!mlir::isa<mlir::FloatType>(elem_ty)) {
                OPENVINO_THROW("PRelu MLIR: only floating-point types supported");
            }
            auto zero = body.create<mlir::arith::ConstantOp>(l, body.getFloatAttr(elem_ty, 0.0));
            auto cmp = body.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OGE, args[0], zero);
            auto neg = body.create<mlir::arith::MulFOp>(l, args[0], args[1]);
            return body.create<mlir::arith::SelectOp>(l, cmp, args[0], neg);
        });
}

}  // namespace metal_plugin
}  // namespace ov
