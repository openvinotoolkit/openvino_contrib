// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/power.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_pow_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::Power>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (elem_ty.isIntOrIndex())
                return static_cast<mlir::Value>(body.create<mlir::math::IPowIOp>(l, args[0], args[1]));
            return static_cast<mlir::Value>(body.create<mlir::math::PowFOp>(l, args[0], args[1]));
        });
}

}  // namespace metal_plugin
}  // namespace ov
