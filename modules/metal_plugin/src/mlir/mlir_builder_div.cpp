// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/divide.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_div_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::Divide>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (elem_ty.isIntOrIndex())
                return static_cast<mlir::Value>(body.create<mlir::arith::DivSIOp>(l, args[0], args[1]));
            return static_cast<mlir::Value>(body.create<mlir::arith::DivFOp>(l, args[0], args[1]));
        });
}

}  // namespace metal_plugin
}  // namespace ov
