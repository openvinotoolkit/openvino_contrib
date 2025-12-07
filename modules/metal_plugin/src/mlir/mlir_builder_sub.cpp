// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/squared_difference.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_sub_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::Subtract>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto node) {
            mlir::Value res = elem_ty.isIntOrIndex()
                                  ? static_cast<mlir::Value>(body.create<mlir::arith::SubIOp>(l, args[0], args[1]))
                                  : static_cast<mlir::Value>(body.create<mlir::arith::SubFOp>(l, args[0], args[1]));
            if (ov::is_type<ov::op::v0::SquaredDifference>(node.get())) {
                // SquaredDifference = (a - b)^2
                res = elem_ty.isIntOrIndex()
                          ? static_cast<mlir::Value>(body.create<mlir::arith::MulIOp>(l, res, res))
                          : static_cast<mlir::Value>(body.create<mlir::arith::MulFOp>(l, res, res));
            }
            return res;
        });
}

mlir::ModuleOp build_mlir_squared_difference_from_model(const std::shared_ptr<const ov::Model>& model,
                                                        mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v0::SquaredDifference>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            auto diff = elem_ty.isIntOrIndex()
                            ? static_cast<mlir::Value>(body.create<mlir::arith::SubIOp>(l, args[0], args[1]))
                            : static_cast<mlir::Value>(body.create<mlir::arith::SubFOp>(l, args[0], args[1]));
            return elem_ty.isIntOrIndex()
                       ? static_cast<mlir::Value>(body.create<mlir::arith::MulIOp>(l, diff, diff))
                       : static_cast<mlir::Value>(body.create<mlir::arith::MulFOp>(l, diff, diff));
        });
}

}  // namespace metal_plugin
}  // namespace ov
