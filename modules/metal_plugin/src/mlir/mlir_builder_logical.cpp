// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"

namespace ov {
namespace metal_plugin {

mlir::ModuleOp build_mlir_logical_and_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::LogicalAnd>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type /*elem_ty*/, auto) {
            // Logical ops expect bool tensors; use arith::AndIOp
            return static_cast<mlir::Value>(body.create<mlir::arith::AndIOp>(l, args[0], args[1]));
        });
}

mlir::ModuleOp build_mlir_logical_or_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::LogicalOr>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type /*elem_ty*/, auto) {
            return static_cast<mlir::Value>(body.create<mlir::arith::OrIOp>(l, args[0], args[1]));
        });
}

mlir::ModuleOp build_mlir_logical_xor_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::LogicalXor>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type /*elem_ty*/, auto) {
            return static_cast<mlir::Value>(body.create<mlir::arith::XOrIOp>(l, args[0], args[1]));
        });
}

}  // namespace metal_plugin
}  // namespace ov
