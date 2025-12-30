// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/maximum.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_min_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::Minimum>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
                auto pred = it.isUnsigned() ? mlir::arith::CmpIPredicate::ult : mlir::arith::CmpIPredicate::slt;
                auto cmp = body.create<mlir::arith::CmpIOp>(l, pred, args[0], args[1]);
                return static_cast<mlir::Value>(body.create<mlir::arith::SelectOp>(l, cmp, args[0], args[1]));
            }
            auto cmp = body.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OLT, args[0], args[1]);
            return static_cast<mlir::Value>(body.create<mlir::arith::SelectOp>(l, cmp, args[0], args[1]));
        });
}

mlir::ModuleOp build_mlir_max_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::Maximum>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem_ty)) {
                auto pred = it.isUnsigned() ? mlir::arith::CmpIPredicate::ugt : mlir::arith::CmpIPredicate::sgt;
                auto cmp = body.create<mlir::arith::CmpIOp>(l, pred, args[0], args[1]);
                return static_cast<mlir::Value>(body.create<mlir::arith::SelectOp>(l, cmp, args[0], args[1]));
            }
            auto cmp = body.create<mlir::arith::CmpFOp>(l, mlir::arith::CmpFPredicate::OGT, args[0], args[1]);
            return static_cast<mlir::Value>(body.create<mlir::arith::SelectOp>(l, cmp, args[0], args[1]));
        });
}

}  // namespace gfx_plugin
}  // namespace ov
