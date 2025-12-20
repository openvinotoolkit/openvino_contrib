// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/greater_eq.hpp"

namespace ov {
namespace metal_plugin {

namespace {

template <typename OVOp, mlir::arith::CmpFPredicate FP, mlir::arith::CmpIPredicate IP>
mlir::ModuleOp build_cmp(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<OVOp>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (elem_ty.isIntOrIndex()) {
                auto cmp = body.create<mlir::arith::CmpIOp>(l, IP, args[0], args[1]);
                return static_cast<mlir::Value>(cmp);
            }
            auto cmp = body.create<mlir::arith::CmpFOp>(l, FP, args[0], args[1]);
            return static_cast<mlir::Value>(cmp);
        });
}

}  // namespace

mlir::ModuleOp build_mlir_equal_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_cmp<ov::op::v1::Equal, mlir::arith::CmpFPredicate::OEQ, mlir::arith::CmpIPredicate::eq>(model, ctx);
}

mlir::ModuleOp build_mlir_not_equal_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_cmp<ov::op::v1::NotEqual, mlir::arith::CmpFPredicate::ONE, mlir::arith::CmpIPredicate::ne>(model, ctx);
}

mlir::ModuleOp build_mlir_less_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_cmp<ov::op::v1::Less, mlir::arith::CmpFPredicate::OLT, mlir::arith::CmpIPredicate::slt>(model, ctx);
}

mlir::ModuleOp build_mlir_greater_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_cmp<ov::op::v1::Greater, mlir::arith::CmpFPredicate::OGT, mlir::arith::CmpIPredicate::sgt>(model, ctx);
}

mlir::ModuleOp build_mlir_less_equal_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_cmp<ov::op::v1::LessEqual, mlir::arith::CmpFPredicate::OLE, mlir::arith::CmpIPredicate::sle>(model, ctx);
}

mlir::ModuleOp build_mlir_greater_equal_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_cmp<ov::op::v1::GreaterEqual, mlir::arith::CmpFPredicate::OGE, mlir::arith::CmpIPredicate::sge>(model, ctx);
}

}  // namespace metal_plugin
}  // namespace ov
