// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_builder_eltwise_common.hpp"
#include "openvino/op/add.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_add_from_node(const std::shared_ptr<const ov::Node>& node,
                                        mlir::MLIRContext& ctx,
                                        const std::vector<MlirInputTransformDesc>& input_transforms) {
    auto add = ov::as_type_ptr<const ov::op::v1::Add>(node);
    OPENVINO_ASSERT(add, "Add MLIR builder: expected Add node");
    return build_mlir_binary_eltwise_from_node<ov::op::v1::Add>(
        add, ctx, input_transforms, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (elem_ty.isIntOrIndex()) {
                return static_cast<mlir::Value>(body.create<mlir::arith::AddIOp>(l, args[0], args[1]));
            }
            return static_cast<mlir::Value>(body.create<mlir::arith::AddFOp>(l, args[0], args[1]));
        });
}

mlir::ModuleOp build_mlir_add_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx) {
    return build_mlir_binary_eltwise_from_model<ov::op::v1::Add>(
        model, ctx, [](mlir::OpBuilder& body, mlir::Location l, mlir::ValueRange args, mlir::Type elem_ty, auto) {
            if (elem_ty.isIntOrIndex())
                return static_cast<mlir::Value>(body.create<mlir::arith::AddIOp>(l, args[0], args[1]));
            return static_cast<mlir::Value>(body.create<mlir::arith::AddFOp>(l, args[0], args[1]));
        });
}

}  // namespace gfx_plugin
}  // namespace ov
