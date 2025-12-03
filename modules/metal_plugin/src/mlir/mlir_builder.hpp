// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/node.hpp"
#include "kernel_ir/kernel_ir_common.hpp"

namespace ov {
class Model;
namespace metal_plugin {

// Build a minimal MLIR module that wraps a single MatMul using linalg.matmul.
mlir::ModuleOp build_mlir_module_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx);

// Build MLIR module for a single unary activation using linalg.generic.
mlir::ModuleOp build_mlir_unary_from_node(const std::shared_ptr<const ov::Node>& node,
                                          mlir::MLIRContext& ctx,
                                          ActivationKind kind,
                                          float alpha);

// Build MLIR module for Add (possibly with broadcast) using linalg.generic.
mlir::ModuleOp build_mlir_broadcast_add_from_model(const std::shared_ptr<const ov::Model>& model,
                                                   mlir::MLIRContext& ctx);

// Build MLIR module for Softmax (currently last-axis only).
mlir::ModuleOp build_mlir_softmax_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_maxpool_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_avgpool_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_batchnorm_from_model(const std::shared_ptr<const ov::Model>& model,
                                               mlir::MLIRContext& ctx);

}  // namespace metal_plugin
}  // namespace ov
