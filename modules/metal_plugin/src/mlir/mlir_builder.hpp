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

// Build MLIR modules for binary eltwise ops with broadcast & dynamic shapes.
mlir::ModuleOp build_mlir_add_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_sub_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_mul_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_div_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_pow_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_mod_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_floor_mod_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_squared_difference_from_model(const std::shared_ptr<const ov::Model>& model,
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
mlir::ModuleOp build_mlir_conv3d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_batchnorm_from_model(const std::shared_ptr<const ov::Model>& model,
                                               mlir::MLIRContext& ctx);

// Build MLIR for a single flat Concat copy (one input -> output with axis_offset/axis_len/inner).
mlir::ModuleOp build_mlir_concat_from_op(const KernelOp& op, mlir::MLIRContext& ctx);

// Build MLIR for Interpolate (nearest/bilinear) on NHWC?; here assumes NCHW 2D.
mlir::ModuleOp build_mlir_interpolate_from_op(const KernelOp& op, mlir::MLIRContext& ctx);

// Build MLIR for a single Split slice copy (one output chunk).
mlir::ModuleOp build_mlir_split_from_op(const KernelOp& op, mlir::MLIRContext& ctx);

}  // namespace metal_plugin
}  // namespace ov
