// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/gfx_mlir_kernel_metadata.hpp"

namespace ov {
namespace gfx_plugin {

inline void annotate_spirv_kernel_binding_attrs(
    mlir::ModuleOp module, const KernelRuntimeBindingState& binding) {
  if (!module) {
    return;
  }
  annotate_kernel_operand_abi_attrs_for_spirv_adapter(module, binding);
}

inline void replace_spirv_kernel_binding_attrs(
    mlir::ModuleOp module, const std::vector<int32_t>& operand_kinds,
    const std::vector<int32_t>& operand_arg_indices,
    const std::vector<int32_t>& scalar_values = {},
    const std::vector<int32_t>& scalar_args = {},
    bool update_scalar_args = true) {
  if (!module) {
    return;
  }
  OPENVINO_ASSERT(operand_kinds.empty() || operand_arg_indices.empty() ||
                      operand_kinds.size() == operand_arg_indices.size(),
                  "GFX MLIR: SPIR-V kernel operand adapter attr sizes mismatch");
  if (!operand_kinds.empty()) {
    module->setAttr("gfx.kernel_operand_kinds",
                    make_kernel_i32_array_attr(module.getContext(),
                                               operand_kinds));
  } else {
    module->removeAttr("gfx.kernel_operand_kinds");
  }
  if (!operand_arg_indices.empty()) {
    module->setAttr("gfx.kernel_operand_arg_indices",
                    make_kernel_i32_array_attr(module.getContext(),
                                               operand_arg_indices));
  } else {
    module->removeAttr("gfx.kernel_operand_arg_indices");
  }
  if (!scalar_values.empty()) {
    module->setAttr("gfx.kernel_scalar_values",
                    make_kernel_i32_array_attr(module.getContext(),
                                               scalar_values));
  } else {
    module->removeAttr("gfx.kernel_scalar_values");
  }
  if (update_scalar_args) {
    if (!scalar_args.empty()) {
      module->setAttr("gfx.kernel_scalar_args",
                      make_kernel_i32_array_attr(module.getContext(),
                                                 scalar_args));
    } else {
      module->removeAttr("gfx.kernel_scalar_args");
    }
  }
}

inline void replace_spirv_kernel_operand_arg_indices(
    mlir::ModuleOp module, const std::vector<int32_t>& operand_arg_indices) {
  if (!module) {
    return;
  }
  if (!operand_arg_indices.empty()) {
    module->setAttr("gfx.kernel_operand_arg_indices",
                    make_kernel_i32_array_attr(module.getContext(),
                                               operand_arg_indices));
  } else {
    module->removeAttr("gfx.kernel_operand_arg_indices");
  }
}

inline void restore_spirv_kernel_binding_attrs_if_missing(
    mlir::ModuleOp module, mlir::Attribute operand_kinds,
    mlir::Attribute operand_arg_indices, mlir::Attribute scalar_values,
    mlir::Attribute scalar_args) {
  if (!module) {
    return;
  }
  if (operand_kinds && !module->getAttr("gfx.kernel_operand_kinds")) {
    module->setAttr("gfx.kernel_operand_kinds", operand_kinds);
  }
  if (operand_arg_indices &&
      !module->getAttr("gfx.kernel_operand_arg_indices")) {
    module->setAttr("gfx.kernel_operand_arg_indices", operand_arg_indices);
  }
  if (scalar_values && !module->getAttr("gfx.kernel_scalar_values")) {
    module->setAttr("gfx.kernel_scalar_values", scalar_values);
  }
  if (scalar_args && !module->getAttr("gfx.kernel_scalar_args")) {
    module->setAttr("gfx.kernel_scalar_args", scalar_args);
  }
}

inline void restore_spirv_kernel_binding_vectors_if_missing(
    mlir::ModuleOp module, const std::vector<int32_t>& operand_kinds,
    const std::vector<int32_t>& operand_arg_indices,
    const std::vector<int32_t>& scalar_values,
    const std::vector<int32_t>& scalar_args) {
  if (!module) {
    return;
  }
  if (!operand_kinds.empty() &&
      !module->getAttr("gfx.kernel_operand_kinds")) {
    module->setAttr("gfx.kernel_operand_kinds",
                    make_kernel_i32_array_attr(module.getContext(),
                                               operand_kinds));
  }
  if (!operand_arg_indices.empty() &&
      !module->getAttr("gfx.kernel_operand_arg_indices")) {
    module->setAttr("gfx.kernel_operand_arg_indices",
                    make_kernel_i32_array_attr(module.getContext(),
                                               operand_arg_indices));
  }
  if (!scalar_values.empty() &&
      !module->getAttr("gfx.kernel_scalar_values")) {
    module->setAttr("gfx.kernel_scalar_values",
                    make_kernel_i32_array_attr(module.getContext(),
                                               scalar_values));
  }
  if (!scalar_args.empty() && !module->getAttr("gfx.kernel_scalar_args")) {
    module->setAttr("gfx.kernel_scalar_args",
                    make_kernel_i32_array_attr(module.getContext(),
                                               scalar_args));
  }
}

}  // namespace gfx_plugin
}  // namespace ov
