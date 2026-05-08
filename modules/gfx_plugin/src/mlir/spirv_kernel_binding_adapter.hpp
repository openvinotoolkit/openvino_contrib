// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "kernel_ir/gfx_kernel_plan.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/codegen_common.hpp"
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

inline void annotate_spirv_fixed_arg_count(mlir::ModuleOp module,
                                           size_t arg_count) {
  if (!module) {
    return;
  }
  mlir::OpBuilder builder(module.getContext());
  module->setAttr(
      "gfx.fixed_arg_count",
      builder.getI32IntegerAttr(static_cast<int32_t>(arg_count)));
}

inline void annotate_spirv_fixed_arg_entry_metadata(
    mlir::ModuleOp module, size_t output_arg_count) {
  if (!module) {
    return;
  }
  auto fixed_arg_count =
      module->getAttrOfType<mlir::IntegerAttr>("gfx.fixed_arg_count");
  if (!fixed_arg_count) {
    return;
  }

  const size_t total_buffer_args =
      static_cast<size_t>(std::max<int64_t>(fixed_arg_count.getInt(), 0));
  const size_t non_output_buffer_args =
      total_buffer_args >= output_arg_count
          ? (total_buffer_args - output_arg_count)
          : 0;
  if (auto func = get_entry_func(module)) {
    const size_t annotate_count =
        std::min(non_output_buffer_args,
                 static_cast<size_t>(func.getNumArguments()));
    mlir::OpBuilder builder(module.getContext());
    for (size_t arg_idx = 0; arg_idx < annotate_count; ++arg_idx) {
      func.setArgAttr(static_cast<unsigned>(arg_idx),
                      "gfx.kernel_runtime_arg_index",
                      builder.getI32IntegerAttr(static_cast<int32_t>(arg_idx)));
    }
  }

  mlir::OpBuilder builder(module.getContext());
  module->setAttr(
      "gfx.kernel_output_arg_count",
      builder.getI32IntegerAttr(static_cast<int32_t>(output_arg_count)));
}

inline KernelPlan make_spirv_fixed_arg_kernel_plan(mlir::ModuleOp module,
                                                   std::string entry_point,
                                                   size_t arg_count) {
  return KernelPlan(module, std::move(entry_point),
                    static_cast<uint32_t>(arg_count));
}

}  // namespace gfx_plugin
}  // namespace ov
