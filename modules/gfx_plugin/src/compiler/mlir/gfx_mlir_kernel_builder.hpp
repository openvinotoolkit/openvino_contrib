// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/node.hpp"
#include "compiler/gfx_kernel_plan.hpp"
#include "compiler/gfx_kernel_spec.hpp"

namespace ov {
namespace gfx_plugin {

mlir::ModuleOp build_mlir_for_node(const std::shared_ptr<const ov::Node>& node,
                                   mlir::MLIRContext& ctx);

std::string find_entry_point(mlir::ModuleOp module);

class MlirKernelPlanBuilder {
public:
    KernelPlan build_plan(const std::shared_ptr<const ov::Node>& node,
                          mlir::MLIRContext& ctx,
                          uint32_t arg_count) const;
    KernelPlan build_plan(const KernelSpec& spec, mlir::MLIRContext& ctx) const;
};

}  // namespace gfx_plugin
}  // namespace ov
