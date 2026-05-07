// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "runtime/gfx_mpsrt_builder_plan.hpp"

namespace ov {
namespace gfx_plugin {

// Apple/MPSRT-specific lowering boundary from canonical stage metadata to the
// transitional runtime builder ABI consumed by the Metal MPSRT runtime.
void populate_gfx_apple_mpsrt_runtime_abi_pipeline(mlir::PassManager& pm);

std::unique_ptr<mlir::Pass> createGfxAppleMpsrtRuntimeAbiCallPlanPass();

bool has_gfx_apple_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module);

bool materialize_gfx_apple_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module);

bool rematerialize_gfx_apple_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module);

bool read_gfx_apple_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module,
                                                GfxMpsrtBuilderPlan& out);

}  // namespace gfx_plugin
}  // namespace ov
