// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxAppleStagePipelineOptions {
    GfxStageOptimizationPlan plan;
    std::string stage_type;
    std::string kernel_entry_point;
    bool materialize_typed_program = true;
};

struct GfxAppleStagePipelineResult {
    bool valid = false;
    bool typed_program_materialized = false;
    GfxMpsrtModuleStagePlan stage_plan;
};

std::unique_ptr<mlir::Pass> createGfxAppleCanonicalizePass();
std::unique_ptr<mlir::Pass> createGfxApplePlacementPass(
    const GfxAppleStagePipelineOptions& options);
std::unique_ptr<mlir::Pass> createGfxAppleStorageAssignmentPass(
    const GfxAppleStagePipelineOptions& options);
std::unique_ptr<mlir::Pass> createGfxAppleFusionPass(
    const GfxAppleStagePipelineOptions& options);
std::unique_ptr<mlir::Pass> createGfxAppleStageManifestPass(
    const GfxAppleStagePipelineOptions& options);

GfxAppleStagePipelineResult run_gfx_apple_stage_pipeline(
    mlir::ModuleOp module,
    const GfxAppleStagePipelineOptions& options);

inline GfxAppleStagePipelineResult run_gfx_apple_stage_pipeline(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    std::string_view stage_type,
    std::string_view kernel_entry_point = {},
    bool materialize_typed_program = true) {
    GfxAppleStagePipelineOptions options{};
    options.plan = plan;
    options.stage_type = std::string(stage_type);
    options.kernel_entry_point = std::string(kernel_entry_point);
    options.materialize_typed_program = materialize_typed_program;
    return run_gfx_apple_stage_pipeline(module, options);
}

}  // namespace gfx_plugin
}  // namespace ov
