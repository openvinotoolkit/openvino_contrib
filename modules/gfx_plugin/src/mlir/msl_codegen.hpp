// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// Convenience include for MSL codegen helpers backed by MLIR analysis.
#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/codegen_common.hpp"
#include "runtime/gfx_msl_kernel_manifest.hpp"
#include "runtime/gfx_stage_policy.hpp"

#include <string>
#include <string_view>

namespace ov {
namespace gfx_plugin {

std::string normalize_msl_source_for_kernel_plan(std::string source,
                                                 std::string_view current_entry_point,
                                                 const GfxMslKernelPlan& plan);
void configure_msl_kernel_source_for_plan(KernelSource& source,
                                          std::string_view stage_type);

// Attach the Apple placement/storage decision to an MLIR module before MSL
// generation. The attrs are intentionally stage-level and match the MPSRT
// serialization boundary required by the Apple MPS+MSL rewrite.
void annotate_msl_module_with_stage_plan(mlir::ModuleOp module,
                                         const GfxStageOptimizationPlan& plan,
                                         const std::string& stage_type);

}  // namespace gfx_plugin
}  // namespace ov
