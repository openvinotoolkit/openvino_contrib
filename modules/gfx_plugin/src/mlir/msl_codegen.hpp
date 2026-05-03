// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// Convenience include for MSL codegen helpers backed by MLIR analysis.
#include "kernel_ir/gfx_codegen_backend.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_mpsrt_matmul_metadata.hpp"
#include "mlir/gfx_mpsrt_source_plan.hpp"
#include "openvino/core/shape.hpp"
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
GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan(KernelSource source,
                                                          std::string_view stage_type);

// Attach the Apple placement/storage decision to an MLIR module before MSL
// generation. The attrs are intentionally stage-level and match the MPSRT
// serialization boundary required by the Apple MPS+MSL rewrite.
void annotate_msl_module_with_stage_plan(mlir::ModuleOp module,
                                         const GfxStageOptimizationPlan& plan,
                                         const std::string& stage_type);

// MPSRT custom-kernel source emitters for manifest-backed mixed plans.
// These helpers live at the MSL/codegen boundary so runtime op files do not
// duplicate custom-kernel ABI source generation.
std::string generate_msl_for_matmul_mpsrt_epilogue(const MatMulCodegenDesc& desc);

struct GfxMatMulMpsrtKernelSourcePlan {
    GfxMatMulMpsrtLoweringKind lowering = GfxMatMulMpsrtLoweringKind::None;
    GfxMpsrtKernelSourcePlan mpsrt_plan;
    KernelSource source;
    bool requires_mpsrt_model = false;

    bool valid() const {
        return lowering != GfxMatMulMpsrtLoweringKind::None && mpsrt_plan.valid() && source.module;
    }
};

GfxMatMulMpsrtKernelSourcePlan lower_matmul_module_to_mpsrt_kernel_source(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const MatMulCodegenDesc& desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b);

}  // namespace gfx_plugin
}  // namespace ov
