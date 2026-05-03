// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// Convenience include for MSL codegen helpers backed by MLIR analysis.
#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_kernel_spec.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_mpsrt_matmul_metadata.hpp"
#include "mlir/gfx_mpsrt_source_plan.hpp"
#include "mlir/IR/MLIRContext.h"
#include "openvino/core/shape.hpp"
#include "runtime/gfx_msl_kernel_manifest.hpp"
#include "runtime/gfx_stage_policy.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ov {
namespace gfx_plugin {

std::string normalize_msl_source_for_kernel_plan(std::string source,
                                                 std::string_view current_entry_point,
                                                 const GfxMslKernelPlan& plan);
void configure_msl_kernel_source_for_plan(KernelSource& source,
                                          std::string_view stage_type);
GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan(KernelSource source,
                                                          std::string_view stage_type);
void configure_msl_kernel_source_for_node(KernelSource& source,
                                          const std::shared_ptr<const ov::Node>& node,
                                          const GpuBufferManager* buffer_manager,
                                          std::string_view stage_type,
                                          bool has_bias,
                                          bool has_activation,
                                          bool has_batchnorm);
void configure_msl_kernel_source_for_spec(KernelSource& source,
                                          const KernelSpec& spec,
                                          const GpuBufferManager* buffer_manager,
                                          std::string_view entry_point);

// Attach the Apple placement/storage decision to an MLIR module before MSL
// generation. The attrs are intentionally stage-level and match the MPSRT
// serialization boundary required by the Apple MPS+MSL rewrite.
void annotate_msl_module_with_stage_plan(mlir::ModuleOp module,
                                         const GfxStageOptimizationPlan& plan,
                                         const std::string& stage_type,
                                         std::string_view kernel_entry_point = {});

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

enum class GfxMatMulMetalKernelSourcePlanKind {
    None,
    Mpsrt,
    MslFallback,
};

struct GfxMatMulMetalKernelSourcePlan {
    GfxMatMulMetalKernelSourcePlanKind kind = GfxMatMulMetalKernelSourcePlanKind::None;
    GfxMatMulMpsrtLoweringKind mpsrt_lowering = GfxMatMulMpsrtLoweringKind::None;
    GfxMpsrtKernelSourcePlan mpsrt_plan;
    KernelSource source;
    bool requires_mpsrt_model = false;

    bool valid() const {
        return kind != GfxMatMulMetalKernelSourcePlanKind::None && source.module;
    }

    bool uses_mpsrt_gemm() const {
        return kind == GfxMatMulMetalKernelSourcePlanKind::Mpsrt;
    }
};

GfxMatMulMetalKernelSourcePlan lower_matmul_node_to_metal_kernel_source(
    mlir::MLIRContext& ctx,
    const GpuBufferManager* buffer_manager,
    const std::shared_ptr<const ov::Node>& node,
    MatMulCodegenDesc desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b);

}  // namespace gfx_plugin
}  // namespace ov
