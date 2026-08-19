// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/shape.hpp"
#include "compiler/stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxMatMulMpsrtLoweringKind {
    None,
    MpsGemm,
    MpsGemmWithMslEpilogue,
};

bool is_supported_mpsrt_matmul_epilogue_activation(ActivationKind kind);

bool can_lower_matmul_to_mpsrt_gemm(const MatMulCodegenDesc& desc);

void annotate_module_with_matmul_mpsrt_epilogue_plan(mlir::ModuleOp module,
                                                     const MatMulCodegenDesc& desc,
                                                     const ov::Shape& shape_a,
                                                     const ov::Shape& shape_b);

GfxMatMulMpsrtLoweringKind annotate_module_with_matmul_mpsrt_plan(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const MatMulCodegenDesc& desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b);

}  // namespace gfx_plugin
}  // namespace ov
