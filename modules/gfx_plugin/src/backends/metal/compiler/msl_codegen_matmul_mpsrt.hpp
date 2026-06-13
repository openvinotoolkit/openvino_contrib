// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_codegen_desc.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "backends/metal/compiler/apple_mpsrt_matmul_metadata.hpp"
#include "backends/metal/compiler/apple_mpsrt_source_plan.hpp"
#include "openvino/core/shape.hpp"
#include "compiler/stage_policy.hpp"

#include <string>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_matmul_mpsrt_epilogue(const MatMulCodegenDesc& desc);

struct GfxMatMulMpsrtLoweringResult {
    GfxMatMulMpsrtLoweringKind lowering = GfxMatMulMpsrtLoweringKind::None;
    GfxMpsrtKernelSourcePlan mpsrt_plan;

    bool valid() const {
        return lowering != GfxMatMulMpsrtLoweringKind::None && mpsrt_plan.valid();
    }
};

GfxMatMulMpsrtLoweringResult lower_matmul_module_to_mpsrt_plan(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const MatMulCodegenDesc& desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b);

}  // namespace gfx_plugin
}  // namespace ov
