// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_codegen_desc.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/gfx_mpsrt_source_plan.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"

#include <memory>
#include <string_view>

namespace ov {
namespace gfx_plugin {

class GpuBufferManager;

GfxMpsrtKernelSourcePlan make_apple_metal_runtime_matmul_kernel_source_plan(
    mlir::MLIRContext& ctx,
    const GpuBufferManager* buffer_manager,
    const std::shared_ptr<const ov::Node>& node,
    MatMulCodegenDesc desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b,
    std::string_view stage_name);

}  // namespace gfx_plugin
}  // namespace ov
