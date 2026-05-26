// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

GfxMslGeneratedKernelSourcePlan make_shapeof_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node,
    mlir::ModuleOp module = {});

GfxMslGeneratedKernelSourcePlan make_tile_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node,
    mlir::ModuleOp module = {});

GfxMslGeneratedKernelSourcePlan make_range_msl_kernel_source_plan(
    const std::shared_ptr<const ov::Node> &node,
    mlir::ModuleOp module = {});

} // namespace gfx_plugin
} // namespace ov
