// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

// Lower linalg::Conv3DNcdhwFcdhwOp to explicit scf.for loops with padding bounds checks.
// Intended to avoid explicit padded buffer allocations on Vulkan.
void run_conv3d_parallel_lowering(mlir::ModuleOp module);

}  // namespace gfx_plugin
}  // namespace ov
