// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

// Lower linalg::Conv2DNchwFchwOp to explicit scf.parallel + scf.for loops.
// Intended to stabilize Vulkan SPIR-V parallel path for Conv2D.
void run_conv2d_parallel_lowering(mlir::ModuleOp module);

}  // namespace gfx_plugin
}  // namespace ov
