// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

// Rewrite selected linalg Conv2D ops to an explicit MLIR im2col + matmul form.
// This keeps algorithm selection in the shared MLIR pipeline instead of routing
// heavy general convolutions through a backend-local runtime kernel family.
void run_conv_im2col_matmul_rewrite(mlir::ModuleOp module);

}  // namespace gfx_plugin
}  // namespace ov
