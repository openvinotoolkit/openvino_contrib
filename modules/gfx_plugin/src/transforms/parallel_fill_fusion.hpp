// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

// Fuse a zero-fill scf.parallel directly into the following compute scf.parallel.
void run_parallel_fill_fusion(mlir::ModuleOp module);

}  // namespace gfx_plugin
}  // namespace ov
