// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

// Lower auxiliary im2col/weight-pack/restore-output linalg.generic ops
// emitted by the shared Conv -> Im2Col+MatMul rewrite to explicit scf.parallel
// loops after bufferization.
void run_conv_im2col_parallel_lowering(mlir::ModuleOp module);

}  // namespace gfx_plugin
}  // namespace ov
