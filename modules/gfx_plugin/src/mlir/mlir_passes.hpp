// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

void run_mlir_pipeline(mlir::ModuleOp module, bool use_alloca = true, bool use_parallel_loops = false);

}  // namespace gfx_plugin
}  // namespace ov
