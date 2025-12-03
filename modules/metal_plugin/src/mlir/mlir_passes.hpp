// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace metal_plugin {

void run_mlir_pipeline(mlir::ModuleOp module);

}  // namespace metal_plugin
}  // namespace ov

