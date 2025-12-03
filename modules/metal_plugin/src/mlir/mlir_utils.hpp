// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace metal_plugin {

void extract_matmul_shape(mlir::ModuleOp module, int64_t& M, int64_t& N, int64_t& K);

}  // namespace metal_plugin
}  // namespace ov

