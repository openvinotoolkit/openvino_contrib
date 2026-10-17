// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

// Fuse trailing parallel post-ops (bias/activation) into the preceding parallel compute loop.
void run_parallel_post_fusion(mlir::ModuleOp module);

}  // namespace gfx_plugin
}  // namespace ov
