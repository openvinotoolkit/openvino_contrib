// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

// Lower an MLIR module to SPIR-V binary. Returns empty on failure and fills log.
std::vector<uint32_t> lower_to_spirv(mlir::ModuleOp module,
                                     const std::string& entry_point,
                                     std::string* log = nullptr);

// Build a minimal SPIR-V module with a no-op kernel.
std::vector<uint32_t> build_stub_spirv(const std::string& entry_point, std::string* log = nullptr);

}  // namespace gfx_plugin
}  // namespace ov
