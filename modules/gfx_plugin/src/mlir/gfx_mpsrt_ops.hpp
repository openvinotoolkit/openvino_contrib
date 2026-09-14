// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace gfx_plugin {

struct GfxMpsrtProgram;

inline constexpr const char* kGfxMpsrtOpsSymbol = "gfx_mpsrt_ops";

void erase_module_mpsrt_ops(mlir::ModuleOp module);

void erase_module_mpsrt_legacy_attrs(mlir::ModuleOp module);

bool module_has_mpsrt_ops_program(mlir::ModuleOp module);

bool materialize_module_mpsrt_ops(mlir::ModuleOp module,
                                  const GfxMpsrtProgram& program);

bool read_module_mpsrt_ops_program(mlir::ModuleOp module,
                                   GfxMpsrtProgram& out);

}  // namespace gfx_plugin
}  // namespace ov
