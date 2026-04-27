// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "mlir/IR/BuiltinOps.h"

#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_batchnorm.hpp"
#include "runtime/gfx_bias.hpp"

namespace ov {
namespace gfx_plugin {

bool apply_fused_activation(mlir::ModuleOp module, ActivationKind kind, float alpha);

bool apply_fused_input_activation(mlir::ModuleOp module, size_t input_idx, ActivationKind kind, float alpha);

bool apply_fused_batchnorm(mlir::ModuleOp module, const BatchNormParams& params);

bool apply_fused_bias(mlir::ModuleOp module, const BiasParams& params);

}  // namespace gfx_plugin
}  // namespace ov
