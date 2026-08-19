// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/mlir_stage_backend_hooks.hpp"

namespace ov {
namespace gfx_plugin {

const MlirStageBackendHooks &apple_mlir_stage_backend_hooks();
void ensure_apple_mlir_stage_backend_hooks_registered();

} // namespace gfx_plugin
} // namespace ov
