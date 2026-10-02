// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace gfx_plugin {

// Lightweight probe for MLIR coverage without exposing MLIR headers to callers.
bool mlir_supports_node(const std::shared_ptr<const ov::Node>& node);
mlir::MLIRContext& gfx_mlir_context();

}  // namespace gfx_plugin
}  // namespace ov
