// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

// Lightweight probe for MLIR coverage without exposing MLIR headers to callers.
bool mlir_supports_node(const std::shared_ptr<const ov::Node>& node);

}  // namespace gfx_plugin
}  // namespace ov
