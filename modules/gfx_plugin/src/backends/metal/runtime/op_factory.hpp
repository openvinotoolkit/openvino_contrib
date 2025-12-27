// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "backends/metal/runtime/memory.hpp"
#include "backends/metal/runtime/op.hpp"

namespace ov {
namespace gfx_plugin {

// Factory that maps OpenVINO nodes to concrete MetalOp implementations.
// Currently returns stub ops for a limited set of supported nodes; new ops
// will be added incrementally as their Metal backends land.
class GFX_OP_API MetalOpFactory {
public:
    // Creates a MetalOp for the provided node or returns nullptr if unsupported.
    static std::unique_ptr<MetalOp> create(const std::shared_ptr<const ov::Node>& node,
                                           void* device = nullptr,
                                           void* queue = nullptr);
    // Clone a compiled MetalOp prototype for per-infer execution.
    static std::unique_ptr<MetalOp> clone(const MetalOp& op);
};

}  // namespace gfx_plugin
}  // namespace ov
