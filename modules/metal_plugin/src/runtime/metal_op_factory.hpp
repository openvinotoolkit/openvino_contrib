// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "runtime/metal_memory.hpp"
#include "runtime/metal_op.hpp"

namespace ov {
namespace metal_plugin {

// Factory that maps OpenVINO nodes to concrete MetalOp implementations.
// Currently returns stub ops for a limited set of supported nodes; new ops
// will be added incrementally as their Metal backends land.
class METAL_OP_API MetalOpFactory {
public:
    // Creates a MetalOp for the provided node or returns nullptr if unsupported.
    static std::unique_ptr<MetalOp> create(const std::shared_ptr<const ov::Node>& node,
                                           void* device = nullptr,
                                           void* queue = nullptr);
    // Clone a compiled MetalOp prototype for per-infer execution.
    static std::unique_ptr<MetalOp> clone(const MetalOp& op);
};

}  // namespace metal_plugin
}  // namespace ov
