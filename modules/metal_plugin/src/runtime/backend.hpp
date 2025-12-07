// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace ov {
namespace metal_plugin {

// Single backend: MLIR-based Metal execution.
class MetalBackend {
public:
    virtual ~MetalBackend() = default;
    virtual void run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) = 0;
    // Enable/disable collection of per-op profiling information.
    virtual void set_profiling(bool /*enable*/) {}
    // Return profiling info from the last inference (may be empty if disabled).
    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const { return {}; }
};

using MetalBackendPtr = std::unique_ptr<MetalBackend>;

}  // namespace metal_plugin
}  // namespace ov
