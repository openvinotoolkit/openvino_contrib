// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "runtime/metal_memory.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace ov {
namespace metal_plugin {

// Single backend: MLIR-based Metal execution.
class MetalBackend {
public:
    virtual ~MetalBackend() = default;
    virtual void run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) = 0;
    // Optional device-native execution path. Returns true if handled on device.
    virtual bool run_device(MetalTensorMap& /*tensors*/, MetalBufferManager& /*mgr*/) { return false; }
    // Factory for per-plugin buffer manager bound to the backend device.
    virtual std::shared_ptr<MetalBufferManager> create_buffer_manager() { return nullptr; }
    // Upload model-lifetime constants to the provided buffer manager.
    virtual void preload_constants(MetalBufferManager& /*mgr*/) {}
    // Enable/disable collection of per-op profiling information.
    virtual void set_profiling(bool /*enable*/) {}
    // Return profiling info from the last inference (may be empty if disabled).
    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const { return {}; }
};

using MetalBackendPtr = std::unique_ptr<MetalBackend>;

}  // namespace metal_plugin
}  // namespace ov
