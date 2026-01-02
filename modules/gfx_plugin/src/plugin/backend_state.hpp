// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

#include "runtime/gfx_profiler.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_buffer_manager.hpp"

namespace ov {
namespace gfx_plugin {

class GpuStage;

struct InferRequestState;

struct BackendResources {
    GpuDeviceHandle device = nullptr;
    GpuCommandQueueHandle queue = nullptr;
    GpuBufferManager* const_manager = nullptr;
};

struct BackendState {
    virtual ~BackendState() = default;
    virtual GpuBackend backend() const = 0;
    virtual BackendResources resources() const = 0;
    // Return true only for backends that require a const buffer manager to build/compile stages.
    virtual bool requires_const_manager() const { return false; }
    virtual bool has_const_manager() const = 0;
    virtual void release() {}
    virtual void init_infer_state(InferRequestState& /*state*/) const {}
    virtual std::unique_ptr<GfxProfiler> create_profiler(const GfxProfilerConfig& /*cfg*/) const { return {}; }
    virtual std::unique_ptr<GpuStage> create_stage(const std::shared_ptr<const ov::Node>& node) const = 0;
    virtual ov::SoPtr<ov::ITensor> get_tensor_override(
        const InferRequestState& /*state*/,
        size_t /*idx*/,
        const std::vector<ov::Output<const ov::Node>>& /*outputs*/) const {
        return {};
    }
    virtual ov::Any get_mem_stats() const { return {}; }
    virtual void set_mem_stats(const ov::Any& /*stats*/) const {}
};

}  // namespace gfx_plugin
}  // namespace ov
