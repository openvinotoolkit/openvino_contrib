// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "runtime/gfx_input_transform.hpp"
#include "runtime/gpu_types.hpp"
#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_batchnorm.hpp"
#include "runtime/gfx_bias.hpp"

namespace ov {
namespace gfx_plugin {

struct GpuStageSubmitPolicy {
    size_t weight = 1;
    bool isolate = false;
};

// Backend-neutral execution stage interface.
class GpuStage {
public:
    virtual ~GpuStage() = default;

    virtual void init(GpuBufferManager* buffer_manager) = 0;
    virtual void compile(GpuBufferManager* buffer_manager) = 0;
    virtual void execute(GpuCommandBufferHandle command_buffer) = 0;

    virtual void set_inputs(const std::vector<GpuTensor*>& inputs) = 0;
    virtual void set_output(GpuTensor* output) = 0;
    virtual void set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
        if (!outputs.empty()) {
            set_output(outputs.front().get());
        }
    }
    virtual void set_input_transform(size_t /*input_idx*/, const GfxInputTransform& /*transform*/) {}

    virtual bool fuse_activation(ActivationKind /*kind*/, float /*alpha*/) { return false; }
    virtual bool fuse_batchnorm(const BatchNormParams& /*params*/) { return false; }
    virtual bool fuse_bias(const BiasParams& /*params*/) { return false; }

    virtual void enable_profiling(bool /*enable*/) {}
    virtual void set_profiler(void* /*profiler*/,
                              uint32_t /*node_id*/,
                              const std::string& /*node_name*/,
                              const std::string& /*node_type*/) {}
    virtual void on_command_buffer_complete() {}

    virtual const std::string& name() const = 0;
    virtual const std::string& type() const = 0;
    virtual GpuStageSubmitPolicy submit_policy() const { return {}; }

    virtual std::unique_ptr<GpuStage> clone() const = 0;
};

}  // namespace gfx_plugin
}  // namespace ov
