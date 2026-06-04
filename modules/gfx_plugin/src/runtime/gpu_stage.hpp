// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/gpu_parallelism_profile.hpp"
#include "common/gpu_stage_submit_policy.hpp"
#include "common/gfx_activation.hpp"
#include "runtime/gfx_batchnorm.hpp"
#include "common/gfx_bias.hpp"
#include "runtime/gfx_input_transform.hpp"
#include "runtime/gpu_types.hpp"

namespace ov {
namespace gfx_plugin {

struct GpuStageRuntimeOptions {
  bool diagnostic_f32_vendor_image = false;
  bool source_kernel_dispatch_enabled = false;
  GpuParallelismProfile source_kernel_fallback_parallelism{};
};

// Backend-neutral execution stage interface.
class GpuStage {
public:
  virtual ~GpuStage() = default;

  virtual void init(GpuBufferManager *buffer_manager) = 0;
  virtual void prepare_runtime_handle(GpuBufferManager *buffer_manager) = 0;
  virtual void execute(GpuCommandBufferHandle command_buffer) = 0;
  virtual void prewarm_runtime_state() {}

  virtual void set_inputs(const std::vector<GpuTensor *> &inputs) = 0;
  virtual void set_output(GpuTensor *output) = 0;
  virtual void set_output_refs(const std::vector<GpuTensor *> &outputs) {
    if (!outputs.empty()) {
      set_output(outputs.front());
    }
  }
  virtual void
  set_outputs(const std::vector<std::unique_ptr<GpuTensor>> &outputs) {
    std::vector<GpuTensor *> refs;
    refs.reserve(outputs.size());
    for (const auto &output : outputs) {
      refs.push_back(output.get());
    }
    set_output_refs(refs);
  }
  virtual void set_input_transform(size_t /*input_idx*/,
                                   const GfxInputTransform & /*transform*/) {}

  virtual bool fuse_activation(ActivationKind /*kind*/, float /*alpha*/) {
    return false;
  }
  virtual bool fuse_input_activation(size_t /*input_idx*/,
                                     ActivationKind /*kind*/, float /*alpha*/) {
    return false;
  }
  virtual bool fuse_residual_add() { return false; }
  virtual bool fuse_batchnorm(const BatchNormParams & /*params*/) {
    return false;
  }
  virtual bool fuse_bias(const BiasParams & /*params*/) { return false; }

  virtual void set_runtime_options(const GpuStageRuntimeOptions & /*options*/) {
  }
  virtual void enable_profiling(bool /*enable*/) {}
  virtual void set_profiler(void * /*profiler*/, uint32_t /*node_id*/,
                            const std::string & /*node_name*/,
                            const std::string & /*node_type*/) {}
  virtual void on_command_buffer_complete() {}

  virtual const std::string &name() const = 0;
  virtual const std::string &type() const = 0;
  virtual GpuStageSubmitPolicy submit_policy() const { return {}; }
  virtual bool has_internal_input_bindings() const { return false; }

  virtual std::unique_ptr<GpuStage> clone() const = 0;
};

} // namespace gfx_plugin
} // namespace ov
