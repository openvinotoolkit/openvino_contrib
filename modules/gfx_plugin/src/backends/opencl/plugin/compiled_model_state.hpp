// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <utility>

#include "backends/opencl/runtime/opencl_api.hpp"
#include "backends/opencl/runtime/opencl_buffer_manager.hpp"
#include "backends/opencl/runtime/opencl_runtime_kernel_loader.hpp"
#include "backends/opencl/runtime/stage_factory.hpp"
#include "openvino/core/except.hpp"
#include "runtime/backend_runtime.hpp"

namespace ov {
namespace gfx_plugin {

struct OpenClBackendState final : BackendState {
  compiler::BackendTarget runtime_target;
  std::shared_ptr<OpenClRuntimeContext> context;
  std::shared_ptr<OpenClBufferManager> const_manager;

  const compiler::BackendTarget &target() const override {
    return runtime_target;
  }
  GpuBackend backend() const override { return runtime_target.backend(); }
  BackendResources resources() const override {
    return {reinterpret_cast<GpuDeviceHandle>(context ? context->device()
                                                      : nullptr),
            reinterpret_cast<GpuCommandQueueHandle>(context ? context->queue()
                                                            : nullptr),
            const_manager.get()};
  }
  bool requires_const_manager() const override { return true; }
  bool has_const_manager() const override { return const_manager != nullptr; }
  void init_infer_state(BackendRequestState &state) const override;
  std::unique_ptr<GpuStage>
  create_stage(const RuntimeStageMaterializationContext &materialization)
      const override {
    const auto &descriptor = materialization.require_descriptor();
    if (descriptor.payload_kind == KernelArtifactPayloadKind::OpenClSource &&
        descriptor.backend_domain == "opencl") {
      OPENVINO_ASSERT(descriptor.payload,
                      "GFX OpenCL: runtime descriptor is missing "
                      "compiler-owned OpenCL source payload for ",
                      materialization.op_type_name());
      if (const auto *payload =
              dynamic_cast<const GfxOpenClSourceArtifactPayload *>(
                  descriptor.payload.get())) {
        std::shared_ptr<const ov::Node> node;
        if (descriptor.temporary_source_node_bridge_required) {
          node = materialization.require_source_node(
              descriptor.temporary_source_node_bridge_reason);
        }
        return OpenClRuntimeKernelLoader(
                   context ? context : OpenClRuntimeContext::instance())
            .load_source_stage(descriptor, payload->artifact(), node);
      }
      OPENVINO_THROW(
          "GFX OpenCL: runtime descriptor has non-OpenCL source payload for ",
          materialization.op_type_name());
    }
    return create_opencl_stage(materialization,
                               reinterpret_cast<GpuDeviceHandle>(
                                   context ? context->device() : nullptr),
                               reinterpret_cast<GpuCommandQueueHandle>(
                                   context ? context->queue() : nullptr));
  }
  std::unique_ptr<GfxProfiler>
  create_profiler(const GfxProfilerConfig &cfg) const override;
};

} // namespace gfx_plugin
} // namespace ov
