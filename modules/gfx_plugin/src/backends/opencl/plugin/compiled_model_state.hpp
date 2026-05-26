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
#include "plugin/backend_state.hpp"

namespace ov {
namespace gfx_plugin {

struct OpenClBackendState final : BackendState {
    std::shared_ptr<OpenClRuntimeContext> context;
    std::shared_ptr<OpenClBufferManager> const_manager;

    GpuBackend backend() const override { return GpuBackend::OpenCL; }
    BackendResources resources() const override {
        return {reinterpret_cast<GpuDeviceHandle>(context ? context->device() : nullptr),
                reinterpret_cast<GpuCommandQueueHandle>(context ? context->queue() : nullptr),
                const_manager.get()};
    }
    bool requires_const_manager() const override { return true; }
    bool has_const_manager() const override { return const_manager != nullptr; }
    void init_infer_state(InferRequestState& state) const override;
    std::unique_ptr<GpuStage> create_stage(const std::shared_ptr<const ov::Node>& node) const override {
        return create_opencl_stage(node,
                                   reinterpret_cast<GpuDeviceHandle>(context ? context->device() : nullptr),
                                   reinterpret_cast<GpuCommandQueueHandle>(context ? context->queue() : nullptr));
    }
    std::unique_ptr<GpuStage> create_stage(
        const std::shared_ptr<const ov::Node>& node,
        const RuntimeStageExecutableDescriptor* descriptor) const override {
        if (descriptor && descriptor->payload_kind == compiler::KernelArtifactPayloadKind::OpenClSource &&
            descriptor->backend_domain == "opencl") {
            OPENVINO_ASSERT(descriptor->payload,
                            "GFX OpenCL: runtime descriptor is missing compiler-owned OpenCL source payload for ",
                            node ? node->get_type_name() : "<null>");
            if (const auto* payload =
                    dynamic_cast<const GfxOpenClSourceArtifactPayload*>(
                        descriptor->payload.get())) {
                return OpenClRuntimeKernelLoader(context ? context : OpenClRuntimeContext::instance())
                    .load_source_stage(node, *descriptor, payload->artifact());
            }
            OPENVINO_THROW("GFX OpenCL: runtime descriptor has non-OpenCL source payload for ",
                           node ? node->get_type_name() : "<null>");
        }
        return create_stage(node);
    }
    std::unique_ptr<GfxProfiler> create_profiler(const GfxProfilerConfig& cfg) const override;
};

}  // namespace gfx_plugin
}  // namespace ov
