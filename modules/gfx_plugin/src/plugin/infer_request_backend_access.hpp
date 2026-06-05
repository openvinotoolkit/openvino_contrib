// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/gfx_plugin/infer_request.hpp"

#include "plugin/infer_request_state.hpp"

namespace ov {
namespace gfx_plugin {

struct InferRequestBackendAccess final {
    static InferRequestState& state(InferRequest& request) {
        OPENVINO_ASSERT(request.m_state, "GFX: infer request state is not initialized");
        return *request.m_state;
    }

    static const std::vector<ov::Output<const ov::Node>>& outputs(InferRequest& request) {
        return request.get_outputs();
    }

    static void bind_inputs_before_infer(
        InferRequest& request,
        const compiler::BackendTarget& expected_target,
        std::vector<GpuTensor>& input_tensors,
        const std::function<GpuTensor(size_t, const ov::Tensor&, BufferHandle*)>& host_binder,
        const std::function<void(size_t, const GpuTensor&)>& device_result_handler,
        GfxProfiler* profiler,
        bool profiling,
        bool with_staging,
        const char* error_prefix) {
        request.bind_inputs_before_infer(expected_target,
                                         input_tensors,
                                         host_binder,
                                         device_result_handler,
                                         profiler,
                                         profiling,
                                         with_staging,
                                         error_prefix);
    }

    static void bind_outputs_after_infer(
        InferRequest& request,
        const std::shared_ptr<const CompiledModel>& cm,
        std::vector<InferStage>& pipeline,
        const std::function<GpuTensor*(size_t)>& output_input_lookup,
        const std::function<OutputBindingResult(size_t,
                                                GpuTensor&,
                                                const OutputViewInfo&,
                                                const ov::Tensor*,
                                                ov::Tensor*,
                                                BufferHandle*)>& device_binder,
        const std::function<void(size_t, const OutputBindingResult&)>& device_result_handler,
        GfxProfiler* profiler,
        bool profiling,
        const char* error_prefix) {
        request.bind_outputs_after_infer(cm,
                                         pipeline,
                                         output_input_lookup,
                                         device_binder,
                                         device_result_handler,
                                         profiler,
                                         profiling,
                                         error_prefix);
    }
};

}  // namespace gfx_plugin
}  // namespace ov
