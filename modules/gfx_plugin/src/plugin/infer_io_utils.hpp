// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/runtime/tensor.hpp"
#include "runtime/gpu_tensor.hpp"
#include "runtime/infer_pipeline.hpp"

namespace ov {
namespace gfx_plugin {

struct OutputBindingResult {
    GpuTensor device_tensor;
    ov::Tensor host_tensor;
};

struct HostInputBinding {
    GpuTensor tensor;
    size_t bytes = 0;
};

struct HostOutputBinding {
    ov::Tensor host;
    size_t bytes = 0;
};

HostInputBinding prepare_host_input_binding(const ov::Tensor& host,
                                            GpuBackend backend,
                                            const char* error_prefix);

void prepare_reusable_host_output_plan(PreparedInferHostOutputPlan& plan,
                                       const PreparedInferOutputPlan& output_plan,
                                       const std::vector<ov::Tensor>& bound_output_hosts);

HostOutputBinding prepare_host_output_binding(const OutputViewInfo& info,
                                              const ov::Tensor* host_override,
                                              ov::Tensor* reusable_host = nullptr);

bool init_stage_output_desc(GpuBackend backend,
                            InferStage& stage,
                            size_t out_idx,
                            GpuTensor& out_ref,
                            GpuBufferDesc& desc,
                            bool is_model_output,
                            bool skip_view_ops,
                            const char* error_prefix);

template <typename OutputLookupFn,
          typename HostOverrideFn,
          typename RemoteSetterFn,
          typename DeviceSetterFn>
inline void bind_outputs_common(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                                const std::shared_ptr<const ov::Model>& runtime_model,
                                const std::unordered_map<const ov::Node*, size_t>& node_map,
                                const std::unordered_map<const ov::Node*, size_t>& param_map,
                                std::vector<InferStage>& pipeline,
                                OutputLookupFn output_input_lookup,
                                std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                                HostOverrideFn host_override,
                                RemoteSetterFn remote_setter,
                                DeviceSetterFn device_setter,
                                const PreparedInferOutputPlan* prepared_plan,
                                bool allow_missing,
                                const char* error_prefix) {
    for_each_output_tensor(public_outputs,
                           runtime_model,
                           node_map,
                           param_map,
                           pipeline,
                           output_input_lookup,
                           remote_outputs,
                           host_override,
                           remote_setter,
                           device_setter,
                           prepared_plan,
                           allow_missing,
                           error_prefix);
}

}  // namespace gfx_plugin
}  // namespace ov
