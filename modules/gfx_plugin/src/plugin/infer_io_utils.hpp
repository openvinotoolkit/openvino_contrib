// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <utility>

#include "openvino/runtime/tensor.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/gpu_tensor.hpp"
#include "infer_pipeline.hpp"

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

HostOutputBinding prepare_host_output_binding(const OutputViewInfo& info,
                                              const ov::Tensor* host_override);

bool init_stage_output_desc(GpuBackend backend,
                            InferStage& stage,
                            size_t out_idx,
                            GpuTensor& out_ref,
                            GpuBufferDesc& desc,
                            bool is_model_output,
                            bool skip_view_ops,
                            const char* error_prefix);

void release_stage_output_handles(std::vector<BufferHandle>& handles, GpuBufferPool& pool);

void prepare_stage_output_handles(std::vector<std::vector<BufferHandle>>& stage_handles,
                                  const std::vector<InferStage>& pipeline,
                                  GpuBufferPool& pool,
                                  bool release_view_only);

template <typename PostBuildFn, typename DescribeOutputFn>
inline std::vector<InferStage> build_pipeline_with_outputs(
    const std::vector<PipelineStageDesc>& descs,
    GpuBufferManager* buffer_manager,
    void* profiler,
    bool profiling_enabled,
    const std::vector<ov::Output<const ov::Node>>& outputs,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
    const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_inputs,
    GpuBackend expected_backend,
    GpuBufferPool& pool,
    std::vector<std::vector<BufferHandle>>& stage_handles,
    PostBuildFn&& post_build,
    DescribeOutputFn&& describe_output,
    const char* error_prefix) {
    auto pipeline = build_bound_pipeline(descs,
                                         buffer_manager,
                                         profiler,
                                         profiling_enabled,
                                         outputs,
                                         node_map,
                                         param_map,
                                         remote_outputs,
                                         remote_inputs,
                                         expected_backend,
                                         error_prefix);
    post_build(pipeline);
    prepare_stage_output_handles(stage_handles, pipeline, pool, /*release_view_only=*/true);
    allocate_stage_outputs(pipeline,
                           stage_handles,
                           pool,
                           std::forward<DescribeOutputFn>(describe_output),
                           error_prefix);
    return pipeline;
}

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
                                bool allow_missing,
                                bool allow_fallback_one,
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
                           allow_missing,
                           allow_fallback_one,
                           error_prefix);
}

}  // namespace gfx_plugin
}  // namespace ov
