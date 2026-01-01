// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_io_utils.hpp"

#include "openvino/core/except.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

HostInputBinding prepare_host_input_binding(const ov::Tensor& host,
                                            GpuBackend backend,
                                            const char* error_prefix) {
    OPENVINO_ASSERT(host && host.data(), error_prefix, ": input host tensor is empty");
    HostInputBinding binding{};
    binding.bytes = host.get_byte_size();
    binding.tensor.shape = host.get_shape();
    binding.tensor.expected_type = host.get_element_type();
    binding.tensor.buf.type = host.get_element_type();
    binding.tensor.buf.backend = backend;
    return binding;
}

HostOutputBinding prepare_host_output_binding(const OutputViewInfo& info,
                                              const ov::Tensor* host_override) {
    HostOutputBinding binding{};
    binding.bytes = tensor_byte_size(info.shape, info.type);
    if (host_override && *host_override) {
        binding.host = *host_override;
    } else {
        binding.host = ov::Tensor(info.type, info.shape);
    }
    return binding;
}

bool init_stage_output_desc(GpuBackend backend,
                            InferStage& stage,
                            size_t out_idx,
                            GpuTensor& out_ref,
                            GpuBufferDesc& desc,
                            bool is_model_output,
                            bool skip_view_ops,
                            const char* error_prefix) {
    if (skip_view_ops && is_view_op(stage)) {
        return false;
    }
    const auto out_shape = ensure_stage_output_shape(stage, out_idx);
    if (out_shape.empty()) {
        return false;
    }
    const auto et = resolve_stage_output_type(stage, out_ref, out_idx, error_prefix);
    size_t bytes = tensor_byte_size(out_shape, et);

    const bool prefer_private = !is_model_output;
    out_ref.prefer_private = prefer_private;
    desc.cpu_read = !prefer_private;
    desc.cpu_write = !prefer_private;
    desc.prefer_device_local = prefer_private;

    desc.bytes = bytes;
    desc.type = et;
    desc.usage = is_model_output ? BufferUsage::IO : BufferUsage::Intermediate;
    return true;
}

void release_stage_output_handles(std::vector<BufferHandle>& handles, GpuBufferPool& pool) {
    for (auto& handle : handles) {
        pool.release(handle);
    }
}

void prepare_stage_output_handles(std::vector<std::vector<BufferHandle>>& stage_handles,
                                  const std::vector<InferStage>& pipeline,
                                  GpuBufferPool& pool,
                                  bool release_view_only) {
    if (stage_handles.size() != pipeline.size()) {
        stage_handles.assign(pipeline.size(), {});
    }
    for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
        const auto& stage = pipeline[stage_idx];
        if (release_view_only && !is_view_op(stage)) {
            continue;
        }
        auto& handles = stage_handles[stage_idx];
        if (handles.size() < stage.outputs.size()) {
            handles.resize(stage.outputs.size());
        }
        release_stage_output_handles(handles, pool);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
