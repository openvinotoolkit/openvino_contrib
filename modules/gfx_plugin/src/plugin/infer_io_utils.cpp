// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_io_utils.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

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
    size_t bytes = et.size();
    for (auto d : out_shape) {
        bytes *= d;
    }

    if (backend == GpuBackend::Metal) {
        out_ref.prefer_private = !is_model_output;
        desc.cpu_read = !out_ref.prefer_private;
        desc.cpu_write = !out_ref.prefer_private;
        desc.prefer_device_local = out_ref.prefer_private;
    } else {
        out_ref.prefer_private = true;
        desc.cpu_read = false;
        desc.cpu_write = false;
        desc.prefer_device_local = true;
    }

    desc.bytes = bytes;
    desc.type = et;
    desc.usage = BufferUsage::Intermediate;
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
