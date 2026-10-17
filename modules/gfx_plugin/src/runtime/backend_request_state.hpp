// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "runtime/gfx_profiler.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/infer_pipeline_state.hpp"

namespace ov {
namespace gfx_plugin {

struct BackendInferState {
    virtual ~BackendInferState() = default;
    std::unique_ptr<GfxProfiler> profiler;
    std::vector<InferStage> reusable_pipeline;
    PreparedInferExecutionPlan reusable_execution_plan;
    PreparedInferOutputPlan reusable_output_plan;
    PreparedInferHostOutputPlan reusable_host_output_plan;
    bool reusable_pipeline_runtime_prewarmed = false;
    std::vector<std::vector<BufferHandle>> stage_output_handles;
    StageOutputBufferWorkspace stage_output_workspace;
    std::vector<BufferHandle> input_handles;
    std::vector<BufferHandle> input_staging_handles;
    std::vector<BufferHandle> output_staging_handles;
    std::vector<GpuTensor> output_tensors;
};

struct BackendRequestState {
    std::unique_ptr<BackendInferState> backend;
};

}  // namespace gfx_plugin
}  // namespace ov
