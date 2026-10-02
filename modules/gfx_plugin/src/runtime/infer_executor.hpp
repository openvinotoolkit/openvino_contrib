// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "runtime/backend_request_state.hpp"
#include "runtime/infer_pipeline.hpp"
#include "runtime/infer_submission.hpp"
#include "runtime/runtime_execution_plan.hpp"

namespace ov {
namespace gfx_plugin {

using InferPipelinePostPrepareHook =
    std::function<void(std::vector<InferStage>&)>;
using InferStageOutputDescInitializer = StageOutputDescInitializer;

struct InferRuntimeExecutionConfig {
    BackendInferState* state = nullptr;
    std::shared_ptr<const RuntimeExecutionPlan> execution_plan;
    GpuBufferManager* buffer_manager = nullptr;
    void* stage_profiler = nullptr;
    bool profiling_enabled = false;
    std::vector<std::shared_ptr<GfxRemoteTensor>>* remote_outputs = nullptr;
    const std::vector<std::shared_ptr<GfxRemoteTensor>>* remote_inputs = nullptr;
    const compiler::BackendTarget* expected_target = nullptr;
    std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor;
    GpuBufferPool* pool = nullptr;
    InferPipelinePostPrepareHook post_prepare;
    const std::vector<GpuTensor>* runtime_input_tensors = nullptr;
    InferStageOutputDescInitializer init_output_desc;
    InferInputLookup input_lookup;
    InferSubmissionSession* submission = nullptr;
    InferSubmissionTuningCaps submission_caps;
    InferStageHook on_stage;
    GfxProfiler* profiler = nullptr;
    const char* error_prefix = "GFX";
};

struct InferRuntimeExecutionResult {
    GpuCommandBufferHandle completed_command_buffer = nullptr;
    InferSubmissionTuning submission_tuning{};
    std::vector<InferStage>* pipeline = nullptr;
};

std::vector<InferStage>& prepare_reusable_infer_runtime_pipeline(
    const InferRuntimeExecutionConfig& config);

InferRuntimeExecutionResult prepare_and_execute_infer_runtime(
    InferRuntimeExecutionConfig config);

}  // namespace gfx_plugin
}  // namespace ov
