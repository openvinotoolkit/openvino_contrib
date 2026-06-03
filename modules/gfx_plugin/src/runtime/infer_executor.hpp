// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "openvino/core/model.hpp"
#include "runtime/backend_request_state.hpp"
#include "runtime/infer_pipeline.hpp"
#include "runtime/infer_submission.hpp"

namespace ov {
namespace gfx_plugin {

using InferPipelinePostPrepareHook =
    std::function<void(std::vector<InferStage>&)>;
using InferStageOutputDescInitializer = StageOutputDescInitializer;

struct InferRuntimeExecutionConfig {
    BackendInferState* state = nullptr;
    const std::vector<PipelineStageDesc>* descs = nullptr;
    GpuBufferManager* buffer_manager = nullptr;
    void* stage_profiler = nullptr;
    bool profiling_enabled = false;
    const std::shared_ptr<const ov::Model>* runtime_model = nullptr;
    const std::vector<ov::Output<const ov::Node>>* public_outputs = nullptr;
    const std::unordered_map<const ov::Node*, size_t>* node_map = nullptr;
    const std::unordered_map<const ov::Node*, size_t>* param_map = nullptr;
    std::vector<std::shared_ptr<GfxRemoteTensor>>* remote_outputs = nullptr;
    const std::vector<std::shared_ptr<GfxRemoteTensor>>* remote_inputs = nullptr;
    GpuBackend expected_backend = GpuBackend::Unknown;
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
