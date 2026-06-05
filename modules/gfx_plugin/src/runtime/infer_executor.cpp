// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/infer_executor.hpp"

#include <chrono>
#include <utility>

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

void reset_reusable_pipeline_outputs(std::vector<InferStage>& pipeline) {
    for (auto& stage : pipeline) {
        if (!stage.stage) {
            continue;
        }
        for (size_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
            auto& out = stage.outputs[out_idx];
            if (!out) {
                continue;
            }
            out->buf = {};
            out->i64_values.clear();
            if (stage.node && out_idx < stage.node->get_output_size()) {
                if (stage.node->get_output_partial_shape(out_idx).is_static()) {
                    out->shape = stage.node->get_output_shape(out_idx);
                } else {
                    out->shape.clear();
                }
            }
        }
    }
}

void configure_pipeline_profiling(std::vector<InferStage>& pipeline,
                                  void* profiler,
                                  bool profiling_enabled) {
    for (size_t stage_id = 0; stage_id < pipeline.size(); ++stage_id) {
        auto& stage = pipeline[stage_id];
        if (!stage.stage) {
            continue;
        }
        stage.stage->enable_profiling(profiling_enabled);
        if (!profiling_enabled || !profiler) {
            continue;
        }
        const std::string node_name =
            stage.node ? stage.node->get_friendly_name() : stage.stage->name();
        const std::string node_type =
            stage.node ? stage.node->get_type_name() : stage.stage->type();
        stage.stage->set_profiler(profiler,
                                  static_cast<uint32_t>(stage_id),
                                  node_name,
                                  node_type);
    }
}

void release_stage_output_handles(std::vector<BufferHandle>& handles,
                                  GpuBufferPool& pool) {
    for (auto& handle : handles) {
        pool.release(handle);
    }
}

void prepare_stage_output_handles(
    std::vector<std::vector<BufferHandle>>& stage_handles,
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

std::vector<InferStage>& prepare_reusable_pipeline_for_runtime(
    const InferRuntimeExecutionConfig& config) {
    auto& reusable_pipeline = config.state->reusable_pipeline;
    OPENVINO_ASSERT(config.expected_target,
                    config.error_prefix,
                    ": runtime execution requires compiler BackendTarget");
    if (reusable_pipeline.empty()) {
        reusable_pipeline = build_bound_pipeline(*config.descs,
                                                 config.buffer_manager,
                                                 config.stage_profiler,
                                                 config.profiling_enabled,
                                                 *config.runtime_model,
                                                 *config.public_outputs,
                                                 *config.node_map,
                                                 *config.param_map,
                                                 *config.remote_outputs,
                                                 *config.remote_inputs,
                                                 *config.expected_target,
                                                 config.runtime_descriptor,
                                                 config.error_prefix);
    }

    configure_pipeline_profiling(reusable_pipeline,
                                 config.stage_profiler,
                                 config.profiling_enabled);
    reset_reusable_pipeline_outputs(reusable_pipeline);
    normalize_remote_outputs(*config.remote_outputs,
                             *config.expected_target,
                             config.error_prefix);
    bind_remote_outputs(*config.public_outputs,
                        *config.runtime_model,
                        *config.node_map,
                        *config.param_map,
                        *config.remote_outputs,
                        *config.remote_inputs,
                        reusable_pipeline,
                        config.error_prefix);
    if (config.post_prepare) {
        config.post_prepare(reusable_pipeline);
    }
    if (config.runtime_input_tensors) {
        prepare_reusable_execution_plan(config.state->reusable_execution_plan,
                                        reusable_pipeline,
                                        *config.node_map,
                                        *config.param_map);
        assign_runtime_stage_output_shapes(reusable_pipeline,
                                           config.state->reusable_execution_plan,
                                           *config.runtime_input_tensors,
                                           config.error_prefix);
    }
    prepare_stage_output_handles(config.state->stage_output_handles,
                                 reusable_pipeline,
                                 *config.pool,
                                 /*release_view_only=*/true);
    allocate_stage_outputs(reusable_pipeline,
                           config.state->stage_output_handles,
                           *config.pool,
                           config.init_output_desc,
                           &config.state->stage_output_workspace,
                           config.error_prefix);
    return reusable_pipeline;
}

}  // namespace

std::vector<InferStage>& prepare_reusable_infer_runtime_pipeline(
    const InferRuntimeExecutionConfig& config) {
    OPENVINO_ASSERT(config.state, config.error_prefix, ": infer runtime state is null");
    OPENVINO_ASSERT(config.descs, config.error_prefix, ": pipeline descriptors are null");
    OPENVINO_ASSERT(config.runtime_model, config.error_prefix, ": runtime model handle is null");
    OPENVINO_ASSERT(config.public_outputs, config.error_prefix, ": public outputs are null");
    OPENVINO_ASSERT(config.node_map, config.error_prefix, ": node map is null");
    OPENVINO_ASSERT(config.param_map, config.error_prefix, ": parameter map is null");
    OPENVINO_ASSERT(config.remote_outputs, config.error_prefix, ": remote outputs are null");
    OPENVINO_ASSERT(config.remote_inputs, config.error_prefix, ": remote inputs are null");
    OPENVINO_ASSERT(config.pool, config.error_prefix, ": GPU buffer pool is null");
    OPENVINO_ASSERT(config.init_output_desc, config.error_prefix, ": output descriptor initializer is not configured");

    return prepare_reusable_pipeline_for_runtime(config);
}

InferRuntimeExecutionResult prepare_and_execute_infer_runtime(
    InferRuntimeExecutionConfig config) {
    OPENVINO_ASSERT(config.state, config.error_prefix, ": infer runtime state is null");
    OPENVINO_ASSERT(config.descs, config.error_prefix, ": pipeline descriptors are null");
    OPENVINO_ASSERT(config.runtime_model, config.error_prefix, ": runtime model handle is null");
    OPENVINO_ASSERT(config.public_outputs, config.error_prefix, ": public outputs are null");
    OPENVINO_ASSERT(config.node_map, config.error_prefix, ": node map is null");
    OPENVINO_ASSERT(config.param_map, config.error_prefix, ": parameter map is null");
    OPENVINO_ASSERT(config.remote_outputs, config.error_prefix, ": remote outputs are null");
    OPENVINO_ASSERT(config.remote_inputs, config.error_prefix, ": remote inputs are null");
    OPENVINO_ASSERT(config.pool, config.error_prefix, ": GPU buffer pool is null");
    OPENVINO_ASSERT(config.submission, config.error_prefix, ": submission session is null");
    OPENVINO_ASSERT(config.input_lookup, config.error_prefix, ": input lookup is not configured");
    OPENVINO_ASSERT(config.init_output_desc, config.error_prefix, ": output descriptor initializer is not configured");

    auto& state = *config.state;
    const bool profiling = config.profiler != nullptr;
    const auto pipeline_prepare_start =
        profiling ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};

    auto& pipeline = prepare_reusable_infer_runtime_pipeline(config);

    prepare_reusable_execution_plan(state.reusable_execution_plan,
                                    pipeline,
                                    *config.node_map,
                                    *config.param_map);

    if (profiling) {
        config.profiler->record_segment(
            "compile",
            "prepare_reusable_pipeline",
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - pipeline_prepare_start));
    }

    if (!state.reusable_pipeline_runtime_prewarmed) {
        const auto prewarm_start =
            profiling ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point{};
        prewarm_pipeline_runtime_state(pipeline,
                                       *config.node_map,
                                       *config.param_map,
                                       config.input_lookup,
                                       &state.reusable_execution_plan);
        state.reusable_pipeline_runtime_prewarmed = true;
        if (profiling) {
            config.profiler->record_segment(
                "compile",
                "prewarm_reusable_pipeline",
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - prewarm_start));
        }
    }

    config.submission_caps.supports_incremental_submit =
        config.submission->supports_incremental_submit();
    const auto submission_tuning =
        select_infer_submission_tuning(config.submission_caps, pipeline.size());
    record_infer_submission_tuning_counters(submission_tuning,
                                            config.submission_caps,
                                            config.profiler);

    const auto infer_start =
        profiling ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
    execute_pipeline_with_submission(pipeline,
                                     *config.node_map,
                                     *config.param_map,
                                     config.input_lookup,
                                     *config.submission,
                                     submission_tuning.config,
                                     config.profiler,
                                     &state.reusable_execution_plan,
                                     config.on_stage);
    if (profiling) {
        config.profiler->record_segment(
            "infer",
            "execute_pipeline_with_submission",
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - infer_start));
    }

    InferRuntimeExecutionResult result;
    result.completed_command_buffer =
        config.submission->completed_command_buffer();
    result.submission_tuning = submission_tuning;
    result.pipeline = &pipeline;
    return result;
}

}  // namespace gfx_plugin
}  // namespace ov
