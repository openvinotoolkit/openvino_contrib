// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/infer_executor.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "openvino/core/except.hpp"
#include "runtime/memory_manager.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::string stage_descriptor_name(const InferStage& stage) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (descriptor && !descriptor->stage_name.empty()) {
        return descriptor->stage_name;
    }
    return stage.stage ? stage.stage->name() : std::string("<null>");
}

std::string stage_descriptor_op_family(const InferStage& stage) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (descriptor && !descriptor->op_family.empty()) {
        return descriptor->op_family;
    }
    return stage.stage ? stage.stage->type() : std::string("<null>");
}

bool descriptor_output_static_shape(const InferStage& stage,
                                    size_t output_idx,
                                    ov::Shape& shape) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (!descriptor || output_idx >= descriptor->output_bindings.size()) {
        return false;
    }
    return parse_static_shape_contract(
        descriptor->output_bindings[output_idx].partial_shape, shape);
}

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
            ov::Shape descriptor_shape;
            if (descriptor_output_static_shape(stage, out_idx, descriptor_shape)) {
                out->shape = std::move(descriptor_shape);
            } else {
                out->shape.clear();
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
        stage.stage->set_profiler(profiler,
                                  static_cast<uint32_t>(stage_id),
                                  stage_descriptor_name(stage),
                                  stage_descriptor_op_family(stage));
    }
}

void release_stage_output_handles(std::vector<BufferHandle>& handles,
                                  GpuBufferPool& pool) {
    for (auto& handle : handles) {
        pool.release(handle);
    }
}

bool has_bound_remote_output(
    const std::vector<std::shared_ptr<GfxRemoteTensor>>* remote_outputs) {
    if (!remote_outputs) {
        return false;
    }
    return std::any_of(remote_outputs->begin(),
                       remote_outputs->end(),
                       [](const std::shared_ptr<GfxRemoteTensor>& remote) {
                           return remote != nullptr;
                       });
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
    OPENVINO_ASSERT(config.runtime_descriptor,
                    config.error_prefix,
                    ": runtime executable descriptor is null");
    OPENVINO_ASSERT(config.execution_plan,
                    config.error_prefix,
                    ": runtime execution plan is null");
    OPENVINO_ASSERT(config.runtime_descriptor.get() ==
                        config.execution_plan->descriptor_ptr().get(),
                    config.error_prefix,
                    ": runtime descriptor does not match execution plan");
    OPENVINO_ASSERT(!config.runtime_descriptor->public_outputs.empty(),
                    config.error_prefix,
                    ": runtime executable descriptor has no public output descriptors");
    if (reusable_pipeline.empty()) {
        reusable_pipeline = build_bound_pipeline(config.execution_plan->stages(),
                                                 config.buffer_manager,
                                                 config.stage_profiler,
                                                 config.profiling_enabled,
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
    prepare_reusable_execution_plan(config.state->reusable_execution_plan,
                                    reusable_pipeline);
    if (config.runtime_input_tensors) {
        assign_runtime_stage_output_shapes(reusable_pipeline,
                                           config.state->reusable_execution_plan,
                                           *config.runtime_input_tensors,
                                           config.error_prefix);
    }
    prepare_reusable_output_plan(config.state->reusable_output_plan,
                                 *config.runtime_descriptor,
                                 reusable_pipeline,
                                 config.error_prefix);
    if (has_bound_remote_output(config.remote_outputs)) {
        bind_remote_outputs(config.state->reusable_output_plan,
                            *config.remote_outputs,
                            *config.remote_inputs,
                            reusable_pipeline,
                            config.error_prefix);
    }
    if (config.post_prepare) {
        config.post_prepare(reusable_pipeline);
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
    OPENVINO_ASSERT(config.execution_plan, config.error_prefix, ": runtime execution plan is null");
    OPENVINO_ASSERT(config.remote_outputs, config.error_prefix, ": remote outputs are null");
    OPENVINO_ASSERT(config.remote_inputs, config.error_prefix, ": remote inputs are null");
    OPENVINO_ASSERT(config.pool, config.error_prefix, ": GPU buffer pool is null");
    OPENVINO_ASSERT(config.init_output_desc, config.error_prefix, ": output descriptor initializer is not configured");

    return prepare_reusable_pipeline_for_runtime(config);
}

InferRuntimeExecutionResult prepare_and_execute_infer_runtime(
    InferRuntimeExecutionConfig config) {
    OPENVINO_ASSERT(config.state, config.error_prefix, ": infer runtime state is null");
    OPENVINO_ASSERT(config.execution_plan, config.error_prefix, ": runtime execution plan is null");
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
                                    pipeline);

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
