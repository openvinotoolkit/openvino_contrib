// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/infer_submission.hpp"

#include <algorithm>

namespace ov {
namespace gfx_plugin {

void TrackedInferSubmissionSession::begin_recording() {
    m_completed_command_buffer = nullptr;
    m_current_command_buffer = begin_recording_impl();
}

GpuCommandBufferHandle TrackedInferSubmissionSession::current_command_buffer() const {
    OPENVINO_ASSERT(m_current_command_buffer, "GFX: submission command buffer is not initialized");
    return m_current_command_buffer;
}

void TrackedInferSubmissionSession::submit_recorded(bool continue_recording) {
    OPENVINO_ASSERT(m_current_command_buffer, "GFX: submission command buffer is not initialized");
    submit_recorded_impl(m_current_command_buffer, continue_recording);
    m_completed_command_buffer = m_current_command_buffer;
    m_current_command_buffer = continue_recording ? begin_recording_impl() : nullptr;
}

void TrackedInferSubmissionSession::finish() {
    finish_impl();
}

GpuCommandBufferHandle TrackedInferSubmissionSession::completed_command_buffer() const {
    return m_completed_command_buffer;
}

void execute_pipeline_with_submission(
    std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    const InferInputLookup& input_lookup,
    InferSubmissionSession& submission,
    const InferSubmissionConfig& config,
    PreparedInferExecutionPlan* prepared_plan,
    const InferStageHook& on_stage) {
    submission.begin_recording();

    size_t recorded_stage_count = 0;
    size_t recorded_output_bytes = 0;
    std::vector<InferStage*> recorded_stages;
    auto flush_submission = [&](bool continue_recording) {
        if (recorded_stage_count == 0) {
            return;
        }
        const bool needs_incremental_submit = continue_recording;
        if (needs_incremental_submit && !submission.supports_incremental_submit()) {
            return;
        }
        submission.submit_recorded(continue_recording);
        notify_pipeline_submission_complete(recorded_stages);
        recorded_stage_count = 0;
        recorded_output_bytes = 0;
        recorded_stages.clear();
    };

    execute_pipeline(
        pipeline,
        node_map,
        param_map,
        input_lookup,
        [&](InferStage& stage, const std::vector<GpuTensor*>& resolved) {
            const auto policy = stage.stage->submit_policy();
            if (policy.isolate && recorded_stage_count > 0) {
                flush_submission(true);
            }

            const auto command_buffer = submission.current_command_buffer();
            if (on_stage) {
                on_stage(stage, resolved, command_buffer);
            } else {
                stage.stage->execute(command_buffer);
            }
            recorded_stage_count += std::max<size_t>(policy.weight, 1);
            recorded_stages.push_back(&stage);
            for (const auto& out : stage.outputs) {
                if (out && out->buf.valid()) {
                    recorded_output_bytes += out->buf.size;
                }
            }

            if (policy.isolate ||
                recorded_stage_count >= config.max_stages_per_submit ||
                recorded_output_bytes >= config.max_output_bytes_per_submit) {
                flush_submission(true);
            }
        },
        prepared_plan);

    flush_submission(false);
    submission.finish();
}

void notify_pipeline_submission_complete(std::vector<InferStage*>& stages) {
    for (auto* stage : stages) {
        if (stage && stage->stage) {
            stage->stage->on_command_buffer_complete();
        }
    }
}

}  // namespace gfx_plugin
}  // namespace ov
