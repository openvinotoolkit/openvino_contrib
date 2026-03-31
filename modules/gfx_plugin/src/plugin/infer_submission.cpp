// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/infer_submission.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>

#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {

InferSubmissionTuning select_infer_submission_tuning(const InferSubmissionTuningCaps& caps, size_t stage_count) {
    InferSubmissionTuning tuning{};
    tuning.slot_count = 1u;

    const size_t pipeline_stages = std::max<size_t>(stage_count, 1u);
    if (!caps.supports_incremental_submit) {
        tuning.config.max_stages_per_submit = pipeline_stages;
        tuning.config.max_output_bytes_per_submit = std::numeric_limits<size_t>::max() / 4u;
        tuning.config.allow_incremental_submit = false;
        return tuning;
    }

    const uint32_t simd_width =
        std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
    const uint32_t max_threads_per_group = std::max<uint32_t>(caps.max_total_threads_per_group, 1u);
    const bool mobile_class = max_threads_per_group <= 128u;
    const bool constrained_class = max_threads_per_group <= 256u;
    const bool deep_pipeline = pipeline_stages >= 64u;
    const bool very_deep_pipeline = pipeline_stages >= 128u;
    const bool extremely_deep_pipeline = pipeline_stages >= 256u;
    const bool prefer_monolithic_vulkan_submit = caps.backend == GpuBackend::Vulkan && extremely_deep_pipeline;

    if (prefer_monolithic_vulkan_submit) {
        tuning.slot_count = 1u;
        tuning.config.max_stages_per_submit = pipeline_stages;
        tuning.config.max_output_bytes_per_submit = std::numeric_limits<size_t>::max() / 4u;
        tuning.config.allow_incremental_submit = false;
        return tuning;
    }

    // Incremental submit only helps when the next window can be recorded while the
    // previous one is still in flight. A single slot degenerates into immediate
    // submit-wait-submit reuse and turns deep mobile pipelines into fence-thrashing.
    if (caps.supports_incremental_submit) {
        size_t max_slots = 4u;
        size_t stages_per_slot = 96u;
        if (mobile_class) {
            max_slots = 8u;
            stages_per_slot = simd_width >= 64u ? 48u : 32u;
        } else if (constrained_class) {
            max_slots = 6u;
            stages_per_slot = simd_width >= 64u ? 64u : 48u;
        } else {
            max_slots = caps.backend == GpuBackend::Vulkan ? 6u : 4u;
            stages_per_slot = simd_width >= 64u ? (caps.backend == GpuBackend::Vulkan ? 72u : 96u)
                                                : (caps.backend == GpuBackend::Vulkan ? 56u : 64u);
        }

        tuning.slot_count = 1u;
        if (deep_pipeline) {
            tuning.slot_count =
                std::min(max_slots, 1u + ((pipeline_stages - 1u) / std::max<size_t>(stages_per_slot, 1u)));
        } else if (pipeline_stages >= std::max<size_t>(stages_per_slot / 2u, 2u)) {
            tuning.slot_count = std::min<size_t>(2u, max_slots);
        }

        if (very_deep_pipeline) {
            tuning.slot_count = std::max<size_t>(tuning.slot_count, std::min<size_t>(max_slots, 3u));
        }
        if (extremely_deep_pipeline) {
            tuning.slot_count = std::max<size_t>(tuning.slot_count, std::min<size_t>(max_slots, 4u));
        }
    }

    size_t weighted_stage_budget = pipeline_stages;
    size_t output_byte_budget = std::numeric_limits<size_t>::max() / 4u;
    if (mobile_class) {
        weighted_stage_budget = simd_width >= 64u ? 12u : 8u;
        output_byte_budget = 24u * 1024u * 1024u;
        if (deep_pipeline) {
            weighted_stage_budget = simd_width >= 64u ? 10u : 6u;
            output_byte_budget = 16u * 1024u * 1024u;
        }
        if (very_deep_pipeline) {
            weighted_stage_budget = simd_width >= 64u ? 8u : 4u;
            output_byte_budget = 8u * 1024u * 1024u;
        }
    } else if (constrained_class) {
        weighted_stage_budget = simd_width >= 64u ? 20u : 16u;
        output_byte_budget = 48u * 1024u * 1024u;
        if (very_deep_pipeline) {
            weighted_stage_budget = simd_width >= 64u ? 16u : 12u;
            output_byte_budget = 32u * 1024u * 1024u;
        } else if (deep_pipeline) {
            weighted_stage_budget = simd_width >= 64u ? 18u : 14u;
            output_byte_budget = 40u * 1024u * 1024u;
        }
    } else {
        weighted_stage_budget = simd_width >= 64u ? 24u : 20u;
        output_byte_budget = 64u * 1024u * 1024u;
        if (very_deep_pipeline) {
            weighted_stage_budget = simd_width >= 64u ? 20u : 16u;
            output_byte_budget = 48u * 1024u * 1024u;
        } else if (deep_pipeline) {
            weighted_stage_budget = simd_width >= 64u ? 22u : 18u;
            output_byte_budget = 56u * 1024u * 1024u;
        }
    }

    if (caps.supports_incremental_submit && tuning.slot_count > 1u) {
        const size_t slot_parallelism = std::min<size_t>(tuning.slot_count, 4u);
        const size_t extra_stage_budget =
            (slot_parallelism - 1u) * std::max<size_t>(weighted_stage_budget / 3u, 2u);
        const size_t extra_output_budget =
            (slot_parallelism - 1u) * std::max<size_t>(output_byte_budget / 4u, 4u * 1024u * 1024u);
        weighted_stage_budget = std::min(pipeline_stages, weighted_stage_budget + extra_stage_budget);
        output_byte_budget = std::min(std::numeric_limits<size_t>::max() / 4u, output_byte_budget + extra_output_budget);
    }

    tuning.config.max_stages_per_submit = std::max<size_t>(weighted_stage_budget, 1u);
    tuning.config.max_output_bytes_per_submit = std::max<size_t>(output_byte_budget, 1u);
    tuning.config.allow_incremental_submit = true;
    return tuning;
}

void record_infer_submission_tuning_counters(const InferSubmissionTuning& tuning,
                                             const InferSubmissionTuningCaps& caps,
                                             GfxProfiler* profiler) {
    if (!profiler) {
        return;
    }
    profiler->set_counter("submission_slot_count", tuning.slot_count);
    profiler->set_counter("submission_max_stages_per_window", tuning.config.max_stages_per_submit);
    profiler->set_counter("submission_max_output_bytes_per_window", tuning.config.max_output_bytes_per_submit);
    profiler->set_counter("submission_preferred_simd_width", std::max<uint32_t>(caps.preferred_simd_width, 1u));
    profiler->set_counter("submission_subgroup_size", std::max<uint32_t>(caps.subgroup_size, 1u));
    profiler->set_counter("submission_max_total_threads_per_group",
                          std::max<uint32_t>(caps.max_total_threads_per_group, 1u));
    profiler->set_counter("submission_incremental_submit_supported", caps.supports_incremental_submit ? 1u : 0u);
    profiler->set_counter("submission_incremental_submit_enabled", tuning.config.allow_incremental_submit ? 1u : 0u);
}

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
    GfxProfiler* profiler,
    PreparedInferExecutionPlan* prepared_plan,
    const InferStageHook& on_stage,
    const InferSubmissionFinalRecordHook& on_before_final_submit,
    bool recording_started) {
    if (!recording_started) {
        submission.begin_recording();
    }

    size_t recorded_stage_count = 0;
    size_t recorded_output_bytes = 0;
    std::vector<InferStage*> recorded_stages;
    size_t submission_window_index = 0;
    auto describe_recorded_stages = [&]() {
        std::ostringstream oss;
        for (size_t index = 0; index < recorded_stages.size(); ++index) {
            if (index != 0) {
                oss << " | ";
            }
            const auto* infer_stage = recorded_stages[index];
            const auto* gpu_stage = infer_stage ? infer_stage->stage.get() : nullptr;
            oss << (gpu_stage ? gpu_stage->name() : "<null>")
                << " ["
                << (gpu_stage ? gpu_stage->type() : "unknown")
                << "]";
        }
        return oss.str();
    };
    auto flush_submission = [&](bool continue_recording) {
        if (recorded_stage_count == 0) {
            return;
        }
        const bool needs_incremental_submit = continue_recording;
        if (needs_incremental_submit &&
            (!config.allow_incremental_submit || !submission.supports_incremental_submit())) {
            return;
        }
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("InferSubmit") << "Submitting window #" << submission_window_index
                                         << " continue=" << (continue_recording ? "yes" : "no")
                                         << " weighted_stages=" << recorded_stage_count
                                         << " output_bytes=" << recorded_output_bytes
                                         << " stages=" << describe_recorded_stages();
        }
        const bool profiling = (profiler != nullptr);
        const auto submit_start = profiling ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        submission.submit_recorded(continue_recording);
        if (profiling) {
            const auto submit_cpu_us =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - submit_start);
            profiler->increment_counter("submit_count");
            profiler->record_segment("submit",
                                     "window#" + std::to_string(submission_window_index),
                                     submit_cpu_us,
                                     0,
                                     0,
                                     0,
                                     recorded_output_bytes);
            if (continue_recording) {
                profiler->increment_counter("incremental_submit_count");
            }
        }
        notify_pipeline_submission_complete(recorded_stages);
        ++submission_window_index;
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
            const size_t stage_weight = std::max<size_t>(policy.weight, 1);
            if (policy.isolate && recorded_stage_count > 0) {
                flush_submission(true);
            }

            const std::string& stage_type = stage.stage->type();

            const auto command_buffer = submission.current_command_buffer();
            if (gfx_log_debug_enabled()) {
                gfx_log_debug("InferSubmit") << "Record stage " << stage.stage->name()
                                             << " [" << stage_type << "]"
                                             << " weight=" << stage_weight
                                             << " isolate=" << (policy.isolate ? "yes" : "no")
                                             << " current_window=" << submission_window_index;
            }
            if (on_stage) {
                on_stage(stage, resolved, command_buffer);
            } else {
                stage.stage->execute(command_buffer);
            }
            recorded_stage_count += stage_weight;
            recorded_stages.push_back(&stage);
            for (const auto& out : stage.outputs) {
                if (out && out->buf.valid()) {
                    recorded_output_bytes += out->buf.size;
                }
            }

            // "isolate" means "start this stage in a fresh submission window", not
            // "force this stage to be the only stage in that window". Keeping the
            // stage and its immediate epilogue/consumer chain in the same command
            // buffer avoids pathological submit-wait-submit fragmentation on deep
            // mobile Vulkan graphs while still respecting the weighted window budget.
            if (recorded_stage_count >= config.max_stages_per_submit ||
                recorded_output_bytes >= config.max_output_bytes_per_submit) {
                flush_submission(true);
            }
        },
        prepared_plan);

    if (on_before_final_submit) {
        const auto extra_work = on_before_final_submit(submission.current_command_buffer());
        recorded_stage_count += extra_work.weight;
        recorded_output_bytes += extra_work.output_bytes;
    }

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
