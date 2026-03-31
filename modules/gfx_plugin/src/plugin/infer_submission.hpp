// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

#include "openvino/core/node.hpp"
#include "plugin/infer_pipeline.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

struct InferSubmissionConfig {
    size_t max_stages_per_submit = 16;
    size_t max_output_bytes_per_submit = 16u * 1024u * 1024u;
    bool allow_incremental_submit = true;
};

struct InferSubmissionTuningCaps {
    GpuBackend backend = GpuBackend::Metal;
    uint32_t preferred_simd_width = 1;
    uint32_t subgroup_size = 1;
    uint32_t max_total_threads_per_group = 1;
    bool supports_incremental_submit = true;
};

struct InferSubmissionTuning {
    size_t slot_count = 1;
    InferSubmissionConfig config{};
};

struct InferSubmissionExtraWork {
    size_t weight = 0;
    size_t output_bytes = 0;
};

class InferSubmissionSession {
public:
    virtual ~InferSubmissionSession() = default;

    virtual void begin_recording() = 0;
    virtual GpuCommandBufferHandle current_command_buffer() const = 0;
    virtual void submit_recorded(bool continue_recording) = 0;
    virtual void finish() = 0;
    virtual bool supports_incremental_submit() const { return true; }
    virtual GpuCommandBufferHandle completed_command_buffer() const { return nullptr; }
};

class TrackedInferSubmissionSession : public InferSubmissionSession {
public:
    void begin_recording() final;
    GpuCommandBufferHandle current_command_buffer() const final;
    void submit_recorded(bool continue_recording) final;
    void finish() final;
    GpuCommandBufferHandle completed_command_buffer() const final;

protected:
    virtual GpuCommandBufferHandle begin_recording_impl() = 0;
    virtual void submit_recorded_impl(GpuCommandBufferHandle command_buffer, bool continue_recording) = 0;
    virtual void finish_impl() = 0;

    void set_completed_command_buffer(GpuCommandBufferHandle command_buffer) {
        m_completed_command_buffer = command_buffer;
    }

private:
    GpuCommandBufferHandle m_current_command_buffer = nullptr;
    GpuCommandBufferHandle m_completed_command_buffer = nullptr;
};

class SingleFlightInferSubmissionSession : public TrackedInferSubmissionSession {
protected:
    GpuCommandBufferHandle begin_recording_impl() final {
        if (!m_slot_prepared) {
            prepare_submission_slot();
            m_slot_prepared = true;
        }
        return begin_recording_on_slot();
    }

    void submit_recorded_impl(GpuCommandBufferHandle command_buffer, bool continue_recording) final {
        submit_recorded_on_slot(command_buffer, continue_recording);
    }

    void finish_impl() final {
        finish_submission_slot();
        m_slot_prepared = false;
    }

    virtual void prepare_submission_slot() {}
    virtual GpuCommandBufferHandle begin_recording_on_slot() = 0;
    virtual void submit_recorded_on_slot(GpuCommandBufferHandle command_buffer, bool continue_recording) = 0;
    virtual void finish_submission_slot() = 0;

private:
    bool m_slot_prepared = false;
};

class RotatingSlotInferSubmissionSession : public TrackedInferSubmissionSession {
public:
    explicit RotatingSlotInferSubmissionSession(size_t slot_count)
        : m_slot_count(std::max<size_t>(slot_count, 1)) {}

protected:
    GpuCommandBufferHandle begin_recording_impl() final {
        prepare_submission_slot(m_current_slot);
        return begin_recording_on_slot(m_current_slot);
    }

    void submit_recorded_impl(GpuCommandBufferHandle command_buffer, bool continue_recording) final {
        submit_recorded_on_slot(m_current_slot, command_buffer, continue_recording);
        if (continue_recording) {
            m_current_slot = (m_current_slot + 1) % m_slot_count;
        }
    }

    void finish_impl() final {
        finish_submission_slots();
        m_current_slot = 0;
    }

    virtual void prepare_submission_slot(size_t /*slot_index*/) {}
    virtual GpuCommandBufferHandle begin_recording_on_slot(size_t slot_index) = 0;
    virtual void submit_recorded_on_slot(size_t slot_index,
                                         GpuCommandBufferHandle command_buffer,
                                         bool continue_recording) = 0;
    virtual void finish_submission_slots() = 0;

private:
    size_t m_slot_count = 1;
    size_t m_current_slot = 0;
};

using InferInputLookup = std::function<GpuTensor*(size_t)>;
using InferStageHook =
    std::function<void(InferStage&, const std::vector<GpuTensor*>&, GpuCommandBufferHandle)>;
using InferSubmissionFinalRecordHook =
    std::function<InferSubmissionExtraWork(GpuCommandBufferHandle)>;

InferSubmissionTuning select_infer_submission_tuning(const InferSubmissionTuningCaps& caps, size_t stage_count);
void record_infer_submission_tuning_counters(const InferSubmissionTuning& tuning,
                                             const InferSubmissionTuningCaps& caps,
                                             GfxProfiler* profiler);

void execute_pipeline_with_submission(
    std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    const InferInputLookup& input_lookup,
    InferSubmissionSession& submission,
    const InferSubmissionConfig& config,
    GfxProfiler* profiler = nullptr,
    PreparedInferExecutionPlan* prepared_plan = nullptr,
    const InferStageHook& on_stage = {},
    const InferSubmissionFinalRecordHook& on_before_final_submit = {},
    bool recording_started = false);

void notify_pipeline_submission_complete(std::vector<InferStage*>& stages);

}  // namespace gfx_plugin
}  // namespace ov
