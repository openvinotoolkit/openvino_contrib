// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

#include "openvino/core/node.hpp"
#include "plugin/infer_pipeline.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

struct InferSubmissionConfig {
    size_t max_stages_per_submit = 16;
    size_t max_output_bytes_per_submit = 16u * 1024u * 1024u;
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

using InferInputLookup = std::function<GpuTensor*(size_t)>;
using InferStageHook =
    std::function<void(InferStage&, const std::vector<GpuTensor*>&, GpuCommandBufferHandle)>;

void execute_pipeline_with_submission(
    std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    const InferInputLookup& input_lookup,
    InferSubmissionSession& submission,
    const InferSubmissionConfig& config,
    PreparedInferExecutionPlan* prepared_plan = nullptr,
    const InferStageHook& on_stage = {});

void notify_pipeline_submission_complete(std::vector<InferStage*>& stages);

}  // namespace gfx_plugin
}  // namespace ov
