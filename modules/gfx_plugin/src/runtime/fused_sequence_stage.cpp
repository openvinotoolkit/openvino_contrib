// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/fused_sequence_stage.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

FusedSequenceStage::FusedSequenceStage(std::vector<FusedStageInfo> stages,
                                       std::string name,
                                       std::string type)
    : m_stages(std::move(stages)),
      m_name(std::move(name)),
      m_type(std::move(type)) {}

void FusedSequenceStage::init(GpuBufferManager* buffer_manager) {
    for (auto& info : m_stages) {
        if (info.stage) {
            info.stage->init(buffer_manager);
        }
    }
}

void FusedSequenceStage::compile(GpuBufferManager* buffer_manager) {
    for (auto& info : m_stages) {
        if (info.stage) {
            info.stage->compile(buffer_manager);
        }
    }
}

void FusedSequenceStage::set_inputs(const std::vector<GpuTensor*>& inputs) {
    m_inputs = inputs;
}

void FusedSequenceStage::set_output(GpuTensor* output) {
    m_outputs.clear();
    if (output) {
        m_outputs.push_back(output);
    }
}

void FusedSequenceStage::set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
    m_outputs.clear();
    m_outputs.reserve(outputs.size());
    for (const auto& out : outputs) {
        m_outputs.push_back(out.get());
    }
    for (auto& info : m_stages) {
        if (info.stage && info.output_index < m_outputs.size()) {
            info.stage->set_output(m_outputs[info.output_index]);
        }
    }
}

void FusedSequenceStage::enable_profiling(bool enable) {
    m_profiling_enabled = enable;
    for (auto& info : m_stages) {
        if (info.stage) {
            info.stage->enable_profiling(false);
        }
    }
}

void FusedSequenceStage::set_profiler(void* /*profiler*/,
                                      uint32_t /*node_id*/,
                                      const std::string& /*node_name*/,
                                      const std::string& /*node_type*/) {
    // Intentionally no-op: sub-stages execute within a single pipeline slot.
}

GpuStageSubmitPolicy FusedSequenceStage::submit_policy() const {
    GpuStageSubmitPolicy policy{};
    for (const auto& info : m_stages) {
        if (!info.stage) {
            continue;
        }
        const auto child = info.stage->submit_policy();
        policy.weight += child.weight > 0 ? (child.weight - 1) : 0;
        policy.isolate = policy.isolate || child.isolate;
    }
    return policy;
}

void FusedSequenceStage::execute(GpuCommandBufferHandle command_buffer) {
    if (m_outputs.size() < m_stages.size()) {
        OPENVINO_THROW("GFX: fused stage outputs are not fully bound (",
                       m_outputs.size(),
                       " < ",
                       m_stages.size(),
                       ")");
    }

    for (auto& info : m_stages) {
        if (!info.stage) {
            continue;
        }
        std::vector<GpuTensor*> resolved_inputs;
        resolved_inputs.reserve(info.inputs.size());
        for (const auto& binding : info.inputs) {
            switch (binding.kind) {
                case FusedInputKind::External:
                    resolved_inputs.push_back(binding.index < m_inputs.size()
                                                  ? m_inputs[binding.index]
                                                  : nullptr);
                    break;
                case FusedInputKind::Output:
                    resolved_inputs.push_back(binding.index < m_outputs.size()
                                                  ? m_outputs[binding.index]
                                                  : nullptr);
                    break;
                case FusedInputKind::None:
                default:
                    resolved_inputs.push_back(nullptr);
                    break;
            }
        }
        info.stage->set_inputs(resolved_inputs);
        if (info.output_index < m_outputs.size()) {
            info.stage->set_output(m_outputs[info.output_index]);
        }
        info.stage->execute(command_buffer);
    }
}

std::unique_ptr<GpuStage> FusedSequenceStage::clone() const {
    std::vector<FusedStageInfo> stages;
    stages.reserve(m_stages.size());
    for (const auto& info : m_stages) {
        FusedStageInfo copy;
        if (info.stage) {
            copy.stage = info.stage->clone();
        }
        copy.inputs = info.inputs;
        copy.output_index = info.output_index;
        stages.emplace_back(std::move(copy));
    }
    auto cloned = std::make_unique<FusedSequenceStage>(std::move(stages), m_name, m_type);
    cloned->enable_profiling(m_profiling_enabled);
    return cloned;
}

}  // namespace gfx_plugin
}  // namespace ov
