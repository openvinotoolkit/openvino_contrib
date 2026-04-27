// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/fused_sequence_stage.hpp"

#include <algorithm>
#include <sstream>

#include "openvino/core/except.hpp"
#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool stage_may_alias_first_input(const GpuStage& stage) {
    const auto& type = stage.type();
    return type == "ReadValue" ||
           type == "Reshape" ||
           type == "Squeeze" ||
           type == "Unsqueeze" ||
           type == "Transpose" ||
           type == "Slice" ||
           type == "StridedSlice";
}

}  // namespace

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

void FusedSequenceStage::set_output_refs(const std::vector<GpuTensor*>& outputs) {
    m_outputs = outputs;
}

void FusedSequenceStage::set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
    std::vector<GpuTensor*> refs;
    refs.reserve(outputs.size());
    for (const auto& output : outputs) {
        refs.push_back(output.get());
    }
    set_output_refs(refs);
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

bool FusedSequenceStage::describe_output_lifetimes(std::vector<GpuStageOutputLifetime>& lifetimes) const {
    size_t output_count = 0;
    for (const auto& info : m_stages) {
        for (const auto output_index : info.output_indices) {
            output_count = std::max(output_count, output_index + 1);
        }
        for (const auto& input : info.inputs) {
            if (input.kind == FusedInputKind::Output) {
                output_count = std::max(output_count, input.index + 1);
            }
        }
    }
    if (output_count == 0) {
        return false;
    }

    lifetimes.assign(output_count, {});
    for (size_t stage_idx = 0; stage_idx < m_stages.size(); ++stage_idx) {
        const auto& info = m_stages[stage_idx];
        for (const auto output_index : info.output_indices) {
            if (output_index >= lifetimes.size()) {
                continue;
            }
            auto& lifetime = lifetimes[output_index];
            lifetime.produced_at = std::min(lifetime.produced_at, stage_idx);
            lifetime.last_used_at = std::max(lifetime.last_used_at == GpuStageOutputLifetime::npos ? stage_idx
                                                                                                   : lifetime.last_used_at,
                                             stage_idx);
        }
        for (const auto& input : info.inputs) {
            if (input.kind != FusedInputKind::Output || input.index >= lifetimes.size()) {
                continue;
            }
            auto& lifetime = lifetimes[input.index];
            if (lifetime.produced_at == GpuStageOutputLifetime::npos) {
                continue;
            }
            lifetime.last_used_at = std::max(lifetime.last_used_at, stage_idx);
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& info : m_stages) {
            if (!info.stage || !stage_may_alias_first_input(*info.stage) || info.inputs.empty()) {
                continue;
            }
            const auto& input = info.inputs.front();
            if (input.kind != FusedInputKind::Output || input.index >= lifetimes.size()) {
                continue;
            }
            auto& input_lifetime = lifetimes[input.index];
            if (!input_lifetime.valid()) {
                continue;
            }
            for (const auto output_index : info.output_indices) {
                if (output_index >= lifetimes.size()) {
                    continue;
                }
                const auto& output_lifetime = lifetimes[output_index];
                if (!output_lifetime.valid()) {
                    continue;
                }
                if (output_lifetime.last_used_at > input_lifetime.last_used_at) {
                    input_lifetime.last_used_at = output_lifetime.last_used_at;
                    changed = true;
                }
            }
        }
    }

    bool any = false;
    for (const auto& lifetime : lifetimes) {
        any = any || lifetime.valid();
    }
    return any;
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
        if (gfx_log_debug_enabled()) {
            std::ostringstream oss;
            oss << "child=" << info.stage->name() << " [" << info.stage->type() << "] inputs=";
            for (size_t input_idx = 0; input_idx < info.inputs.size(); ++input_idx) {
                if (input_idx) {
                    oss << ",";
                }
                const auto& binding = info.inputs[input_idx];
                switch (binding.kind) {
                    case FusedInputKind::External:
                        oss << "ext";
                        break;
                    case FusedInputKind::Output:
                        oss << "out";
                        break;
                    case FusedInputKind::None:
                    default:
                        oss << "none";
                        break;
                }
                oss << binding.index;
            }
            oss << " outputs=";
            for (size_t output_idx = 0; output_idx < info.output_indices.size(); ++output_idx) {
                if (output_idx) {
                    oss << ",";
                }
                oss << info.output_indices[output_idx];
            }
            gfx_log_debug("FusedSequence") << oss.str();
        }
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
        std::vector<GpuTensor*> resolved_outputs;
        resolved_outputs.reserve(info.output_indices.size());
        for (size_t output_index : info.output_indices) {
            resolved_outputs.push_back(output_index < m_outputs.size() ? m_outputs[output_index] : nullptr);
        }
        info.stage->set_output_refs(resolved_outputs);
        info.stage->execute(command_buffer);
    }
}

void FusedSequenceStage::prewarm_runtime_state() {
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
        std::vector<GpuTensor*> resolved_outputs;
        resolved_outputs.reserve(info.output_indices.size());
        for (size_t output_index : info.output_indices) {
            resolved_outputs.push_back(output_index < m_outputs.size() ? m_outputs[output_index] : nullptr);
        }
        info.stage->set_output_refs(resolved_outputs);
        info.stage->prewarm_runtime_state();
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
        copy.output_indices = info.output_indices;
        stages.emplace_back(std::move(copy));
    }
    auto cloned = std::make_unique<FusedSequenceStage>(std::move(stages), m_name, m_type);
    cloned->enable_profiling(m_profiling_enabled);
    return cloned;
}

}  // namespace gfx_plugin
}  // namespace ov
