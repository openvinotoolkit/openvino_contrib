// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

enum class FusedInputKind {
    External,
    Output,
    None
};

struct FusedInputBinding {
    FusedInputKind kind = FusedInputKind::None;
    size_t index = 0;
};

struct FusedStageInfo {
    std::unique_ptr<GpuStage> stage;
    std::vector<FusedInputBinding> inputs;
    size_t output_index = 0;
};

// Executes a fixed sequence of stages inside a single pipeline slot.
class FusedSequenceStage final : public GpuStage {
public:
    FusedSequenceStage(std::vector<FusedStageInfo> stages,
                       std::string name,
                       std::string type = "FusedSequence");

    void init(GpuBufferManager* buffer_manager) override;
    void compile(GpuBufferManager* buffer_manager) override;
    void execute(GpuCommandBufferHandle command_buffer) override;

    void set_inputs(const std::vector<GpuTensor*>& inputs) override;
    void set_output(GpuTensor* output) override;
    void set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) override;

    void enable_profiling(bool enable) override;
    void set_profiler(void* profiler,
                      uint32_t node_id,
                      const std::string& node_name,
                      const std::string& node_type) override;

    const std::string& name() const override { return m_name; }
    const std::string& type() const override { return m_type; }
    GpuStageSubmitPolicy submit_policy() const override;

    std::unique_ptr<GpuStage> clone() const override;

private:
    std::vector<FusedStageInfo> m_stages;
    std::vector<GpuTensor*> m_inputs;
    std::vector<GpuTensor*> m_outputs;
    std::string m_name;
    std::string m_type;
    bool m_profiling_enabled = false;
};

}  // namespace gfx_plugin
}  // namespace ov
