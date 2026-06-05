// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeStageExecutableDescriptor;

class StatefulAssignStage final : public GpuStage {
public:
    explicit StatefulAssignStage(std::string name);

    void init(GpuBufferManager* buffer_manager) override;
    void prepare_runtime_handle(GpuBufferManager* buffer_manager) override;
    void execute(GpuCommandBufferHandle command_buffer) override;

    void set_inputs(const std::vector<GpuTensor*>& inputs) override;
    void set_output(GpuTensor* output) override;

    const std::string& name() const override;
    const std::string& type() const override;
    std::unique_ptr<GpuStage> clone() const override;

private:
    std::string m_name;
    std::string m_type{"Assign"};
    std::vector<GpuTensor*> m_inputs;
};

std::unique_ptr<GpuStage> create_stateful_stage(
    const RuntimeStageExecutableDescriptor& descriptor);

}  // namespace gfx_plugin
}  // namespace ov
