// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/node.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

class StatefulAssignStage final : public GpuStage {
public:
    explicit StatefulAssignStage(const std::shared_ptr<const ov::Node>& node);

    void init(GpuBufferManager* buffer_manager) override;
    void compile(GpuBufferManager* buffer_manager) override;
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

std::unique_ptr<GpuStage> create_stateful_stage(const std::shared_ptr<const ov::Node>& node);

}  // namespace gfx_plugin
}  // namespace ov
