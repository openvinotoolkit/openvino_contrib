// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/stateful_stage.hpp"

#include <utility>

#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class StatefulReadValueStage final : public GpuStage {
public:
    explicit StatefulReadValueStage(std::string name)
        : m_name(std::move(name)) {}

    void init(GpuBufferManager* /*buffer_manager*/) override {}

    void prepare_runtime_handle(GpuBufferManager* /*buffer_manager*/) override {}

    void execute(GpuCommandBufferHandle /*command_buffer*/) override {}

    void set_inputs(const std::vector<GpuTensor*>& inputs) override {
        m_inputs = inputs;
    }

    void set_output(GpuTensor* /*output*/) override {}

    const std::string& name() const override {
        return m_name;
    }

    const std::string& type() const override {
        return m_type;
    }

    std::unique_ptr<GpuStage> clone() const override {
        auto stage = std::make_unique<StatefulReadValueStage>(m_name);
        stage->m_inputs = m_inputs;
        return stage;
    }

private:
    std::string m_name;
    std::string m_type{"ReadValue"};
    std::vector<GpuTensor*> m_inputs;
};

}  // namespace

StatefulAssignStage::StatefulAssignStage(std::string name)
    : m_name(std::move(name)) {}

void StatefulAssignStage::init(GpuBufferManager* /*buffer_manager*/) {}

void StatefulAssignStage::prepare_runtime_handle(GpuBufferManager* /*buffer_manager*/) {}

void StatefulAssignStage::execute(GpuCommandBufferHandle /*command_buffer*/) {}

void StatefulAssignStage::set_inputs(const std::vector<GpuTensor*>& inputs) {
    m_inputs = inputs;
}

void StatefulAssignStage::set_output(GpuTensor* /*output*/) {}

const std::string& StatefulAssignStage::name() const {
    return m_name;
}

const std::string& StatefulAssignStage::type() const {
    return m_type;
}

std::unique_ptr<GpuStage> StatefulAssignStage::clone() const {
    auto stage = std::make_unique<StatefulAssignStage>(m_name);
    stage->m_name = m_name;
    stage->m_type = m_type;
    return stage;
}

std::unique_ptr<GpuStage> create_stateful_stage(
    const std::shared_ptr<const ov::Node>& node,
    const RuntimeStageExecutableDescriptor* descriptor) {
    if (descriptor && descriptor->stateful_effect == "assign") {
        return std::make_unique<StatefulAssignStage>(
            node ? node->get_friendly_name() : std::string{"Assign"});
    }
    if (descriptor && descriptor->stateful_effect == "read_value") {
        return std::make_unique<StatefulReadValueStage>(
            node ? node->get_friendly_name() : std::string{"ReadValue"});
    }
    return {};
}

}  // namespace gfx_plugin
}  // namespace ov
