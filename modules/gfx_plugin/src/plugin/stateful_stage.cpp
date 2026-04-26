// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/stateful_stage.hpp"

#include "openvino/op/util/assign_base.hpp"

namespace ov {
namespace gfx_plugin {

StatefulAssignStage::StatefulAssignStage(const std::shared_ptr<const ov::Node>& node) {
    m_name = node ? node->get_friendly_name() : std::string{"Assign"};
}

void StatefulAssignStage::init(GpuBufferManager* /*buffer_manager*/) {}

void StatefulAssignStage::compile(GpuBufferManager* /*buffer_manager*/) {}

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
    auto stage = std::make_unique<StatefulAssignStage>(nullptr);
    stage->m_name = m_name;
    stage->m_type = m_type;
    return stage;
}

std::unique_ptr<GpuStage> create_stateful_stage(const std::shared_ptr<const ov::Node>& node) {
    if (ov::as_type_ptr<const ov::op::util::AssignBase>(node)) {
        return std::make_unique<StatefulAssignStage>(node);
    }
    return {};
}

}  // namespace gfx_plugin
}  // namespace ov
