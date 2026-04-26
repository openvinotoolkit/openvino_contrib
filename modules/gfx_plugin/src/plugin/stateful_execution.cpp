// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/stateful_execution.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

bool is_stateful_read_value(const std::shared_ptr<const ov::Node>& node) {
    return static_cast<bool>(ov::as_type_ptr<const ov::op::util::ReadValueBase>(node));
}

bool is_stateful_assign(const std::shared_ptr<const ov::Node>& node) {
    return static_cast<bool>(ov::as_type_ptr<const ov::op::util::AssignBase>(node));
}

std::string get_stateful_variable_id(const std::shared_ptr<const ov::Node>& node) {
    if (auto read = ov::as_type_ptr<const ov::op::util::ReadValueBase>(node)) {
        return read->get_variable_id();
    }
    if (auto assign = ov::as_type_ptr<const ov::op::util::AssignBase>(node)) {
        return assign->get_variable_id();
    }
    return {};
}

bool execute_stateful_stage(InferRequestState& state,
                            InferStage& stage,
                            const std::vector<GpuTensor*>& resolved_inputs,
                            GpuBufferPool& pool,
                            GpuCommandBufferHandle command_buffer) {
    if (!stage.node || !stage.stage) {
        return false;
    }

    const auto variable_id = get_stateful_variable_id(stage.node);
    if (variable_id.empty()) {
        return false;
    }

    if (is_stateful_read_value(stage.node)) {
        auto runtime_inputs = resolved_inputs;
        auto state_it = state.variable_states.find(variable_id);
        if (state_it != state.variable_states.end() &&
            state_it->second.initialized &&
            state_it->second.tensor.buf.valid()) {
            runtime_inputs.assign(1, &state_it->second.tensor);
        }
        stage.stage->set_inputs(runtime_inputs);
        stage.stage->execute(command_buffer);
        return true;
    }

    if (is_stateful_assign(stage.node)) {
        OPENVINO_ASSERT(!resolved_inputs.empty() && resolved_inputs[0] && resolved_inputs[0]->buf.valid(),
                        "GFX: Assign input buffer is not available for variable ",
                        variable_id);
        const auto& src = *resolved_inputs[0];
        const auto src_type = src.expected_type == ov::element::dynamic ? src.buf.type : src.expected_type;
        const auto bytes = src.shape.empty() ? src.buf.size : tensor_byte_size(src.shape, src_type);
        OPENVINO_ASSERT(bytes <= src.buf.size,
                        "GFX: Assign source buffer is too small for variable ",
                        variable_id);

        auto& slot = state.variable_states[variable_id];
        GpuBufferDesc desc{};
        desc.bytes = bytes;
        desc.type = src_type;
        desc.usage = BufferUsage::Intermediate;
        desc.cpu_read = false;
        desc.cpu_write = false;
        desc.prefer_device_local = true;
        desc.label = "stateful_variable";
        auto persistent = pool.ensure(slot.handle, desc);
        gpu_copy_buffer(command_buffer, src.buf, persistent, bytes);

        slot.tensor = src;
        slot.tensor.buf = persistent;
        slot.tensor.buf.backend = persistent.backend;
        slot.tensor.buf.type = src_type;
        slot.tensor.expected_type = src_type;
        slot.initialized = true;
        return true;
    }

    return false;
}

}  // namespace gfx_plugin
}  // namespace ov
