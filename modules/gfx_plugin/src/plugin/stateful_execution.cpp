// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/stateful_execution.hpp"

#include <chrono>

#include "openvino/core/except.hpp"
#include "openvino/op/concat.hpp"
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

namespace {

std::string get_stateful_variable_id(const ov::Node* node) {
    if (auto read = dynamic_cast<const ov::op::util::ReadValueBase*>(node)) {
        return read->get_variable_id();
    }
    if (auto assign = dynamic_cast<const ov::op::util::AssignBase*>(node)) {
        return assign->get_variable_id();
    }
    return {};
}

ov::element::Type tensor_storage_type(const GpuTensor& tensor) {
    return tensor.expected_type == ov::element::dynamic ? tensor.buf.type : tensor.expected_type;
}

ov::element::Type resolve_assign_output_type(const InferStage& stage,
                                             const std::vector<GpuTensor*>& resolved_inputs,
                                             GpuTensor& output) {
    if (output.expected_type != ov::element::dynamic) {
        return output.expected_type;
    }
    if (stage.node && stage.node->get_output_size() > 0 &&
        stage.node->get_output_element_type(0) != ov::element::dynamic) {
        return stage.node->get_output_element_type(0);
    }
    for (auto* input : resolved_inputs) {
        if (input && input->buf.valid()) {
            const auto type = tensor_storage_type(*input);
            if (type != ov::element::dynamic) {
                return type;
            }
        }
    }
    return ov::element::dynamic;
}

ov::Shape resolve_assign_output_shape(const InferStage& stage,
                                      const std::vector<GpuTensor*>& resolved_inputs,
                                      const GpuTensor& output) {
    if (!output.shape.empty()) {
        return output.shape;
    }
    if (stage.node && stage.node->get_output_size() > 0 &&
        stage.node->get_output_partial_shape(0).is_static()) {
        return stage.node->get_output_shape(0);
    }

    auto concat = stage.node ? ov::as_type_ptr<const ov::op::v0::Concat>(stage.node) : nullptr;
    if (!concat || resolved_inputs.empty()) {
        return {};
    }

    ov::Shape out_shape;
    const size_t input_count = std::min(resolved_inputs.size(), stage.node->get_input_size());
    for (size_t i = 0; i < input_count; ++i) {
        ov::Shape in_shape;
        if (resolved_inputs[i] && !resolved_inputs[i]->shape.empty()) {
            in_shape = resolved_inputs[i]->shape;
        } else if (stage.node->get_input_partial_shape(i).is_static()) {
            in_shape = stage.node->get_input_shape(i);
        }
        if (in_shape.empty()) {
            return {};
        }
        if (out_shape.empty()) {
            out_shape = in_shape;
            continue;
        }
        if (in_shape.size() != out_shape.size()) {
            return {};
        }
        const size_t axis = normalize_axis(concat->get_axis(), out_shape.size(), "GFX: stateful Concat");
        for (size_t dim = 0; dim < in_shape.size(); ++dim) {
            if (dim == axis) {
                out_shape[dim] += in_shape[dim];
            } else if (out_shape[dim] != in_shape[dim]) {
                return {};
            }
        }
    }
    return out_shape;
}

const ov::op::util::AssignBase* find_direct_assign_consumer(const InferStage& stage, size_t output_idx) {
    if (!stage.node || output_idx >= stage.node->get_output_size()) {
        return nullptr;
    }
    const ov::op::util::AssignBase* assign = nullptr;
    for (const auto& target : stage.node->output(output_idx).get_target_inputs()) {
        if (target.get_index() != 0) {
            continue;
        }
        auto* candidate = dynamic_cast<const ov::op::util::AssignBase*>(target.get_node());
        if (!candidate) {
            continue;
        }
        if (assign && assign != candidate) {
            return nullptr;
        }
        assign = candidate;
    }
    return assign;
}

}  // namespace

bool try_bind_direct_stateful_assign_output(InferRequestState& state,
                                            InferStage& stage,
                                            const std::vector<GpuTensor*>& resolved_inputs,
                                            GpuBufferPool& pool,
                                            GfxProfiler* profiler) {
    if (!stage.node || stage.outputs.size() != 1 || !stage.outputs[0]) {
        return false;
    }
    const auto* assign = find_direct_assign_consumer(stage, 0);
    if (!assign) {
        return false;
    }
    const auto variable_id = get_stateful_variable_id(assign);
    if (variable_id.empty()) {
        return false;
    }

    auto& out = *stage.outputs[0];
    const ov::Shape shape = resolve_assign_output_shape(stage, resolved_inputs, out);
    const auto type = resolve_assign_output_type(stage, resolved_inputs, out);
    if (shape.empty() || type == ov::element::dynamic) {
        return false;
    }
    const size_t bytes = tensor_byte_size(shape, type);
    if (bytes == 0) {
        return false;
    }

    GpuBufferDesc desc{};
    desc.bytes = bytes;
    desc.type = type;
    desc.usage = BufferUsage::Intermediate;
    desc.cpu_read = false;
    desc.cpu_write = false;
    desc.prefer_device_local = true;
    desc.label = "stateful_variable";

    auto& slot = state.variable_states[variable_id];
    auto persistent = pool.ensure(slot.handle, desc);
    if (!persistent.valid()) {
        return false;
    }
    out.buf = persistent;
    out.shape = shape;
    out.expected_type = type;
    if (profiler) {
        profiler->increment_counter("stateful_assign_prebind_count");
        profiler->increment_counter("stateful_assign_prebind_bytes", static_cast<uint64_t>(bytes));
    }
    return true;
}

bool execute_stateful_stage(InferRequestState& state,
                            InferStage& stage,
                            const std::vector<GpuTensor*>& resolved_inputs,
                            GpuBufferPool& pool,
                            GpuCommandBufferHandle command_buffer,
                            GfxProfiler* profiler) {
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
        if (profiler) {
            profiler->increment_counter("stateful_assign_count");
            profiler->increment_counter("stateful_assign_bytes", static_cast<uint64_t>(bytes));
        }
        const bool already_in_slot = persistent.valid() &&
                                     src.buf.buffer == persistent.buffer &&
                                     src.buf.offset == persistent.offset;
        if (!already_in_slot) {
            const auto copy_start = profiler ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
            gpu_copy_buffer(command_buffer, src.buf, persistent, bytes);
            if (profiler) {
                profiler->increment_counter("stateful_assign_copy_count");
                profiler->increment_counter("stateful_assign_copy_bytes", static_cast<uint64_t>(bytes));
                profiler->record_segment("stateful",
                                         "assign_copy",
                                         std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now() - copy_start),
                                         0,
                                         1,
                                         bytes,
                                         bytes);
            }
        } else if (profiler) {
            profiler->increment_counter("stateful_assign_alias_count");
            profiler->increment_counter("stateful_assign_alias_bytes", static_cast<uint64_t>(bytes));
        }

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
