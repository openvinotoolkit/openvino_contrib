// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/stateful_execution.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "openvino/core/except.hpp"
#include "runtime/backend_runtime.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/memory_manager.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

const RuntimeStageExecutableDescriptor* stateful_descriptor_or_null(const InferStage& stage) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (!descriptor || descriptor->stateful_effect == "none" ||
        descriptor->stateful_variable_id.empty()) {
        return nullptr;
    }
    return descriptor;
}

const RuntimeTensorBindingContract* output_binding_or_null(const InferStage& stage,
                                                           size_t output_idx) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (!descriptor || output_idx >= descriptor->output_bindings.size()) {
        return nullptr;
    }
    return &descriptor->output_bindings[output_idx];
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

ov::Shape resolve_input_shape(const InferStage& stage,
                              const std::vector<GpuTensor*>& resolved_inputs,
                              size_t input_idx) {
    if (input_idx < resolved_inputs.size() && resolved_inputs[input_idx] &&
        !resolved_inputs[input_idx]->shape.empty()) {
        return resolved_inputs[input_idx]->shape;
    }
    if (stage.node && input_idx < stage.node->get_input_size() &&
        stage.node->get_input_partial_shape(input_idx).is_static()) {
        return stage.node->get_input_shape(input_idx);
    }
    return {};
}

ov::Shape resolve_sum_inputs_along_axis_shape(
    const InferStage& stage,
    const std::vector<GpuTensor*>& resolved_inputs,
    int64_t axis_i64) {
    if (axis_i64 < 0 || resolved_inputs.empty()) {
        return {};
    }
    const auto axis = static_cast<size_t>(axis_i64);
    ov::Shape out_shape;
    const size_t input_count = stage.node
                                   ? std::min(resolved_inputs.size(),
                                              stage.node->get_input_size())
                                   : resolved_inputs.size();
    for (size_t i = 0; i < input_count; ++i) {
        auto in_shape = resolve_input_shape(stage, resolved_inputs, i);
        if (in_shape.empty() || axis >= in_shape.size()) {
            return {};
        }
        if (out_shape.empty()) {
            out_shape = std::move(in_shape);
            continue;
        }
        if (in_shape.size() != out_shape.size()) {
            return {};
        }
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

ov::Shape resolve_assign_output_shape(const InferStage& stage,
                                      const std::vector<GpuTensor*>& resolved_inputs,
                                      const GpuTensor& output,
                                      const RuntimeTensorBindingContract* binding) {
    if (!output.shape.empty()) {
        return output.shape;
    }
    if (binding &&
        binding->stateful_prebind_shape_rule == "sum_inputs_along_axis") {
        return resolve_sum_inputs_along_axis_shape(
            stage, resolved_inputs, binding->stateful_prebind_shape_axis);
    }
    if (stage.node && stage.node->get_output_size() > 0 &&
        stage.node->get_output_partial_shape(0).is_static()) {
        return stage.node->get_output_shape(0);
    }
    return {};
}

std::string stateful_prebind_variable_id(const InferStage& stage,
                                         size_t output_idx,
                                         const RuntimeTensorBindingContract* binding) {
    if (binding) {
        return binding->stateful_prebind_variable_id;
    }
    if (output_idx < stage.direct_stateful_assign_variable_ids.size()) {
        return stage.direct_stateful_assign_variable_ids[output_idx];
    }
    return {};
}

ov::Shape resolve_state_tensor_shape(const StatefulVariableTensorState& slot) {
    if (!slot.tensor.shape.empty()) {
        return slot.tensor.shape;
    }
    if (slot.host_tensor) {
        return slot.host_tensor.get_shape();
    }
    if (slot.expected_shape.is_static()) {
        return slot.expected_shape.to_shape();
    }
    return {};
}

bool has_resolved_state_tensor_shape(const StatefulVariableTensorState& slot) {
    return !slot.tensor.shape.empty() || slot.host_tensor ||
           slot.expected_shape.is_static();
}

ov::element::Type resolve_state_tensor_type(const StatefulVariableTensorState& slot) {
    if (slot.tensor.expected_type != ov::element::dynamic) {
        return slot.tensor.expected_type;
    }
    if (slot.tensor.buf.type != ov::element::dynamic) {
        return slot.tensor.buf.type;
    }
    if (slot.host_tensor) {
        return slot.host_tensor.get_element_type();
    }
    return slot.expected_type;
}

bool upload_host_state_if_needed(StatefulVariableTensorState& slot,
                                 const std::string& variable_id,
                                 GpuBufferPool& pool,
                                 GpuCommandBufferHandle command_buffer,
                                 GfxProfiler* profiler) {
    if (!slot.host_dirty && slot.initialized && slot.tensor.buf.valid()) {
        return true;
    }
    OPENVINO_ASSERT(slot.host_tensor,
                    "GFX: variable state host tensor is not initialized for variable ",
                    variable_id);
    const auto type = slot.host_tensor.get_element_type();
    const auto shape = slot.host_tensor.get_shape();
    const size_t bytes = slot.host_tensor.get_byte_size();
    if (bytes == 0) {
        slot.initialized = false;
        slot.host_dirty = false;
        slot.host_stale = false;
        return false;
    }

    GpuBufferDesc persistent_desc{};
    persistent_desc.bytes = bytes;
    persistent_desc.type = type;
    persistent_desc.usage = BufferUsage::Intermediate;
    persistent_desc.cpu_read = false;
    persistent_desc.cpu_write = false;
    persistent_desc.prefer_device_local = true;
    persistent_desc.label = "stateful_variable";
    auto persistent = pool.ensure(slot.handle, persistent_desc);

    GpuBufferDesc staging_desc{};
    staging_desc.bytes = bytes;
    staging_desc.type = type;
    staging_desc.usage = BufferUsage::Staging;
    staging_desc.cpu_read = false;
    staging_desc.cpu_write = true;
    staging_desc.prefer_device_local = false;
    staging_desc.label = "stateful_variable_upload";
    auto staging = pool.ensure(slot.upload_handle, staging_desc);

    gpu_copy_from_host(staging, slot.host_tensor.data(), bytes);
    gpu_copy_buffer(command_buffer, staging, persistent, bytes);

    slot.tensor.buf = persistent;
    slot.tensor.shape = shape;
    slot.tensor.expected_type = type;
    slot.initialized = true;
    slot.host_dirty = false;
    slot.host_stale = false;
    if (profiler) {
        profiler->increment_counter("stateful_set_state_upload_count");
        profiler->increment_counter("stateful_set_state_upload_bytes", static_cast<uint64_t>(bytes));
    }
    return true;
}

}  // namespace

bool sync_stateful_variable_host(StatefulVariableTensorState& slot,
                                 const std::string& variable_id,
                                 const BackendResources& resources,
                                 GfxProfiler* profiler) {
    if (!slot.host_stale || slot.host_dirty) {
        return true;
    }
    if (!slot.initialized || !slot.tensor.buf.valid()) {
        slot.host_stale = false;
        return true;
    }

    const auto shape = resolve_state_tensor_shape(slot);
    const auto type = resolve_state_tensor_type(slot);
    OPENVINO_ASSERT(type != ov::element::dynamic,
                    "GFX: variable state element type is dynamic for variable ",
                    variable_id);
    OPENVINO_ASSERT(has_resolved_state_tensor_shape(slot),
                    "GFX: variable state shape is dynamic for variable ",
                    variable_id);
    const size_t bytes = tensor_byte_size(shape, type);
    if (!slot.host_tensor ||
        slot.host_tensor.get_element_type() != type ||
        slot.host_tensor.get_shape() != shape) {
        slot.host_tensor = ov::Tensor(type, shape);
    }
    if (bytes == 0) {
        slot.host_stale = false;
        return true;
    }

    OPENVINO_ASSERT(resources.const_manager,
                    "GFX: variable state readback requires backend buffer manager for variable ",
                    variable_id);
    GpuBuffer readback = slot.tensor.buf;
    GpuBuffer staging;
    if (!readback.host_visible) {
        OPENVINO_ASSERT(resources.queue,
                        "GFX: variable state readback requires backend command queue for variable ",
                        variable_id);
        GpuBufferDesc staging_desc{};
        staging_desc.bytes = bytes;
        staging_desc.type = type;
        staging_desc.usage = BufferUsage::Staging;
        staging_desc.cpu_read = true;
        staging_desc.cpu_write = false;
        staging_desc.prefer_device_local = false;
        staging_desc.label = "stateful_variable_readback";
        staging = resources.const_manager->allocate_temp(staging_desc);
        gpu_copy_buffer(resources.queue, slot.tensor.buf, staging, bytes);
        readback = staging;
    }

    gpu_copy_to_host(readback, slot.host_tensor.data(), bytes);
    if (staging.valid()) {
        resources.const_manager->release_temp(std::move(staging));
    }
    slot.host_stale = false;
    if (profiler) {
        profiler->increment_counter("stateful_get_state_readback_count");
        profiler->increment_counter("stateful_get_state_readback_bytes", static_cast<uint64_t>(bytes));
    }
    return true;
}

bool try_bind_direct_stateful_assign_output(StatefulVariableStateMap& variable_states,
                                            InferStage& stage,
                                            const std::vector<GpuTensor*>& resolved_inputs,
                                            GpuBufferPool& pool,
                                            GfxProfiler* profiler) {
    bool any = false;
    for (size_t output_idx = 0; output_idx < stage.outputs.size(); ++output_idx) {
        if (!stage.outputs[output_idx]) {
            continue;
        }
        const auto* binding = output_binding_or_null(stage, output_idx);
        const auto variable_id =
            stateful_prebind_variable_id(stage, output_idx, binding);
        if (variable_id.empty()) {
            continue;
        }

        auto& out = *stage.outputs[output_idx];
        const ov::Shape shape =
            resolve_assign_output_shape(stage, resolved_inputs, out, binding);
        const auto type = resolve_assign_output_type(stage, resolved_inputs, out);
        if (shape.empty() || type == ov::element::dynamic) {
            continue;
        }
        const size_t bytes = tensor_byte_size(shape, type);
        if (bytes == 0) {
            continue;
        }

        GpuBufferDesc desc{};
        desc.bytes = bytes;
        desc.type = type;
        desc.usage = BufferUsage::Intermediate;
        desc.cpu_read = false;
        desc.cpu_write = false;
        desc.prefer_device_local = true;
        desc.label = "stateful_variable";

        auto& slot = variable_states[variable_id];
        auto persistent = pool.ensure(slot.handle, desc);
        if (!persistent.valid()) {
            continue;
        }
        out.buf = persistent;
        out.shape = shape;
        out.expected_type = type;
        if (profiler) {
            profiler->increment_counter("stateful_assign_prebind_count");
            profiler->increment_counter("stateful_assign_prebind_bytes", static_cast<uint64_t>(bytes));
        }
        any = true;
    }
    return any;
}

bool execute_stateful_stage(StatefulVariableStateMap& variable_states,
                            InferStage& stage,
                            const std::vector<GpuTensor*>& resolved_inputs,
                            GpuBufferPool& pool,
                            GpuCommandBufferHandle command_buffer,
                            GfxProfiler* profiler) {
    if (!stage.node || !stage.stage) {
        return false;
    }

    const auto* descriptor = stateful_descriptor_or_null(stage);
    if (!descriptor) {
        return false;
    }
    const auto& variable_id = descriptor->stateful_variable_id;

    if (descriptor->stateful_effect == "read_value") {
        auto state_it = variable_states.find(variable_id);
        OPENVINO_ASSERT(state_it != variable_states.end(),
                        "GFX: ReadValue variable state is not registered for variable ",
                        variable_id);
        auto& slot = state_it->second;
        OPENVINO_ASSERT(upload_host_state_if_needed(slot, variable_id, pool, command_buffer, profiler) &&
                            slot.initialized &&
                            slot.tensor.buf.valid(),
                        "GFX: ReadValue variable state is not materialized for variable ",
                        variable_id);
        OPENVINO_ASSERT(!stage.outputs.empty() && stage.outputs.front() &&
                            stage.outputs.front()->buf.valid(),
                        "GFX: ReadValue output buffer is not allocated by compiler-owned memory plan for variable ",
                        variable_id);

        auto& snapshot = *stage.outputs.front();
        const auto type = resolve_state_tensor_type(slot);
        const auto shape = resolve_state_tensor_shape(slot);
        OPENVINO_ASSERT(type != ov::element::dynamic,
                        "GFX: ReadValue variable state element type is dynamic for variable ",
                        variable_id);
        OPENVINO_ASSERT(has_resolved_state_tensor_shape(slot),
                        "GFX: ReadValue variable state shape is dynamic for variable ",
                        variable_id);
        const size_t bytes = tensor_byte_size(shape, type);
        OPENVINO_ASSERT(bytes <= slot.tensor.buf.size,
                        "GFX: ReadValue variable state buffer is too small for variable ",
                        variable_id);
        OPENVINO_ASSERT(bytes <= snapshot.buf.size,
                        "GFX: ReadValue snapshot output buffer is too small for variable ",
                        variable_id);
        snapshot.shape = shape;
        snapshot.expected_type = type;
        snapshot.i64_values.clear();
        if (bytes > 0 &&
            !same_gpu_allocation(slot.tensor.buf, snapshot.buf)) {
            gpu_copy_buffer(command_buffer, slot.tensor.buf, snapshot.buf, bytes);
            if (profiler) {
                profiler->increment_counter("stateful_read_value_snapshot_copy_count");
                profiler->increment_counter("stateful_read_value_snapshot_copy_bytes",
                                            static_cast<uint64_t>(bytes));
            }
        }
        return true;
    }

    if (descriptor->stateful_effect == "assign") {
        OPENVINO_ASSERT(!resolved_inputs.empty() && resolved_inputs[0] && resolved_inputs[0]->buf.valid(),
                        "GFX: Assign input buffer is not available for variable ",
                        variable_id);
        const auto& src = *resolved_inputs[0];
        const auto src_type = src.expected_type == ov::element::dynamic ? src.buf.type : src.expected_type;
        const auto bytes = src.shape.empty() ? src.buf.size : tensor_byte_size(src.shape, src_type);
        OPENVINO_ASSERT(bytes <= src.buf.size,
                        "GFX: Assign source buffer is too small for variable ",
                        variable_id);

        auto& slot = variable_states[variable_id];
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
        slot.expected_type = src_type;
        if (slot.expected_shape.is_dynamic() && !src.shape.empty()) {
            slot.expected_shape = ov::PartialShape(src.shape);
        }
        slot.host_dirty = false;
        slot.host_stale = true;
        slot.initialized = true;
        return true;
    }

    return false;
}

void execute_infer_stage_with_stateful_contract(
    StatefulVariableStateMap& variable_states,
    InferStage& stage,
    const std::vector<GpuTensor*>& resolved_inputs,
    GpuBufferPool& pool,
    GpuCommandBufferHandle command_buffer,
    GfxProfiler* profiler,
    const StatefulBackendStageExecutor& backend_execute) {
    OPENVINO_ASSERT(stage.stage,
                    "GFX: runtime infer stage is not materialized");
    try_bind_direct_stateful_assign_output(variable_states,
                                           stage,
                                           resolved_inputs,
                                           pool,
                                           profiler);
    if (execute_stateful_stage(variable_states,
                               stage,
                               resolved_inputs,
                               pool,
                               command_buffer,
                               profiler)) {
        return;
    }
    if (backend_execute) {
        backend_execute(stage, resolved_inputs, command_buffer);
        return;
    }
    stage.stage->execute(command_buffer);
}

}  // namespace gfx_plugin
}  // namespace ov
