// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/mpsrt/mpsrt_request.hpp"

#import <Metal/Metal.h>

#include <chrono>

#include "backends/metal/runtime/metal_command_encoder.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
    return false;
}

const MpsrtPreparedMslDispatch* find_prepared_msl_dispatch(const MpsrtPreparedModel& prepared_model,
                                                           size_t stage_index) {
    for (const auto& dispatch : prepared_model.msl_dispatches) {
        if (dispatch.stage_index == stage_index) {
            return &dispatch;
        }
    }
    return nullptr;
}

bool has_value(const std::vector<GfxMpsrtValue>& values, GfxMpsrtValue value) {
    for (const auto known : values) {
        if (known == value) {
            return true;
        }
    }
    return false;
}

}  // namespace

void MpsrtTensorBindings::clear() {
    m_bindings.clear();
}

void MpsrtTensorBindings::bind(GfxMpsrtValue value, MpsrtBoundBuffer buffer) {
    for (auto& binding : m_bindings) {
        if (binding.value == value) {
            binding.buffer = buffer;
            return;
        }
    }
    m_bindings.push_back({value, buffer});
}

const MpsrtBoundBuffer* MpsrtTensorBindings::lookup(GfxMpsrtValue value) const {
    for (const auto& binding : m_bindings) {
        if (binding.value == value) {
            return &binding.buffer;
        }
    }
    return nullptr;
}

std::vector<MpsrtBoundBuffer> make_mpsrt_bound_buffers(const std::vector<void*>& buffers,
                                                       const std::vector<size_t>& offsets) {
    std::vector<MpsrtBoundBuffer> bound;
    bound.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        bound.push_back({buffers[i], i < offsets.size() ? offsets[i] : 0});
    }
    return bound;
}

bool build_mpsrt_tensor_bindings(const MpsrtModel& model,
                                 const std::vector<MpsrtBoundBuffer>& input_buffers,
                                 const std::vector<MpsrtBoundBuffer>& output_buffers,
                                 const MpsrtTransientAllocator& transient_allocator,
                                 MpsrtTensorBindings& bindings,
                                 MpsrtBindingBuildResult* result,
                                 std::string* error) {
    if (result) {
        *result = {};
    }
    bindings.clear();
    if (input_buffers.size() != model.input_values.size()) {
        return fail(error, "GFX MPSRT: input binding count does not match model input values");
    }
    if (output_buffers.size() != model.output_values.size()) {
        return fail(error, "GFX MPSRT: output binding count does not match model output values");
    }

    for (size_t i = 0; i < model.input_values.size(); ++i) {
        if (!input_buffers[i].buffer) {
            return fail(error, "GFX MPSRT: input binding buffer is null at index " + std::to_string(i));
        }
        bindings.bind(model.input_values[i], input_buffers[i]);
        if (result) {
            ++result->external_inputs_bound;
        }
    }
    for (size_t i = 0; i < model.output_values.size(); ++i) {
        if (!output_buffers[i].buffer) {
            return fail(error, "GFX MPSRT: output binding buffer is null at index " + std::to_string(i));
        }
        bindings.bind(model.output_values[i], output_buffers[i]);
        if (result) {
            ++result->external_outputs_bound;
        }
    }

    for (const auto& tensor : model.tensors) {
        if (bindings.lookup(tensor.value)) {
            continue;
        }

        const bool is_const = (tensor.desc.flags & GfxMpsrtTensorFlagConst) != 0;
        if (is_const) {
            if (result) {
                ++result->const_tensors_skipped;
            }
            continue;
        }

        const bool is_transient = (tensor.desc.flags & GfxMpsrtTensorFlagTransient) != 0 ||
                                  (!has_value(model.input_values, tensor.value) &&
                                   !has_value(model.output_values, tensor.value));
        if (!is_transient) {
            return fail(error, "GFX MPSRT: tensor value " + std::to_string(tensor.value) +
                                   " is neither externally bound nor transient");
        }
        if (!transient_allocator) {
            return fail(error, "GFX MPSRT: transient tensor allocator is not set");
        }
        MpsrtBoundBuffer allocated = transient_allocator(tensor);
        if (!allocated.buffer) {
            return fail(error, "GFX MPSRT: transient allocator returned null for value " +
                                   std::to_string(tensor.value));
        }
        bindings.bind(tensor.value, allocated);
        if (result) {
            ++result->transient_buffers_allocated;
        }
    }
    return true;
}

bool build_mpsrt_external_tensor_bindings(const MpsrtModel& model,
                                          const std::vector<MpsrtBoundBuffer>& external_buffers,
                                          const MpsrtTransientAllocator& transient_allocator,
                                          MpsrtTensorBindings& bindings,
                                          MpsrtBindingBuildResult* result,
                                          std::string* error) {
    if (result) {
        *result = {};
    }
    bindings.clear();

    std::vector<GfxMpsrtValue> external_values = model.external_values;
    if (external_values.empty()) {
        external_values = model.input_values;
        external_values.insert(external_values.end(), model.output_values.begin(), model.output_values.end());
    }
    std::vector<GfxMpsrtValue> external_output_values = model.external_output_values;
    if (external_output_values.empty()) {
        external_output_values = model.output_values;
    }
    if (external_buffers.size() != external_values.size()) {
        return fail(error, "GFX MPSRT: external binding count does not match model external values");
    }

    for (size_t i = 0; i < external_values.size(); ++i) {
        if (!external_buffers[i].buffer) {
            return fail(error, "GFX MPSRT: external binding buffer is null at index " + std::to_string(i));
        }
        bindings.bind(external_values[i], external_buffers[i]);
        if (result) {
            if (has_value(external_output_values, external_values[i])) {
                ++result->external_outputs_bound;
            } else {
                ++result->external_inputs_bound;
            }
        }
    }

    for (const auto& tensor : model.tensors) {
        if (bindings.lookup(tensor.value)) {
            continue;
        }

        const bool is_const = (tensor.desc.flags & GfxMpsrtTensorFlagConst) != 0;
        if (is_const) {
            if (result) {
                ++result->const_tensors_skipped;
            }
            continue;
        }

        const bool is_transient = (tensor.desc.flags & GfxMpsrtTensorFlagTransient) != 0 ||
                                  (!has_value(model.input_values, tensor.value) &&
                                   !has_value(model.output_values, tensor.value) &&
                                   !has_value(external_values, tensor.value));
        if (!is_transient) {
            return fail(error, "GFX MPSRT: tensor value " + std::to_string(tensor.value) +
                                   " is neither externally bound nor transient");
        }
        if (!transient_allocator) {
            return fail(error, "GFX MPSRT: transient tensor allocator is not set");
        }
        MpsrtBoundBuffer allocated = transient_allocator(tensor);
        if (!allocated.buffer) {
            return fail(error, "GFX MPSRT: transient allocator returned null for value " +
                                   std::to_string(tensor.value));
        }
        bindings.bind(tensor.value, allocated);
        if (result) {
            ++result->transient_buffers_allocated;
        }
    }
    return true;
}

MpsrtPreparedMslDispatch make_prepared_msl_dispatch_from_pipeline(const MpsrtRuntimeStage& stage,
                                                                  size_t stage_index,
                                                                  id<MTLComputePipelineState> pipeline) {
    OPENVINO_ASSERT(stage.kind == GfxMpsrtStageKind::MSLDispatch,
                    "GFX MPSRT: cannot prepare non-MSL stage from Metal pipeline");
    OPENVINO_ASSERT(pipeline, "GFX MPSRT: Metal pipeline is null");

    MpsrtPreparedMslDispatch prepared;
    prepared.stage_index = stage_index;
    prepared.stage_record_key = stage.stage_record_key;
    prepared.dispatch_entry_point = stage.dispatch_entry_point;
    prepared.dispatch_kernel_family_id = stage.dispatch_kernel_family_id;
    prepared.dispatch_threads_per_threadgroup = stage.dispatch_threads_per_threadgroup;
    prepared.thread_execution_width = static_cast<uint32_t>([pipeline threadExecutionWidth]);
    prepared.max_total_threads_per_threadgroup = static_cast<uint32_t>([pipeline maxTotalThreadsPerThreadgroup]);
    prepared.pipeline_cache_hit = true;
    prepared.pipeline = pipeline;
    return prepared;
}

bool MpsrtRequest::encode_msl_dispatch(GpuCommandBufferHandle command_buffer,
                                       const MpsrtPreparedMslDispatch& prepared,
                                       const KernelDispatch& dispatch,
                                       const std::vector<MpsrtBoundBuffer>& buffers,
                                       const KernelExecutionHooks* hooks,
                                       MpsrtMslEncodeResult* result) const {
    if (result) {
        *result = {};
    }
    OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
    OPENVINO_ASSERT(prepared.pipeline, "GFX MPSRT: prepared MSL pipeline is null");

    const auto setup_start = hooks && (hooks->on_segment || hooks->on_counter)
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    bool encoder_created = false;
    id<MTLComputeCommandEncoder> enc =
        static_cast<id<MTLComputeCommandEncoder>>(metal_get_or_create_compute_encoder(command_buffer, &encoder_created));
    OPENVINO_ASSERT(enc, "GFX MPSRT: failed to create compute encoder");

    const bool pipeline_bound =
        metal_set_compute_pipeline_if_needed(command_buffer,
                                             reinterpret_cast<GpuCommandEncoderHandle>(enc),
                                             prepared.pipeline);

    std::vector<void*> raw_buffers;
    std::vector<size_t> offsets;
    raw_buffers.reserve(buffers.size());
    offsets.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        OPENVINO_ASSERT(buffer.buffer, "GFX MPSRT: bound MSL buffer is null");
        raw_buffers.push_back(buffer.buffer);
        offsets.push_back(buffer.offset);
    }
    const size_t bound_buffers =
        metal_bind_compute_buffers_if_needed(command_buffer,
                                             reinterpret_cast<GpuCommandEncoderHandle>(enc),
                                             raw_buffers,
                                             offsets);

    if (result) {
        result->encoder_created = encoder_created;
        result->pipeline_bound = pipeline_bound;
        result->bound_buffers = bound_buffers;
    }
    if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_msl_request_encode_count", 1);
        if (encoder_created) {
            hooks->on_counter("mpsrt_msl_encoder_create_count", 1);
        }
        if (pipeline_bound) {
            hooks->on_counter("mpsrt_msl_pipeline_bind_count", 1);
        }
        hooks->on_counter("mpsrt_msl_bound_buffer_count", static_cast<uint64_t>(bound_buffers));
    }
    if (hooks && hooks->on_segment) {
        const auto setup_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - setup_start);
        hooks->on_segment("mpsrt_encode",
                          prepared.dispatch_entry_point,
                          setup_cpu_us,
                          0,
                          static_cast<uint32_t>(bound_buffers),
                          0,
                          0,
                          0,
                          0,
                          -1,
                          0,
                          reinterpret_cast<uint64_t>(command_buffer));
    }

    if (hooks && hooks->on_begin) {
        hooks->on_begin(enc);
    }

    const size_t grid_x = dispatch.grid[0];
    const size_t grid_y = dispatch.grid[1];
    const size_t grid_z = dispatch.grid[2];
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) {
        if (hooks && hooks->on_end) {
            hooks->on_end(enc);
        }
        return true;
    }

    MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);
    MTLSize tg = MTLSizeMake(dispatch.threads_per_group[0],
                             dispatch.threads_per_group[1],
                             dispatch.threads_per_group[2]);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];

    if (hooks && hooks->on_end) {
        hooks->on_end(enc);
    }
    return true;
}

bool MpsrtRequest::build_msl_stage_buffers(const MpsrtRuntimeStage& stage,
                                           const MpsrtTensorBindings& bindings,
                                           std::vector<MpsrtBoundBuffer>& buffers,
                                           std::string* error) const {
    buffers.clear();
    if (stage.kind != GfxMpsrtStageKind::MSLDispatch) {
        return fail(error, "GFX MPSRT: cannot bind buffers for non-MSL stage");
    }
    std::vector<GfxMpsrtValue> buffer_order = stage.kernel_buffer_order;
    if (buffer_order.empty()) {
        if (stage.msl_dispatch_desc.input_count != stage.inputs.size()) {
            return fail(error, "GFX MPSRT: MSL stage input count metadata mismatch");
        }
        if (stage.msl_dispatch_desc.output_count != stage.outputs.size()) {
            return fail(error, "GFX MPSRT: MSL stage output count metadata mismatch");
        }
        buffer_order = stage.inputs;
        buffer_order.insert(buffer_order.end(), stage.outputs.begin(), stage.outputs.end());
    } else if (stage.msl_dispatch_desc.input_count + stage.msl_dispatch_desc.output_count != buffer_order.size()) {
        return fail(error, "GFX MPSRT: MSL stage kernel buffer order metadata mismatch");
    }

    buffers.reserve(buffer_order.size());
    for (const auto value : buffer_order) {
        const auto* bound = bindings.lookup(value);
        if (!bound || !bound->buffer) {
            return fail(error, "GFX MPSRT: missing tensor binding for kernel buffer value " + std::to_string(value));
        }
        buffers.push_back(*bound);
    }
    return true;
}

bool MpsrtRequest::encode_prepared_model(GpuCommandBufferHandle command_buffer,
                                         const MpsrtModel& model,
                                         const MpsrtPreparedModel& prepared_model,
                                         const std::vector<KernelDispatch>& stage_dispatches,
                                         const MpsrtTensorBindings& bindings,
                                         const KernelExecutionHooks* hooks,
                                         MpsrtModelEncodeResult* result,
                                         std::string* error) const {
    if (result) {
        *result = {};
    }
    OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
    if (stage_dispatches.size() < model.stages.size()) {
        return fail(error, "GFX MPSRT: missing dispatch descriptors for prepared model stages");
    }

    if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_model_request_encode_count", 1);
    }

    std::vector<MpsrtBoundBuffer> stage_buffers;
    for (size_t stage_index = 0; stage_index < model.stages.size(); ++stage_index) {
        const auto& stage = model.stages[stage_index];
        if (stage.kind != GfxMpsrtStageKind::MSLDispatch) {
            if (result) {
                ++result->skipped_non_msl_stages;
            }
            if (hooks && hooks->on_counter) {
                hooks->on_counter("mpsrt_model_request_skipped_non_msl_stage_count", 1);
            }
            continue;
        }

        const auto* prepared = find_prepared_msl_dispatch(prepared_model, stage_index);
        if (!prepared) {
            return fail(error, "GFX MPSRT: missing prepared MSL dispatch for stage " + std::to_string(stage_index));
        }
        if (!build_msl_stage_buffers(stage, bindings, stage_buffers, error)) {
            return false;
        }

        MpsrtMslEncodeResult stage_result;
        if (!encode_msl_dispatch(command_buffer,
                                 *prepared,
                                 stage_dispatches[stage_index],
                                 stage_buffers,
                                 hooks,
                                 &stage_result)) {
            return fail(error, "GFX MPSRT: failed to encode MSL stage " + std::to_string(stage_index));
        }
        if (result) {
            ++result->encoded_msl_dispatches;
            result->bound_buffers += stage_result.bound_buffers;
        }
        if (hooks && hooks->on_counter) {
            hooks->on_counter("mpsrt_model_request_msl_stage_encode_count", 1);
        }
    }
    return true;
}

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
