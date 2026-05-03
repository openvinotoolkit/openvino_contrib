// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/mpsrt/mpsrt_request.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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

const MpsrtPreparedMpsGemm* find_prepared_mps_gemm(const MpsrtPreparedModel& prepared_model,
                                                   size_t stage_index) {
    for (const auto& gemm : prepared_model.mps_gemm_stages) {
        if (gemm.stage_index == stage_index) {
            return &gemm;
        }
    }
    return nullptr;
}

const MpsrtPreparedMpsConv2D* find_prepared_mps_conv2d(const MpsrtPreparedModel& prepared_model,
                                                       size_t stage_index) {
    for (const auto& conv : prepared_model.mps_conv2d_stages) {
        if (conv.stage_index == stage_index) {
            return &conv;
        }
    }
    return nullptr;
}

bool is_mps_conv2d_stage(GfxMpsrtStageKind kind) {
    return kind == GfxMpsrtStageKind::MPSConv2D ||
           kind == GfxMpsrtStageKind::MPSGroupConv2D;
}

bool has_value(const std::vector<GfxMpsrtValue>& values, GfxMpsrtValue value) {
    for (const auto known : values) {
        if (known == value) {
            return true;
        }
    }
    return false;
}

const MpsrtRuntimeTensor* find_tensor(const MpsrtModel& model, GfxMpsrtValue value) {
    for (const auto& tensor : model.tensors) {
        if (tensor.value == value) {
            return &tensor;
        }
    }
    return nullptr;
}

MPSDataType mps_data_type_from_gfx(uint32_t dtype) {
    switch (static_cast<GfxMpsrtDType>(dtype)) {
        case GfxMpsrtDType::F16:
            return MPSDataTypeFloat16;
        case GfxMpsrtDType::F32:
            return MPSDataTypeFloat32;
        default:
            return MPSDataTypeInvalid;
    }
}

uint32_t matrix_count_or_one(const GfxMpsrtTensorAbiDesc& desc) {
    return desc.matrix_count == 0 ? 1 : desc.matrix_count;
}

NSUInteger matrix_bytes_for_desc(const GfxMpsrtTensorAbiDesc& desc) {
    return static_cast<NSUInteger>(desc.matrix_rows) *
           static_cast<NSUInteger>(desc.matrix_row_bytes);
}

size_t matrix_batch_offset(const GfxMpsrtTensorAbiDesc& desc, uint32_t batch_index) {
    if (matrix_count_or_one(desc) == 1) {
        return static_cast<size_t>(desc.byte_offset);
    }
    return static_cast<size_t>(desc.byte_offset) +
           static_cast<size_t>(batch_index) * static_cast<size_t>(matrix_bytes_for_desc(desc));
}

bool make_mps_matrix_descriptor(const GfxMpsrtTensorAbiDesc& desc,
                                MPSMatrixDescriptor*& out,
                                const char* name,
                                std::string* error,
                                uint32_t matrix_count_override = 0) {
    out = nil;
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
        return fail(error, std::string("GFX MPSRT: MPS GEMM ") + name + " tensor is not matrix storage");
    }
    if (desc.matrix_rows == 0 || desc.matrix_columns == 0 || desc.matrix_row_bytes == 0) {
        return fail(error, std::string("GFX MPSRT: MPS GEMM ") + name + " matrix descriptor is incomplete");
    }
    const MPSDataType data_type = mps_data_type_from_gfx(desc.dtype);
    if (data_type == MPSDataTypeInvalid) {
        return fail(error, std::string("GFX MPSRT: MPS GEMM ") + name + " dtype is unsupported");
    }
    const uint32_t matrix_count = matrix_count_override == 0 ? matrix_count_or_one(desc) : matrix_count_override;
    const NSUInteger matrix_bytes = matrix_bytes_for_desc(desc);
    if (matrix_count > 1) {
        out = [MPSMatrixDescriptor matrixDescriptorWithRows:desc.matrix_rows
                                                    columns:desc.matrix_columns
                                                   matrices:matrix_count
                                                   rowBytes:desc.matrix_row_bytes
                                                matrixBytes:matrix_bytes
                                                   dataType:data_type];
    } else {
        out = [MPSMatrixDescriptor matrixDescriptorWithRows:desc.matrix_rows
                                                    columns:desc.matrix_columns
                                                   rowBytes:desc.matrix_row_bytes
                                                   dataType:data_type];
    }
    if (!out) {
        return fail(error, std::string("GFX MPSRT: failed to create MPS GEMM ") + name + " descriptor");
    }
    return true;
}

bool validate_mps_gemm_batch_contract(const GfxMpsrtTensorAbiDesc& lhs,
                                      const GfxMpsrtTensorAbiDesc& rhs,
                                      const GfxMpsrtTensorAbiDesc& output,
                                      std::string* error) {
    const uint32_t lhs_count = matrix_count_or_one(lhs);
    const uint32_t rhs_count = matrix_count_or_one(rhs);
    const uint32_t output_count = matrix_count_or_one(output);
    if (output_count == 0) {
        return fail(error, "GFX MPSRT: MPS GEMM output matrix count is zero");
    }
    if ((lhs_count != output_count && lhs_count != 1) ||
        (rhs_count != output_count && rhs_count != 1)) {
        return fail(error, "GFX MPSRT: MPS GEMM batch matrix counts must be either 1 or output matrix count");
    }
    return true;
}

bool lookup_bound_buffer(const MpsrtTensorBindings& bindings,
                         GfxMpsrtValue value,
                         const char* name,
                         MpsrtBoundBuffer& out,
                         std::string* error) {
    const auto* bound = bindings.lookup(value);
    if (!bound || !bound->buffer) {
        return fail(error, std::string("GFX MPSRT: missing tensor binding for MPS GEMM ") + name);
    }
    out = *bound;
    return true;
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

bool MpsrtRequest::encode_mps_gemm(GpuCommandBufferHandle command_buffer,
                                   const MpsrtModel& model,
                                   const MpsrtRuntimeStage& stage,
                                   const MpsrtPreparedMpsGemm& prepared,
                                   const MpsrtTensorBindings& bindings,
                                   const KernelExecutionHooks* hooks,
                                   MpsrtMpsGemmEncodeResult* result,
                                   std::string* error) const {
    if (result) {
        *result = {};
    }
    OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
    if (stage.kind != GfxMpsrtStageKind::MPSGemm) {
        return fail(error, "GFX MPSRT: cannot encode non-GEMM stage with MPS GEMM");
    }
    if (!prepared.kernel) {
        return fail(error, "GFX MPSRT: prepared MPS GEMM kernel is null");
    }
    if (stage.inputs.size() != 2 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(error, "GFX MPSRT: MPS GEMM requires two inputs and one output");
    }

    const auto* lhs_tensor = find_tensor(model, stage.inputs[0]);
    const auto* rhs_tensor = find_tensor(model, stage.inputs[1]);
    if (!lhs_tensor || !rhs_tensor) {
        return fail(error, "GFX MPSRT: MPS GEMM input tensor descriptor is missing");
    }

    MpsrtBoundBuffer lhs_buffer;
    MpsrtBoundBuffer rhs_buffer;
    MpsrtBoundBuffer output_buffer;
    if (!lookup_bound_buffer(bindings, stage.inputs[0], "lhs", lhs_buffer, error) ||
        !lookup_bound_buffer(bindings, stage.inputs[1], "rhs", rhs_buffer, error) ||
        !lookup_bound_buffer(bindings, stage.outputs[0], "output", output_buffer, error)) {
        return false;
    }

    MPSMatrixDescriptor* lhs_desc = nil;
    MPSMatrixDescriptor* rhs_desc = nil;
    MPSMatrixDescriptor* output_desc = nil;
    if (!make_mps_matrix_descriptor(lhs_tensor->desc, lhs_desc, "lhs", error) ||
        !make_mps_matrix_descriptor(rhs_tensor->desc, rhs_desc, "rhs", error) ||
        !make_mps_matrix_descriptor(stage.output_descs.front(), output_desc, "output", error)) {
        return false;
    }
    if (!validate_mps_gemm_batch_contract(lhs_tensor->desc, rhs_tensor->desc, stage.output_descs.front(), error)) {
        return false;
    }

    metal_end_compute_encoder(command_buffer);
    id<MTLCommandBuffer> command = static_cast<id<MTLCommandBuffer>>(command_buffer);
    const auto encode_start = hooks && hooks->on_segment ? std::chrono::steady_clock::now()
                                                         : std::chrono::steady_clock::time_point{};
    const uint32_t output_count = matrix_count_or_one(stage.output_descs.front());
    const bool needs_batch_loop = output_count > 1 &&
                                  (matrix_count_or_one(lhs_tensor->desc) != output_count ||
                                   matrix_count_or_one(rhs_tensor->desc) != output_count);
    size_t kernel_encodes = 0;
    if (!needs_batch_loop) {
        MPSMatrix* lhs_matrix =
            [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(lhs_buffer.buffer)
                                       offset:static_cast<NSUInteger>(lhs_buffer.offset + lhs_tensor->desc.byte_offset)
                                   descriptor:lhs_desc];
        MPSMatrix* rhs_matrix =
            [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(rhs_buffer.buffer)
                                       offset:static_cast<NSUInteger>(rhs_buffer.offset + rhs_tensor->desc.byte_offset)
                                   descriptor:rhs_desc];
        MPSMatrix* output_matrix =
            [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(output_buffer.buffer)
                                       offset:static_cast<NSUInteger>(output_buffer.offset +
                                                                      stage.output_descs.front().byte_offset)
                                   descriptor:output_desc];
        if (!lhs_matrix || !rhs_matrix || !output_matrix) {
            [lhs_matrix release];
            [rhs_matrix release];
            [output_matrix release];
            return fail(error, "GFX MPSRT: failed to create MPS GEMM matrix wrappers");
        }

        [(MPSMatrixMultiplication*)prepared.kernel encodeToCommandBuffer:command
                                                              leftMatrix:lhs_matrix
                                                             rightMatrix:rhs_matrix
                                                            resultMatrix:output_matrix];
        [lhs_matrix release];
        [rhs_matrix release];
        [output_matrix release];
        kernel_encodes = 1;
    } else {
        MPSMatrixDescriptor* single_lhs_desc = nil;
        MPSMatrixDescriptor* single_rhs_desc = nil;
        MPSMatrixDescriptor* single_output_desc = nil;
        if (!make_mps_matrix_descriptor(lhs_tensor->desc, single_lhs_desc, "lhs", error, 1) ||
            !make_mps_matrix_descriptor(rhs_tensor->desc, single_rhs_desc, "rhs", error, 1) ||
            !make_mps_matrix_descriptor(stage.output_descs.front(), single_output_desc, "output", error, 1)) {
            return false;
        }
        for (uint32_t batch = 0; batch < output_count; ++batch) {
            MPSMatrix* lhs_matrix =
                [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(lhs_buffer.buffer)
                                           offset:static_cast<NSUInteger>(
                                               lhs_buffer.offset + matrix_batch_offset(lhs_tensor->desc, batch))
                                       descriptor:single_lhs_desc];
            MPSMatrix* rhs_matrix =
                [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(rhs_buffer.buffer)
                                           offset:static_cast<NSUInteger>(
                                               rhs_buffer.offset + matrix_batch_offset(rhs_tensor->desc, batch))
                                       descriptor:single_rhs_desc];
            MPSMatrix* output_matrix =
                [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(output_buffer.buffer)
                                           offset:static_cast<NSUInteger>(
                                               output_buffer.offset +
                                               matrix_batch_offset(stage.output_descs.front(), batch))
                                       descriptor:single_output_desc];
            if (!lhs_matrix || !rhs_matrix || !output_matrix) {
                [lhs_matrix release];
                [rhs_matrix release];
                [output_matrix release];
                return fail(error, "GFX MPSRT: failed to create MPS GEMM broadcast matrix wrappers");
            }

            [(MPSMatrixMultiplication*)prepared.kernel encodeToCommandBuffer:command
                                                                  leftMatrix:lhs_matrix
                                                                 rightMatrix:rhs_matrix
                                                                resultMatrix:output_matrix];
            [lhs_matrix release];
            [rhs_matrix release];
            [output_matrix release];
            ++kernel_encodes;
        }
    }

    if (result) {
        result->bound_buffers = 3 * kernel_encodes;
        result->kernel_encodes = kernel_encodes;
    }
    if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_mps_gemm_request_encode_count", 1);
        hooks->on_counter("mpsrt_mps_gemm_kernel_encode_count", kernel_encodes);
        hooks->on_counter("mpsrt_mps_gemm_bound_buffer_count", 3 * kernel_encodes);
    }
    if (hooks && hooks->on_segment) {
        const auto setup_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
        hooks->on_segment("mpsrt_encode",
                          stage.stage_record_key,
                          setup_cpu_us,
                          0,
                          3,
                          0,
                          0,
                          0,
                          0,
                          -1,
                          0,
                          reinterpret_cast<uint64_t>(command_buffer));
    }
    return true;
}

bool MpsrtRequest::encode_mps_conv2d(GpuCommandBufferHandle command_buffer,
                                     const MpsrtModel& model,
                                     const MpsrtRuntimeStage& stage,
                                     const MpsrtPreparedMpsConv2D& prepared,
                                     const MpsrtTensorBindings& bindings,
                                     const KernelExecutionHooks* hooks,
                                     MpsrtMpsConv2DEncodeResult* result,
                                     std::string* error) const {
    if (result) {
        *result = {};
    }
    OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
    (void)model;
    (void)bindings;
    (void)hooks;
    if (!is_mps_conv2d_stage(stage.kind)) {
        return fail(error, "GFX MPSRT: cannot encode non-Conv2D stage with MPS Conv2D");
    }
    if (!prepared.weights_buffer) {
        return fail(error, "GFX MPSRT: prepared MPS Conv2D weights buffer is null");
    }
    return fail(error,
                "GFX MPSRT: MPS Conv2D const-pack is prepared, but MPSImage wrappers and "
                "MPSCNNConvolution encode are not implemented yet");
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
        if (stage.kind == GfxMpsrtStageKind::MPSGemm) {
            const auto* prepared = find_prepared_mps_gemm(prepared_model, stage_index);
            if (!prepared) {
                return fail(error, "GFX MPSRT: missing prepared MPS GEMM for stage " + std::to_string(stage_index));
            }
            MpsrtMpsGemmEncodeResult stage_result;
            if (!encode_mps_gemm(command_buffer,
                                 model,
                                 stage,
                                 *prepared,
                                 bindings,
                                 hooks,
                                 &stage_result,
                                 error)) {
                return false;
            }
            if (result) {
                ++result->encoded_mps_gemm_stages;
                result->bound_buffers += stage_result.bound_buffers;
            }
            if (hooks && hooks->on_counter) {
                hooks->on_counter("mpsrt_model_request_mps_gemm_stage_encode_count", 1);
            }
            continue;
        }

        if (is_mps_conv2d_stage(stage.kind)) {
            const auto* prepared = find_prepared_mps_conv2d(prepared_model, stage_index);
            if (!prepared) {
                return fail(error, "GFX MPSRT: missing prepared MPS Conv2D for stage " + std::to_string(stage_index));
            }
            MpsrtMpsConv2DEncodeResult stage_result;
            if (!encode_mps_conv2d(command_buffer,
                                   model,
                                   stage,
                                   *prepared,
                                   bindings,
                                   hooks,
                                   &stage_result,
                                   error)) {
                return false;
            }
            if (result) {
                ++result->encoded_mps_conv2d_stages;
                result->bound_buffers += stage_result.bound_resources;
            }
            if (hooks && hooks->on_counter) {
                hooks->on_counter("mpsrt_model_request_mps_conv2d_stage_encode_count", 1);
            }
            continue;
        }

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
