// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/mpsrt/mpsrt_request.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <chrono>

#include "backends/metal/runtime/metal_command_encoder.hpp"
#include "openvino/core/except.hpp"
#include "runtime/gfx_mpsrt_kernel_manifest_adapter.hpp"

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

const MpsrtPreparedMpsPool2D* find_prepared_mps_pool2d(const MpsrtPreparedModel& prepared_model,
                                                       size_t stage_index) {
    for (const auto& pool : prepared_model.mps_pool2d_stages) {
        if (pool.stage_index == stage_index) {
            return &pool;
        }
    }
    return nullptr;
}

const MpsrtPreparedMpsResize2D* find_prepared_mps_resize2d(const MpsrtPreparedModel& prepared_model,
                                                           size_t stage_index) {
    for (const auto& resize : prepared_model.mps_resize2d_stages) {
        if (resize.stage_index == stage_index) {
            return &resize;
        }
    }
    return nullptr;
}

const MpsrtPreparedMpsSoftmax* find_prepared_mps_softmax(const MpsrtPreparedModel& prepared_model,
                                                         size_t stage_index) {
    for (const auto& softmax : prepared_model.mps_softmax_stages) {
        if (softmax.stage_index == stage_index) {
            return &softmax;
        }
    }
    return nullptr;
}

const MpsrtPreparedMpsTopK* find_prepared_mps_topk(const MpsrtPreparedModel& prepared_model,
                                                   size_t stage_index) {
    for (const auto& topk : prepared_model.mps_topk_stages) {
        if (topk.stage_index == stage_index) {
            return &topk;
        }
    }
    return nullptr;
}

bool is_mps_conv2d_stage(GfxMpsrtStageKind kind) {
    return kind == GfxMpsrtStageKind::MPSConv2D ||
           kind == GfxMpsrtStageKind::MPSGroupConv2D;
}

const MpsrtRuntimeTensor* find_tensor(const MpsrtModel& model, GfxMpsrtValue value) {
    for (const auto& tensor : model.tensors) {
        if (tensor.value == value) {
            return &tensor;
        }
    }
    return nullptr;
}

bool tensor_requires_image_binding(const GfxMpsrtTensorAbiDesc& desc) {
    return desc.storage == static_cast<uint32_t>(GfxMpsrtStorage::Image);
}

bool validate_bound_resource(const GfxMpsrtTensorAbiDesc& desc,
                             const MpsrtBoundBuffer& bound,
                             const std::string& name,
                             std::string* error) {
    if (tensor_requires_image_binding(desc)) {
        if (!bound.texture) {
            return fail(error, "GFX MPSRT: " + name + " image texture binding is null");
        }
        if (bound.offset != 0) {
            return fail(error, "GFX MPSRT: " + name + " image binding must have zero byte offset");
        }
        return true;
    }
    if (!bound.buffer) {
        return fail(error, "GFX MPSRT: " + name + " buffer binding is null");
    }
    return true;
}

void count_transient_resource(const GfxMpsrtTensorAbiDesc& desc, MpsrtBindingBuildResult* result) {
    if (!result) {
        return;
    }
    if (tensor_requires_image_binding(desc)) {
        ++result->transient_images_allocated;
    } else {
        ++result->transient_buffers_allocated;
    }
}

void count_external_tensor_resource(GfxMpsrtExternalBufferRole role, MpsrtBindingBuildResult* result) {
    if (!result) {
        return;
    }
    if (gfx_mpsrt_is_external_output_buffer_role(role)) {
        ++result->external_outputs_bound;
    } else {
        ++result->external_inputs_bound;
    }
}

const MpsrtRuntimeResource* find_resource_for_value_and_lifetime(const MpsrtModel& model,
                                                                 GfxMpsrtValue value,
                                                                 MpsrtRuntimeResourceLifetime lifetime) {
    for (const auto& resource : model.resources) {
        if (resource.has_tensor_value &&
            resource.value == value &&
            resource.lifetime == lifetime) {
            return &resource;
        }
    }
    return nullptr;
}

const MpsrtPreparedResource* find_prepared_resource(const MpsrtPreparedModel* prepared_model,
                                                   uint32_t resource_index) {
    if (!prepared_model) {
        return nullptr;
    }
    for (const auto& resource : prepared_model->resources) {
        if (resource.resource_index == resource_index) {
            return &resource;
        }
    }
    return nullptr;
}

bool validate_bound_resource(const MpsrtRuntimeResource& resource,
                             const MpsrtBoundBuffer& bound,
                             const std::string& name,
                             std::string* error) {
    if (!resource.has_tensor_value) {
        return fail(error, "GFX MPSRT: " + name + " resource is not a tensor");
    }
    return validate_bound_resource(resource.tensor_desc, bound, name, error);
}

bool bind_external_tensor_resource(const MpsrtRuntimeResource& resource,
                                   const MpsrtBoundBuffer& bound,
                                   const std::string& name,
                                   MpsrtTensorBindings& bindings,
                                   MpsrtBindingBuildResult* result,
                                   std::string* error) {
    if (resource.lifetime != MpsrtRuntimeResourceLifetime::External) {
        return fail(error, "GFX MPSRT: " + name + " resource is not external");
    }
    if (!validate_bound_resource(resource, bound, name, error)) {
        return false;
    }
    bindings.bind(resource.value, bound);
    count_external_tensor_resource(resource.role, result);
    return true;
}

bool bind_external_value_resource(const MpsrtModel& model,
                                  GfxMpsrtValue value,
                                  const MpsrtBoundBuffer& bound,
                                  const std::string& name,
                                  MpsrtTensorBindings& bindings,
                                  MpsrtBindingBuildResult* result,
                                  std::string* error) {
    const auto* resource = find_resource_for_value_and_lifetime(model,
                                                                value,
                                                                MpsrtRuntimeResourceLifetime::External);
    if (!resource) {
        return fail(error, "GFX MPSRT: " + name + " has no external runtime resource");
    }
    return bind_external_tensor_resource(*resource, bound, name, bindings, result, error);
}

bool bind_model_owned_resource(const MpsrtRuntimeResource& resource,
                               const MpsrtPreparedModel* prepared_model,
                               MpsrtTensorBindings& bindings,
                               MpsrtBindingBuildResult* result,
                               std::string* error) {
    if (!resource.has_tensor_value) {
        return fail(error, "GFX MPSRT: model-owned non-tensor resources are not supported");
    }
    if (resource.role != GfxMpsrtExternalBufferRole::ConstBuffer) {
        return fail(error, "GFX MPSRT: model-owned resource must be a const buffer");
    }
    const auto* prepared = find_prepared_resource(prepared_model, resource.resource_index);
    if (!prepared) {
        return fail(error,
                    "GFX MPSRT: model-owned resource " +
                        std::to_string(resource.resource_index) + " is missing from prepared resources");
    }
    if (prepared->lifetime != MpsrtRuntimeResourceLifetime::Model ||
        !prepared->has_tensor_value ||
        prepared->value != resource.value ||
        !prepared->buffer) {
        return fail(error, "GFX MPSRT: prepared model resource is not bindable");
    }
    MpsrtBoundBuffer bound{(__bridge void*)prepared->buffer, prepared->offset};
    if (!validate_bound_resource(resource,
                                 bound,
                                 "model-owned binding for resource " +
                                     std::to_string(resource.resource_index),
                                 error)) {
        return false;
    }
    bindings.bind(resource.value, bound);
    if (result) {
        ++result->model_resources_bound;
    }
    return true;
}

bool bind_prepared_transient_resource(const MpsrtRuntimeResource& resource,
                                      const MpsrtPreparedModel* prepared_model,
                                      MpsrtTensorBindings& bindings,
                                      MpsrtBindingBuildResult* result,
                                      std::string* error) {
    if (!resource.has_tensor_value) {
        return fail(error, "GFX MPSRT: transient non-tensor resources are not supported");
    }
    const auto* prepared = find_prepared_resource(prepared_model, resource.resource_index);
    if (!prepared) {
        return fail(error,
                    "GFX MPSRT: transient resource " +
                        std::to_string(resource.resource_index) + " is missing from prepared resources");
    }
    if (prepared->lifetime != MpsrtRuntimeResourceLifetime::Transient ||
        !prepared->has_tensor_value ||
        prepared->value != resource.value) {
        return fail(error, "GFX MPSRT: prepared transient resource is not bindable");
    }
    MpsrtBoundBuffer allocated{};
    if (tensor_requires_image_binding(resource.tensor_desc)) {
        allocated = make_mpsrt_bound_image((__bridge void*)prepared->texture);
    } else {
        allocated = {(__bridge void*)prepared->buffer, prepared->offset};
    }
    if (!validate_bound_resource(resource,
                                 allocated,
                                 "transient binding for resource " + std::to_string(resource.resource_index),
                                 error)) {
        return false;
    }
    bindings.bind(resource.value, allocated);
    count_transient_resource(resource.tensor_desc, result);
    return true;
}

bool bind_unbound_owned_resources(const MpsrtModel& model,
                                  const MpsrtPreparedModel* prepared_model,
                                  MpsrtTensorBindings& bindings,
                                  MpsrtBindingBuildResult* result,
                                  std::string* error) {
    for (const auto& resource : model.resources) {
        switch (resource.lifetime) {
            case MpsrtRuntimeResourceLifetime::External:
                if (resource.has_tensor_value && !bindings.lookup(resource.value)) {
                    return fail(error,
                                "GFX MPSRT: external tensor resource " +
                                    std::to_string(resource.resource_index) + " is not bound");
                }
                break;
            case MpsrtRuntimeResourceLifetime::Model:
                if (resource.has_tensor_value && bindings.lookup(resource.value)) {
                    break;
                }
                if (!bind_model_owned_resource(resource, prepared_model, bindings, result, error)) {
                    return false;
                }
                break;
            case MpsrtRuntimeResourceLifetime::Transient:
                if (resource.has_tensor_value && bindings.lookup(resource.value)) {
                    break;
                }
                if (!bind_prepared_transient_resource(resource, prepared_model, bindings, result, error)) {
                    return false;
                }
                break;
            case MpsrtRuntimeResourceLifetime::Unknown:
                return fail(error, "GFX MPSRT: runtime resource has unknown lifetime");
        }
    }
    return true;
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

bool make_mps_topk_index_matrix_descriptor(const GfxMpsrtTensorAbiDesc& desc,
                                           MPSMatrixDescriptor*& out,
                                           const char* name,
                                           std::string* error) {
    out = nil;
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
        return fail(error, std::string("GFX MPSRT: MPS TopK ") + name + " tensor is not matrix storage");
    }
    if (desc.matrix_rows == 0 || desc.matrix_columns == 0 || desc.matrix_row_bytes == 0) {
        return fail(error, std::string("GFX MPSRT: MPS TopK ") + name + " matrix descriptor is incomplete");
    }
    if (desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::I32) &&
        desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::U32)) {
        return fail(error, std::string("GFX MPSRT: MPS TopK ") + name + " dtype must be i32/u32");
    }
    const uint32_t matrix_count = matrix_count_or_one(desc);
    const NSUInteger matrix_bytes = matrix_bytes_for_desc(desc);
    if (matrix_count > 1) {
        out = [MPSMatrixDescriptor matrixDescriptorWithRows:desc.matrix_rows
                                                    columns:desc.matrix_columns
                                                   matrices:matrix_count
                                                   rowBytes:desc.matrix_row_bytes
                                                matrixBytes:matrix_bytes
                                                   dataType:MPSDataTypeUInt32];
    } else {
        out = [MPSMatrixDescriptor matrixDescriptorWithRows:desc.matrix_rows
                                                    columns:desc.matrix_columns
                                                   rowBytes:desc.matrix_row_bytes
                                                   dataType:MPSDataTypeUInt32];
    }
    if (!out) {
        return fail(error, std::string("GFX MPSRT: failed to create MPS TopK ") + name + " descriptor");
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

bool lookup_bound_image(const MpsrtTensorBindings& bindings,
                        GfxMpsrtValue value,
                        const char* name,
                        MpsrtBoundBuffer& out,
                        std::string* error) {
    const auto* bound = bindings.lookup(value);
    if (!bound || !bound->texture) {
        return fail(error, std::string("GFX MPSRT: missing tensor binding for MPS Conv2D ") + name + " image");
    }
    out = *bound;
    return true;
}

MTLPixelFormat mps_image_pixel_format_from_gfx(uint32_t dtype) {
    switch (static_cast<GfxMpsrtDType>(dtype)) {
        case GfxMpsrtDType::F16:
            return MTLPixelFormatRGBA16Float;
        case GfxMpsrtDType::F32:
            return MTLPixelFormatRGBA32Float;
        default:
            return MTLPixelFormatInvalid;
    }
}

uint32_t image_slice_count(uint32_t feature_channels) {
    return (feature_channels + 3) / 4;
}

uint32_t conv_kernel_height(const GfxMpsrtTensorAbiDesc& weights) {
    return weights.rank == 5 ? weights.dims[3] : weights.dims[2];
}

uint32_t conv_kernel_width(const GfxMpsrtTensorAbiDesc& weights) {
    return weights.rank == 5 ? weights.dims[4] : weights.dims[3];
}

NSInteger mps_conv_offset(uint32_t kernel, uint32_t dilation, uint32_t pad_before) {
    const uint32_t kernel_extent = kernel == 0 ? 1 : kernel;
    const uint32_t dilation_extent = dilation == 0 ? 1 : dilation;
    const uint32_t effective_kernel = kernel_extent + (kernel_extent - 1) * (dilation_extent - 1);
    return static_cast<NSInteger>(effective_kernel / 2) - static_cast<NSInteger>(pad_before);
}

uint32_t align_channels_for_mps_image(uint32_t channels) {
    return ((channels + 3u) / 4u) * 4u;
}

bool make_mps_image_wrapper(const GfxMpsrtTensorAbiDesc& desc,
                            const MpsrtBoundBuffer& bound,
                            const char* name,
                            MPSImage*& out,
                            std::string* error) {
    out = nil;
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Image)) {
        return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name + " tensor is not image storage");
    }
    if (!bound.texture) {
        return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name + " texture binding is null");
    }
    if (bound.offset != 0) {
        return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name + " image binding must have zero byte offset");
    }
    if (desc.image_width == 0 || desc.image_height == 0 || desc.image_feature_channels == 0 ||
        desc.image_batch == 0) {
        return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name + " image descriptor is incomplete");
    }

    id<MTLTexture> texture = static_cast<id<MTLTexture>>(bound.texture);
    if ([texture width] != desc.image_width || [texture height] != desc.image_height) {
        return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name + " texture shape mismatch");
    }
    const MTLPixelFormat expected_pixel_format = mps_image_pixel_format_from_gfx(desc.dtype);
    if (expected_pixel_format == MTLPixelFormatInvalid || [texture pixelFormat] != expected_pixel_format) {
        return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name + " texture pixel format mismatch");
    }

    const uint32_t slices = image_slice_count(desc.image_feature_channels);
    const NSUInteger expected_array_length = static_cast<NSUInteger>(desc.image_batch * slices);
    if ([texture textureType] == MTLTextureType2D) {
        if (expected_array_length != 1) {
            return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name + " 2D texture cannot hold image array");
        }
    } else if ([texture textureType] != MTLTextureType2DArray ||
               [texture arrayLength] != expected_array_length) {
        return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name + " texture array layout mismatch");
    }

    out = [[MPSImage alloc] initWithTexture:texture
                            featureChannels:align_channels_for_mps_image(desc.image_feature_channels)];
    if (!out) {
        return fail(error, std::string("GFX MPSRT: failed to create MPS Conv2D ") + name + " image wrapper");
    }
    if ([out numberOfImages] != desc.image_batch) {
        [out release];
        out = nil;
        return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name + " image batch mismatch");
    }
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

MpsrtBoundBuffer make_mpsrt_bound_image(void* texture) {
    return {nullptr, 0, texture};
}

bool build_mpsrt_tensor_bindings(const MpsrtModel& model,
                                 const std::vector<MpsrtBoundBuffer>& input_buffers,
                                 const std::vector<MpsrtBoundBuffer>& output_buffers,
                                 MpsrtTensorBindings& bindings,
                                 MpsrtBindingBuildResult* result,
                                 std::string* error,
                                 const MpsrtPreparedModel* prepared_model) {
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

    if (model.resources.empty()) {
        return fail(error, "GFX MPSRT: runtime resource table is required for tensor bindings");
    }
    for (size_t i = 0; i < model.input_values.size(); ++i) {
        if (!bind_external_value_resource(model,
                                          model.input_values[i],
                                          input_buffers[i],
                                          "input binding at index " + std::to_string(i),
                                          bindings,
                                          result,
                                          error)) {
            return false;
        }
    }
    for (size_t i = 0; i < model.output_values.size(); ++i) {
        if (!bind_external_value_resource(model,
                                          model.output_values[i],
                                          output_buffers[i],
                                          "output binding at index " + std::to_string(i),
                                          bindings,
                                          result,
                                          error)) {
            return false;
        }
    }
    return bind_unbound_owned_resources(model,
                                        prepared_model,
                                        bindings,
                                        result,
                                        error);
}

bool build_mpsrt_external_tensor_bindings(const MpsrtModel& model,
                                          const std::vector<MpsrtBoundBuffer>& external_buffers,
                                          MpsrtTensorBindings& bindings,
                                          MpsrtBindingBuildResult* result,
                                          std::string* error,
                                          const MpsrtPreparedModel* prepared_model) {
    if (result) {
        *result = {};
    }
    bindings.clear();

    std::vector<GfxMpsrtValue> external_values = model.external_values;
    if (external_values.empty()) {
        external_values = model.input_values;
        external_values.insert(external_values.end(), model.output_values.begin(), model.output_values.end());
    }
    if (model.resources.empty()) {
        return fail(error, "GFX MPSRT: runtime resource table is required for external tensor bindings");
    }
    if (external_buffers.size() != mpsrt_model_external_buffer_abi_count(model)) {
        return fail(error, "GFX MPSRT: external binding count does not match model external buffer ABI");
    }
    if (!model.external_buffer_bindings.empty()) {
        for (const auto& external : model.external_buffer_bindings) {
            const auto* resource = find_mpsrt_external_resource(model, external);
            if (!resource) {
                return fail(error, "GFX MPSRT: external binding references an invalid resource");
            }
            if (!resource->has_tensor_value) {
                if (result) {
                    ++result->external_resources_bound;
                }
                continue;
            }
            if (external.arg_index >= external_buffers.size()) {
                return fail(error, "GFX MPSRT: external buffer ABI index is out of range");
            }
            if (!bind_external_tensor_resource(*resource,
                                               external_buffers[external.arg_index],
                                               "external binding at ABI index " +
                                                   std::to_string(external.arg_index),
                                               bindings,
                                               result,
                                               error)) {
                return false;
            }
        }
    } else {
        for (size_t i = 0; i < external_values.size(); ++i) {
            if (!bind_external_value_resource(model,
                                              external_values[i],
                                              external_buffers[i],
                                              "external binding at index " + std::to_string(i),
                                              bindings,
                                              result,
                                              error)) {
                return false;
            }
        }
    }
    return bind_unbound_owned_resources(model,
                                        prepared_model,
                                        bindings,
                                        result,
                                        error);
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
    const auto& buffer_order = stage.kernel_buffer_order;
    if (buffer_order.empty()) {
        return fail(error, "GFX MPSRT: MSL stage kernel buffer order is not materialized");
    }
    if (stage.msl_dispatch_desc.input_count + stage.msl_dispatch_desc.output_count != buffer_order.size()) {
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
    if (!is_mps_conv2d_stage(stage.kind)) {
        return fail(error, "GFX MPSRT: cannot encode non-Conv2D stage with MPS Conv2D");
    }
    if (!prepared.weights_buffer) {
        return fail(error, "GFX MPSRT: prepared MPS Conv2D weights buffer is null");
    }
    if (!prepared.kernel) {
        return fail(error, "GFX MPSRT: prepared MPS Conv2D kernel is null");
    }
    if (stage.inputs.size() != 2 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(error, "GFX MPSRT: MPS Conv2D requires input, weights and one output");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    const auto* weights_tensor = find_tensor(model, stage.inputs[1]);
    if (!input_tensor || !weights_tensor) {
        return fail(error, "GFX MPSRT: MPS Conv2D input or weights tensor descriptor is missing");
    }

    MpsrtBoundBuffer input_binding;
    MpsrtBoundBuffer output_binding;
    if (!lookup_bound_image(bindings, stage.inputs[0], "input", input_binding, error) ||
        !lookup_bound_image(bindings, stage.outputs[0], "output", output_binding, error)) {
        return false;
    }

    MPSImage* input_image = nil;
    MPSImage* output_image = nil;
    if (!make_mps_image_wrapper(input_tensor->desc, input_binding, "input", input_image, error) ||
        !make_mps_image_wrapper(stage.output_descs.front(), output_binding, "output", output_image, error)) {
        [input_image release];
        [output_image release];
        return false;
    }

    metal_end_compute_encoder(command_buffer);
    id<MTLCommandBuffer> command = static_cast<id<MTLCommandBuffer>>(command_buffer);
    const auto encode_start = hooks && hooks->on_segment ? std::chrono::steady_clock::now()
                                                         : std::chrono::steady_clock::time_point{};

    MPSCNNConvolution* kernel = static_cast<MPSCNNConvolution*>(prepared.kernel);
    kernel.edgeMode = MPSImageEdgeModeZero;
    kernel.offset = (MPSOffset){
        .x = mps_conv_offset(conv_kernel_width(weights_tensor->desc),
                             stage.conv2d_desc.dilations[1] == 0 ? 1 : stage.conv2d_desc.dilations[1],
                             stage.conv2d_desc.pads[1]),
        .y = mps_conv_offset(conv_kernel_height(weights_tensor->desc),
                             stage.conv2d_desc.dilations[0] == 0 ? 1 : stage.conv2d_desc.dilations[0],
                             stage.conv2d_desc.pads[0]),
        .z = 0,
    };
    kernel.clipRect = MTLRegionMake3D(0,
                                      0,
                                      0,
                                      stage.output_descs.front().image_width,
                                      stage.output_descs.front().image_height,
                                      stage.output_descs.front().image_batch);

    [kernel encodeToCommandBuffer:command sourceImage:input_image destinationImage:output_image];
    [input_image release];
    [output_image release];

    if (result) {
        result->bound_resources = 2;
        result->kernel_encodes = 1;
    }
    if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_mps_conv2d_request_encode_count", 1);
        hooks->on_counter("mpsrt_mps_conv2d_kernel_encode_count", 1);
        hooks->on_counter("mpsrt_mps_conv2d_bound_resource_count", 2);
    }
    if (hooks && hooks->on_segment) {
        const auto setup_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
        hooks->on_segment("mpsrt_encode",
                          stage.stage_record_key,
                          setup_cpu_us,
                          0,
                          2,
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

bool MpsrtRequest::encode_mps_pool2d(GpuCommandBufferHandle command_buffer,
                                     const MpsrtModel& model,
                                     const MpsrtRuntimeStage& stage,
                                     const MpsrtPreparedMpsPool2D& prepared,
                                     const MpsrtTensorBindings& bindings,
                                     const KernelExecutionHooks* hooks,
                                     MpsrtMpsPool2DEncodeResult* result,
                                     std::string* error) const {
    if (result) {
        *result = {};
    }
    OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
    if (stage.kind != GfxMpsrtStageKind::MPSPool2D) {
        return fail(error, "GFX MPSRT: cannot encode non-Pool2D stage with MPS Pool2D");
    }
    if (!prepared.kernel) {
        return fail(error, "GFX MPSRT: prepared MPS Pool2D kernel is null");
    }
    if (stage.inputs.size() != 1 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(error, "GFX MPSRT: MPS Pool2D requires one input and one output");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    if (!input_tensor) {
        return fail(error, "GFX MPSRT: MPS Pool2D input tensor descriptor is missing");
    }

    MpsrtBoundBuffer input_binding;
    MpsrtBoundBuffer output_binding;
    if (!lookup_bound_image(bindings, stage.inputs[0], "input", input_binding, error) ||
        !lookup_bound_image(bindings, stage.outputs[0], "output", output_binding, error)) {
        return false;
    }

    MPSImage* input_image = nil;
    MPSImage* output_image = nil;
    if (!make_mps_image_wrapper(input_tensor->desc, input_binding, "input", input_image, error) ||
        !make_mps_image_wrapper(stage.output_descs.front(), output_binding, "output", output_image, error)) {
        [input_image release];
        [output_image release];
        return false;
    }

    metal_end_compute_encoder(command_buffer);
    id<MTLCommandBuffer> command = static_cast<id<MTLCommandBuffer>>(command_buffer);
    const auto encode_start = hooks && hooks->on_segment ? std::chrono::steady_clock::now()
                                                         : std::chrono::steady_clock::time_point{};

    MPSCNNPooling* kernel = static_cast<MPSCNNPooling*>(prepared.kernel);
    kernel.edgeMode = MPSImageEdgeModeZero;
    kernel.offset = (MPSOffset){
        .x = mps_conv_offset(stage.pool2d_desc.kernel[1], 1, stage.pool2d_desc.pads[1]),
        .y = mps_conv_offset(stage.pool2d_desc.kernel[0], 1, stage.pool2d_desc.pads[0]),
        .z = 0,
    };
    kernel.clipRect = MTLRegionMake3D(0,
                                      0,
                                      0,
                                      stage.output_descs.front().image_width,
                                      stage.output_descs.front().image_height,
                                      stage.output_descs.front().image_batch);

    [kernel encodeToCommandBuffer:command sourceImage:input_image destinationImage:output_image];
    [input_image release];
    [output_image release];

    if (result) {
        result->bound_resources = 2;
        result->kernel_encodes = 1;
    }
    if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_mps_pool2d_request_encode_count", 1);
        hooks->on_counter("mpsrt_mps_pool2d_kernel_encode_count", 1);
        hooks->on_counter("mpsrt_mps_pool2d_bound_resource_count", 2);
    }
    if (hooks && hooks->on_segment) {
        const auto setup_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
        hooks->on_segment("mpsrt_encode",
                          stage.stage_record_key,
                          setup_cpu_us,
                          0,
                          2,
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

bool MpsrtRequest::encode_mps_resize2d(GpuCommandBufferHandle command_buffer,
                                       const MpsrtModel& model,
                                       const MpsrtRuntimeStage& stage,
                                       const MpsrtPreparedMpsResize2D& prepared,
                                       const MpsrtTensorBindings& bindings,
                                       const KernelExecutionHooks* hooks,
                                       MpsrtMpsResize2DEncodeResult* result,
                                       std::string* error) const {
    if (result) {
        *result = {};
    }
    OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
    if (stage.kind != GfxMpsrtStageKind::MPSResize2D) {
        return fail(error, "GFX MPSRT: cannot encode non-Resize2D stage with MPS Resize2D");
    }
    if (!prepared.kernel) {
        return fail(error, "GFX MPSRT: prepared MPS Resize2D kernel is null");
    }
    if (stage.resize2d_desc.nearest != 0) {
        return fail(error, "GFX MPSRT: MPS Resize2D encode supports bilinear mode only");
    }
    if (stage.inputs.size() != 1 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(error, "GFX MPSRT: MPS Resize2D requires one input and one output");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    if (!input_tensor) {
        return fail(error, "GFX MPSRT: MPS Resize2D input tensor descriptor is missing");
    }

    MpsrtBoundBuffer input_binding;
    MpsrtBoundBuffer output_binding;
    if (!lookup_bound_image(bindings, stage.inputs[0], "input", input_binding, error) ||
        !lookup_bound_image(bindings, stage.outputs[0], "output", output_binding, error)) {
        return false;
    }

    MPSImage* input_image = nil;
    MPSImage* output_image = nil;
    if (!make_mps_image_wrapper(input_tensor->desc, input_binding, "input", input_image, error) ||
        !make_mps_image_wrapper(stage.output_descs.front(), output_binding, "output", output_image, error)) {
        [input_image release];
        [output_image release];
        return false;
    }

    metal_end_compute_encoder(command_buffer);
    id<MTLCommandBuffer> command = static_cast<id<MTLCommandBuffer>>(command_buffer);
    const auto encode_start = hooks && hooks->on_segment ? std::chrono::steady_clock::now()
                                                         : std::chrono::steady_clock::time_point{};

    MPSImageBilinearScale* kernel = static_cast<MPSImageBilinearScale*>(prepared.kernel);
    kernel.edgeMode = MPSImageEdgeModeClamp;
    kernel.clipRect = MTLRegionMake3D(0,
                                      0,
                                      0,
                                      stage.output_descs.front().image_width,
                                      stage.output_descs.front().image_height,
                                      stage.output_descs.front().image_batch);
    kernel.scaleTransform = nullptr;
    [kernel encodeToCommandBuffer:command sourceImage:input_image destinationImage:output_image];
    [input_image release];
    [output_image release];

    if (result) {
        result->bound_resources = 2;
        result->kernel_encodes = 1;
    }
    if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_mps_resize2d_request_encode_count", 1);
        hooks->on_counter("mpsrt_mps_resize2d_kernel_encode_count", 1);
        hooks->on_counter("mpsrt_mps_resize2d_bound_resource_count", 2);
    }
    if (hooks && hooks->on_segment) {
        const auto setup_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
        hooks->on_segment("mpsrt_encode",
                          stage.stage_record_key,
                          setup_cpu_us,
                          0,
                          2,
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

bool MpsrtRequest::encode_mps_softmax(GpuCommandBufferHandle command_buffer,
                                      const MpsrtModel& model,
                                      const MpsrtRuntimeStage& stage,
                                      const MpsrtPreparedMpsSoftmax& prepared,
                                      const MpsrtTensorBindings& bindings,
                                      const KernelExecutionHooks* hooks,
                                      MpsrtMpsSoftmaxEncodeResult* result,
                                      std::string* error) const {
    if (result) {
        *result = {};
    }
    OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
    if (stage.kind != GfxMpsrtStageKind::MPSSoftmax) {
        return fail(error, "GFX MPSRT: cannot encode non-Softmax stage with MPS Softmax");
    }
    if (!prepared.kernel) {
        return fail(error, "GFX MPSRT: prepared MPS Softmax kernel is null");
    }
    if (stage.softmax_desc.log_softmax != 0) {
        return fail(error, "GFX MPSRT: MPS Softmax encode does not implement LogSoftmax");
    }
    if (stage.inputs.size() != 1 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(error, "GFX MPSRT: MPS Softmax requires one input and one output");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    if (!input_tensor) {
        return fail(error, "GFX MPSRT: MPS Softmax input tensor descriptor is missing");
    }

    MpsrtBoundBuffer input_buffer;
    MpsrtBoundBuffer output_buffer;
    if (!lookup_bound_buffer(bindings, stage.inputs[0], "input", input_buffer, error) ||
        !lookup_bound_buffer(bindings, stage.outputs[0], "output", output_buffer, error)) {
        return false;
    }

    MPSMatrixDescriptor* input_desc = nil;
    MPSMatrixDescriptor* output_desc = nil;
    if (!make_mps_matrix_descriptor(input_tensor->desc, input_desc, "input", error) ||
        !make_mps_matrix_descriptor(stage.output_descs.front(), output_desc, "output", error)) {
        return false;
    }

    MPSMatrix* input_matrix =
        [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(input_buffer.buffer)
                                   offset:static_cast<NSUInteger>(input_buffer.offset +
                                                                  input_tensor->desc.byte_offset)
                               descriptor:input_desc];
    MPSMatrix* output_matrix =
        [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(output_buffer.buffer)
                                   offset:static_cast<NSUInteger>(output_buffer.offset +
                                                                  stage.output_descs.front().byte_offset)
                               descriptor:output_desc];
    if (!input_matrix || !output_matrix) {
        [input_matrix release];
        [output_matrix release];
        return fail(error, "GFX MPSRT: failed to create MPS Softmax matrix wrappers");
    }

    metal_end_compute_encoder(command_buffer);
    id<MTLCommandBuffer> command = static_cast<id<MTLCommandBuffer>>(command_buffer);
    const auto encode_start = hooks && hooks->on_segment ? std::chrono::steady_clock::now()
                                                         : std::chrono::steady_clock::time_point{};

    [(MPSMatrixSoftMax*)prepared.kernel encodeToCommandBuffer:command
                                                  inputMatrix:input_matrix
                                                 resultMatrix:output_matrix];
    [input_matrix release];
    [output_matrix release];

    if (result) {
        result->bound_buffers = 2;
        result->kernel_encodes = 1;
    }
    if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_mps_softmax_request_encode_count", 1);
        hooks->on_counter("mpsrt_mps_softmax_kernel_encode_count", 1);
        hooks->on_counter("mpsrt_mps_softmax_bound_buffer_count", 2);
    }
    if (hooks && hooks->on_segment) {
        const auto setup_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
        hooks->on_segment("mpsrt_encode",
                          stage.stage_record_key,
                          setup_cpu_us,
                          0,
                          2,
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

bool MpsrtRequest::encode_mps_topk(GpuCommandBufferHandle command_buffer,
                                   const MpsrtModel& model,
                                   const MpsrtRuntimeStage& stage,
                                   const MpsrtPreparedMpsTopK& prepared,
                                   const MpsrtTensorBindings& bindings,
                                   const KernelExecutionHooks* hooks,
                                   MpsrtMpsTopKEncodeResult* result,
                                   std::string* error) const {
    if (result) {
        *result = {};
    }
    OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
    if (stage.kind != GfxMpsrtStageKind::MPSTopK) {
        return fail(error, "GFX MPSRT: cannot encode non-TopK stage with MPS TopK");
    }
    if (!prepared.kernel) {
        return fail(error, "GFX MPSRT: prepared MPS TopK kernel is null");
    }
    if (stage.topk_desc.mode_max == 0) {
        return fail(error, "GFX MPSRT: MPS TopK encode supports MAX mode only");
    }
    if (stage.inputs.size() != 1 || stage.outputs.size() != 2 || stage.output_descs.size() != 2) {
        return fail(error, "GFX MPSRT: MPS TopK requires one input and two outputs");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    if (!input_tensor) {
        return fail(error, "GFX MPSRT: MPS TopK input tensor descriptor is missing");
    }

    MpsrtBoundBuffer input_buffer;
    MpsrtBoundBuffer values_buffer;
    MpsrtBoundBuffer indices_buffer;
    if (!lookup_bound_buffer(bindings, stage.inputs[0], "input", input_buffer, error) ||
        !lookup_bound_buffer(bindings, stage.outputs[0], "values output", values_buffer, error) ||
        !lookup_bound_buffer(bindings, stage.outputs[1], "indices output", indices_buffer, error)) {
        return false;
    }

    MPSMatrixDescriptor* input_desc = nil;
    MPSMatrixDescriptor* values_desc = nil;
    MPSMatrixDescriptor* indices_desc = nil;
    if (!make_mps_matrix_descriptor(input_tensor->desc, input_desc, "input", error) ||
        !make_mps_matrix_descriptor(stage.output_descs[0], values_desc, "values output", error) ||
        !make_mps_topk_index_matrix_descriptor(stage.output_descs[1],
                                               indices_desc,
                                               "indices output",
                                               error)) {
        return false;
    }

    MPSMatrix* input_matrix =
        [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(input_buffer.buffer)
                                   offset:static_cast<NSUInteger>(input_buffer.offset +
                                                                  input_tensor->desc.byte_offset)
                               descriptor:input_desc];
    MPSMatrix* values_matrix =
        [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(values_buffer.buffer)
                                   offset:static_cast<NSUInteger>(values_buffer.offset +
                                                                  stage.output_descs[0].byte_offset)
                               descriptor:values_desc];
    MPSMatrix* indices_matrix =
        [[MPSMatrix alloc] initWithBuffer:static_cast<id<MTLBuffer>>(indices_buffer.buffer)
                                   offset:static_cast<NSUInteger>(indices_buffer.offset +
                                                                  stage.output_descs[1].byte_offset)
                               descriptor:indices_desc];
    if (!input_matrix || !values_matrix || !indices_matrix) {
        [input_matrix release];
        [values_matrix release];
        [indices_matrix release];
        return fail(error, "GFX MPSRT: failed to create MPS TopK matrix wrappers");
    }

    metal_end_compute_encoder(command_buffer);
    id<MTLCommandBuffer> command = static_cast<id<MTLCommandBuffer>>(command_buffer);
    const auto encode_start = hooks && hooks->on_segment ? std::chrono::steady_clock::now()
                                                         : std::chrono::steady_clock::time_point{};

    MPSMatrixFindTopK* kernel = static_cast<MPSMatrixFindTopK*>(prepared.kernel);
    kernel.sourceRows = prepared.rows;
    kernel.sourceColumns = prepared.source_columns;
    kernel.numberOfTopKValues = prepared.k;
    kernel.indexOffset = 0;
    [kernel encodeToCommandBuffer:command
                       inputMatrix:input_matrix
                 resultIndexMatrix:indices_matrix
                 resultValueMatrix:values_matrix];
    [input_matrix release];
    [values_matrix release];
    [indices_matrix release];

    if (result) {
        result->bound_buffers = 3;
        result->kernel_encodes = 1;
    }
    if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_mps_topk_request_encode_count", 1);
        hooks->on_counter("mpsrt_mps_topk_kernel_encode_count", 1);
        hooks->on_counter("mpsrt_mps_topk_bound_buffer_count", 3);
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

        if (stage.kind == GfxMpsrtStageKind::MPSPool2D) {
            const auto* prepared = find_prepared_mps_pool2d(prepared_model, stage_index);
            if (!prepared) {
                return fail(error, "GFX MPSRT: missing prepared MPS Pool2D for stage " + std::to_string(stage_index));
            }
            MpsrtMpsPool2DEncodeResult stage_result;
            if (!encode_mps_pool2d(command_buffer,
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
                ++result->encoded_mps_pool2d_stages;
                result->bound_buffers += stage_result.bound_resources;
            }
            if (hooks && hooks->on_counter) {
                hooks->on_counter("mpsrt_model_request_mps_pool2d_stage_encode_count", 1);
            }
            continue;
        }

        if (stage.kind == GfxMpsrtStageKind::MPSResize2D) {
            const auto* prepared = find_prepared_mps_resize2d(prepared_model, stage_index);
            if (!prepared) {
                return fail(error, "GFX MPSRT: missing prepared MPS Resize2D for stage " + std::to_string(stage_index));
            }
            MpsrtMpsResize2DEncodeResult stage_result;
            if (!encode_mps_resize2d(command_buffer,
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
                ++result->encoded_mps_resize2d_stages;
                result->bound_buffers += stage_result.bound_resources;
            }
            if (hooks && hooks->on_counter) {
                hooks->on_counter("mpsrt_model_request_mps_resize2d_stage_encode_count", 1);
            }
            continue;
        }

        if (stage.kind == GfxMpsrtStageKind::MPSSoftmax) {
            const auto* prepared = find_prepared_mps_softmax(prepared_model, stage_index);
            if (!prepared) {
                return fail(error, "GFX MPSRT: missing prepared MPS Softmax for stage " + std::to_string(stage_index));
            }
            MpsrtMpsSoftmaxEncodeResult stage_result;
            if (!encode_mps_softmax(command_buffer,
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
                ++result->encoded_mps_softmax_stages;
                result->bound_buffers += stage_result.bound_buffers;
            }
            if (hooks && hooks->on_counter) {
                hooks->on_counter("mpsrt_model_request_mps_softmax_stage_encode_count", 1);
            }
            continue;
        }

        if (stage.kind == GfxMpsrtStageKind::MPSTopK) {
            const auto* prepared = find_prepared_mps_topk(prepared_model, stage_index);
            if (!prepared) {
                return fail(error, "GFX MPSRT: missing prepared MPS TopK for stage " + std::to_string(stage_index));
            }
            MpsrtMpsTopKEncodeResult stage_result;
            if (!encode_mps_topk(command_buffer,
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
                ++result->encoded_mps_topk_stages;
                result->bound_buffers += stage_result.bound_buffers;
            }
            if (hooks && hooks->on_counter) {
                hooks->on_counter("mpsrt_model_request_mps_topk_stage_encode_count", 1);
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
