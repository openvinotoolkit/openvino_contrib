// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/mpsrt/mpsrt_context.hpp"

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <utility>

#include "kernel_ir/gfx_kernel_cache.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "backends/metal/common/mpsrt/gfx_mpsrt_storage_bridge.hpp"

@interface OVGfxMpsrtConv2DDataSource : NSObject <MPSCNNConvolutionDataSource> {
@private
    MPSCNNConvolutionDescriptor* _descriptor;
    std::vector<uint8_t> _weights;
    std::vector<float> _bias;
    MPSDataType _dataType;
    NSString* _label;
}
- (instancetype)initWithDescriptor:(MPSCNNConvolutionDescriptor*)descriptor
                           weights:(std::vector<uint8_t>)weights
                              bias:(std::vector<float>)bias
                          dataType:(MPSDataType)dataType
                             label:(NSString*)label;
@end

@implementation OVGfxMpsrtConv2DDataSource

- (instancetype)initWithDescriptor:(MPSCNNConvolutionDescriptor*)descriptor
                           weights:(std::vector<uint8_t>)weights
                              bias:(std::vector<float>)bias
                          dataType:(MPSDataType)dataType
                             label:(NSString*)label {
    self = [super init];
    if (self) {
        _descriptor = [descriptor retain];
        _weights = std::move(weights);
        _bias = std::move(bias);
        _dataType = dataType;
        _label = [label copy];
    }
    return self;
}

- (void)dealloc {
    [_descriptor release];
    [_label release];
    [super dealloc];
}

- (id)copyWithZone:(NSZone*)zone {
    OVGfxMpsrtConv2DDataSource* copy = [[[self class] allocWithZone:zone] initWithDescriptor:_descriptor
                                                                                     weights:_weights
                                                                                        bias:_bias
                                                                                    dataType:_dataType
                                                                                       label:_label];
    return copy;
}

- (MPSDataType)dataType {
    return _dataType;
}

- (MPSCNNConvolutionDescriptor*)descriptor {
    return _descriptor;
}

- (void*)weights {
    return _weights.data();
}

- (float*)biasTerms {
    return _bias.empty() ? nullptr : _bias.data();
}

- (BOOL)load {
    return !_weights.empty() && _descriptor != nil;
}

- (void)purge {
}

- (NSString*)label {
    return _label;
}

- (MPSCNNConvolutionWeightsLayout)weightsLayout {
    return MPSCNNConvolutionWeightsLayoutOHWI;
}

- (MPSDataType)kernelWeightsDataType {
    return _dataType;
}

@end

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {

namespace runtime_mpsrt = ::ov::gfx_plugin::mpsrt;
using runtime_mpsrt::MpsrtModel;
using runtime_mpsrt::MpsrtRuntimeResource;
using runtime_mpsrt::MpsrtRuntimeResourceLifetime;
using runtime_mpsrt::MpsrtRuntimeStage;
using runtime_mpsrt::MpsrtRuntimeTensor;
using runtime_mpsrt::MpsrtTensorBindingPlanEntry;

namespace {

std::string ns_error_message(NSError* error, const char* fallback) {
    if (!error) {
        return fallback ? std::string(fallback) : std::string("unknown Metal error");
    }
    NSString* description = [error localizedDescription];
    return description ? std::string([description UTF8String]) : std::string("unknown Metal error");
}

std::string make_pipeline_cache_key(const MpsrtRuntimeStage& stage, const std::string& source) {
    std::ostringstream key;
    key << stage.dispatch_kernel_family_id << '|' << stage.dispatch_entry_point << '|'
        << stage.dispatch_threads_per_threadgroup << '|' << stage.dispatch_flags << '|'
        << std::hash<std::string>{}(source);
    return key.str();
}

struct SharedPipelineCacheEntry {
    uintptr_t device_key = 0;
    std::string key;
    id<MTLComputePipelineState> pipeline = nil;
};

std::mutex& shared_pipeline_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<SharedPipelineCacheEntry>& shared_pipeline_cache_entries() {
    static std::vector<SharedPipelineCacheEntry> entries;
    return entries;
}

id<MTLComputePipelineState> lookup_shared_pipeline_cache(id<MTLDevice> device,
                                                         const std::string& key) {
    const uintptr_t device_key = reinterpret_cast<uintptr_t>(device);
    std::lock_guard<std::mutex> lock(shared_pipeline_cache_mutex());
    for (const auto& entry : shared_pipeline_cache_entries()) {
        if (entry.device_key == device_key && entry.key == key && entry.pipeline) {
            return entry.pipeline;
        }
    }
    return nil;
}

void store_shared_pipeline_cache(id<MTLDevice> device,
                                 const std::string& key,
                                 id<MTLComputePipelineState> pipeline) {
    if (!device || !pipeline) {
        return;
    }
    const uintptr_t device_key = reinterpret_cast<uintptr_t>(device);
    std::lock_guard<std::mutex> lock(shared_pipeline_cache_mutex());
    for (const auto& entry : shared_pipeline_cache_entries()) {
        if (entry.device_key == device_key && entry.key == key && entry.pipeline) {
            return;
        }
    }
    [pipeline retain];
    shared_pipeline_cache_entries().push_back({device_key, key, pipeline});
}

bool fail(std::string* log, const std::string& message) {
    if (log) {
        *log = message;
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

uint64_t tensor_dense_bytes(const GfxMpsrtTensorAbiDesc& desc) {
    const auto dtype = static_cast<GfxMpsrtDType>(desc.dtype);
    const uint32_t elem_bytes = gfx_mpsrt_element_size_bytes(dtype);
    if (elem_bytes == 0 || desc.rank == 0 || desc.rank > 8) {
        return 0;
    }
    uint64_t elements = 1;
    for (uint32_t i = 0; i < desc.rank; ++i) {
        if (desc.dims[i] == 0) {
            return 0;
        }
        elements *= desc.dims[i];
    }
    return elements * elem_bytes;
}

std::string make_const_tensor_cache_key(GfxMpsrtValue value, const GfxMpsrtTensorAbiDesc& desc, size_t bytes,
                                        uint64_t data_hash) {
    std::ostringstream stream;
    stream << "const|" << value << '|' << desc.dtype << '|' << desc.storage << '|' << desc.layout << '|' << desc.rank
           << '|' << bytes << '|' << data_hash;
    for (uint32_t i = 0; i < desc.rank && i < 8; ++i) {
        stream << '|' << desc.dims[i];
    }
    return stream.str();
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

NSArray<NSNumber*>* mps_shape_from_tensor_desc(const GfxMpsrtTensorAbiDesc& desc) {
    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:desc.rank];
    for (uint32_t i = 0; i < desc.rank && i < 8; ++i) {
        [shape addObject:@(desc.dims[i])];
    }
    return shape;
}

MTLPixelFormat texture_pixel_format_from_gfx(uint32_t dtype) {
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
    return (feature_channels + 3u) / 4u;
}

bool tensor_requires_image_resource(const GfxMpsrtTensorAbiDesc& desc) {
    return desc.storage == static_cast<uint32_t>(GfxMpsrtStorage::Image);
}

MTLTextureDescriptor* new_transient_image_texture_descriptor(const GfxMpsrtTensorAbiDesc& desc, std::string* log) {
    if (!tensor_requires_image_resource(desc)) {
        (void)fail(log, "GFX MPSRT: cannot prepare transient texture descriptor "
                        "for non-image resource");
        return nil;
    }
    if (desc.image_width == 0 || desc.image_height == 0 || desc.image_feature_channels == 0 || desc.image_batch == 0) {
        (void)fail(log, "GFX MPSRT: cannot prepare transient texture descriptor "
                        "for incomplete image descriptor");
        return nil;
    }
    const MTLPixelFormat pixel_format = texture_pixel_format_from_gfx(desc.dtype);
    if (pixel_format == MTLPixelFormatInvalid) {
        (void)fail(log, "GFX MPSRT: cannot prepare transient texture descriptor "
                        "for unsupported image dtype");
        return nil;
    }

    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                                                                                          width:desc.image_width
                                                                                         height:desc.image_height
                                                                                      mipmapped:false];
    descriptor.textureType = MTLTextureType2DArray;
    descriptor.arrayLength = static_cast<NSUInteger>(desc.image_batch * image_slice_count(desc.image_feature_channels));
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    descriptor.storageMode = MTLStorageModePrivate;
    return descriptor;
}

id<MTLTexture> new_transient_image_texture(id<MTLHeap> heap, const GfxMpsrtTensorAbiDesc& desc, std::string* log) {
    if (!heap) {
        (void)fail(log, "GFX MPSRT: prepared resource heap is required for transient texture");
        return nil;
    }
    MTLTextureDescriptor* descriptor = new_transient_image_texture_descriptor(desc, log);
    if (!descriptor) {
        return nil;
    }
    id<MTLTexture> texture = [heap newTextureWithDescriptor:descriptor];
    if (!texture) {
        (void)fail(log, "GFX MPSRT: failed to allocate prepared transient texture "
                        "from resource heap");
    }
    return texture;
}

uint64_t transient_buffer_allocation_bytes(const GfxMpsrtTensorAbiDesc& desc) {
    const uint64_t logical_bytes = desc.byte_length != 0 ? desc.byte_length : tensor_dense_bytes(desc);
    if (logical_bytes == 0) {
        return 0;
    }
    return desc.byte_offset + logical_bytes;
}

bool allocate_prepared_transient_resource_from_heap(id<MTLHeap> heap, MpsrtPreparedResource& prepared,
                                                    std::string* log) {
    if (!prepared.has_tensor_value) {
        return fail(log, "GFX MPSRT: transient resource is not a tensor");
    }
    if (tensor_requires_image_resource(prepared.tensor_desc)) {
        prepared.texture = new_transient_image_texture(heap, prepared.tensor_desc, log);
        if (!prepared.texture) {
            return false;
        }
        prepared.byte_length = 0;
        prepared.offset = 0;
        return true;
    }

    const uint64_t allocation_bytes = transient_buffer_allocation_bytes(prepared.tensor_desc);
    if (allocation_bytes == 0) {
        return fail(log, "GFX MPSRT: transient buffer resource has invalid byte size");
    }
    id<MTLBuffer> buffer = [heap newBufferWithLength:static_cast<NSUInteger>(allocation_bytes)
                                             options:MTLResourceStorageModePrivate];
    if (!buffer) {
        return fail(log, "GFX MPSRT: failed to allocate prepared transient buffer "
                         "from resource heap");
    }
    prepared.byte_length = static_cast<size_t>(allocation_bytes);
    prepared.offset = 0;
    prepared.buffer = buffer;
    return true;
}

NSUInteger align_heap_size(NSUInteger value, NSUInteger alignment) {
    if (alignment <= 1) {
        return value;
    }
    const NSUInteger remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

struct PreparedResourceHeapPlanEntry {
    uint32_t resource_index = 0;
    size_t allocation_size = 0;
    size_t alignment = 1;
    size_t first_stage_index = 0;
    size_t last_stage_index = 0;
};

struct PreparedImageBridgeHeapPlanEntry {
    GfxMpsrtValue value = 0;
    GfxMpsrtStorageBridgeDirection direction = GfxMpsrtStorageBridgeDirection::Unknown;
    GfxMpsrtTensorAbiDesc tensor_desc{};
    size_t allocation_size = 0;
    size_t alignment = 1;
};

struct ActivePreparedHeapResource {
    id<MTLResource> resource = nil;
    size_t last_stage_index = 0;
};

const PreparedResourceHeapPlanEntry* find_heap_plan_entry(const std::vector<PreparedResourceHeapPlanEntry>& entries,
                                                          uint32_t resource_index) {
    for (const auto& entry : entries) {
        if (entry.resource_index == resource_index) {
            return &entry;
        }
    }
    return nullptr;
}

MpsrtPreparedResource* find_prepared_resource(std::vector<MpsrtPreparedResource>& resources, uint32_t resource_index) {
    for (auto& resource : resources) {
        if (resource.resource_index == resource_index) {
            return &resource;
        }
    }
    return nullptr;
}

const MpsrtPreparedResource* find_prepared_model_const_resource(const MpsrtPreparedModel& prepared_model,
                                                               GfxMpsrtValue value,
                                                               std::string* log) {
    for (const auto& resource : prepared_model.resources) {
        if (!resource.has_tensor_value || resource.value != value) {
            continue;
        }
        if (resource.lifetime != MpsrtRuntimeResourceLifetime::Model ||
            resource.role != GfxMpsrtExternalBufferRole::ConstBuffer) {
            (void)fail(log, "GFX MPSRT: prepared resource for model value is not a model-owned const buffer");
            return nullptr;
        }
        if (!resource.buffer || resource.byte_length == 0 || resource.cache_key.empty() ||
            resource.host_bytes.empty()) {
            (void)fail(log, "GFX MPSRT: prepared model const resource is not materialized");
            return nullptr;
        }
        return &resource;
    }
    (void)fail(log, "GFX MPSRT: prepared model const resource is missing for value " + std::to_string(value));
    return nullptr;
}

id<MTLResource> prepared_heap_resource(const MpsrtPreparedResource& prepared) {
    if (prepared.buffer) {
        return prepared.buffer;
    }
    if (prepared.texture) {
        return prepared.texture;
    }
    return nil;
}

bool stage_uses_value(const MpsrtRuntimeStage& stage, GfxMpsrtValue value) {
    return runtime_mpsrt::mpsrt_value_list_contains(stage.inputs, value) ||
           runtime_mpsrt::mpsrt_value_list_contains(stage.outputs, value) ||
           runtime_mpsrt::mpsrt_value_list_contains(stage.kernel_buffer_order, value);
}

std::pair<size_t, size_t> transient_resource_live_window(const MpsrtModel& model,
                                                         const MpsrtRuntimeResource& resource) {
    if (!resource.has_tensor_value || model.stages.empty()) {
        return {0, 0};
    }

    const size_t sentinel = model.stages.size();
    size_t first_stage = sentinel;
    size_t last_stage = 0;
    for (size_t stage_index = 0; stage_index < model.stages.size(); ++stage_index) {
        if (!stage_uses_value(model.stages[stage_index], resource.value)) {
            continue;
        }
        first_stage = std::min(first_stage, stage_index);
        last_stage = std::max(last_stage, stage_index);
    }
    if (first_stage == sentinel) {
        return {0, model.stages.size() - 1};
    }
    return {first_stage, last_stage};
}

bool has_image_bridge_heap_plan_entry(const std::vector<PreparedImageBridgeHeapPlanEntry>& entries, GfxMpsrtValue value,
                                      GfxMpsrtStorageBridgeDirection direction) {
    for (const auto& entry : entries) {
        if (entry.value == value && entry.direction == direction) {
            return true;
        }
    }
    return false;
}

bool append_image_bridge_heap_plan_entry(id<MTLDevice> device, const MpsrtModel& model, GfxMpsrtValue value,
                                         GfxMpsrtStorageBridgeDirection fallback_direction, NSUInteger& required_size,
                                         std::vector<PreparedImageBridgeHeapPlanEntry>& entries, std::string* log) {
    const auto direction =
        runtime_mpsrt::mpsrt_model_external_bridge_direction_for_value(model, value, fallback_direction);
    if (direction == GfxMpsrtStorageBridgeDirection::Unknown ||
        has_image_bridge_heap_plan_entry(entries, value, direction)) {
        return true;
    }
    const auto* tensor = find_tensor(model, value);
    if (!tensor || !gfx_mpsrt_tensor_is_image(tensor->desc)) {
        return true;
    }
    if (!gfx_mpsrt_image_bridge_supported(tensor->desc)) {
        std::ostringstream stream;
        stream << "GFX MPSRT: image bridge supports only static rank-4 f16/f32 "
                  "image tensors"
               << " value=" << value << " rank=" << tensor->desc.rank << " storage=" << tensor->desc.storage
               << " flags=" << tensor->desc.flags;
        return fail(log, stream.str());
    }
    GfxMpsrtStorageBridgeDesc bridge_desc{};
    if (!gfx_mpsrt_make_image_bridge_desc(value, tensor->desc, direction, bridge_desc)) {
        return fail(log, "GFX MPSRT: image bridge storage contract is invalid");
    }
    MTLTextureDescriptor* descriptor = new_transient_image_texture_descriptor(tensor->desc, log);
    if (!descriptor) {
        return false;
    }
    const MTLSizeAndAlign size_align = [device heapTextureSizeAndAlignWithDescriptor:descriptor];
    if (size_align.size == 0) {
        return fail(log, "GFX MPSRT: image bridge texture has invalid heap allocation size");
    }
    required_size = align_heap_size(required_size, size_align.align);
    required_size += size_align.size;
    entries.push_back({bridge_desc.value, bridge_desc.direction, bridge_desc.tensor,
                       static_cast<size_t>(size_align.size), static_cast<size_t>(size_align.align)});
    return true;
}

bool append_image_bridge_heap_plan_entry(id<MTLDevice> device, const MpsrtModel& model,
                                         const MpsrtTensorBindingPlanEntry& binding, NSUInteger& required_size,
                                         std::vector<PreparedImageBridgeHeapPlanEntry>& entries, std::string* log) {
    if (binding.lifetime != MpsrtRuntimeResourceLifetime::External || !binding.has_tensor_value) {
        return true;
    }
    return append_image_bridge_heap_plan_entry(device,
                                               model,
                                               binding.value,
                                               binding.bridge_direction,
                                               required_size,
                                               entries,
                                               log);
}

bool append_image_bridge_heap_plan_entries(id<MTLDevice> device, const MpsrtModel& model, NSUInteger& required_size,
                                           std::vector<PreparedImageBridgeHeapPlanEntry>& entries, std::string* log) {
    std::vector<MpsrtTensorBindingPlanEntry> binding_plan;
    if (!runtime_mpsrt::mpsrt_model_tensor_binding_plan(model, binding_plan, log)) {
        return false;
    }
    if (!binding_plan.empty()) {
        for (const auto& binding : binding_plan) {
            if (!append_image_bridge_heap_plan_entry(device, model, binding, required_size, entries, log)) {
                return false;
            }
        }
        return true;
    }

    for (const auto value : model.input_values) {
        if (!append_image_bridge_heap_plan_entry(device, model, value, GfxMpsrtStorageBridgeDirection::BufferToImage,
                                                 required_size, entries, log)) {
            return false;
        }
    }
    for (const auto value : model.output_values) {
        if (!append_image_bridge_heap_plan_entry(device, model, value, GfxMpsrtStorageBridgeDirection::ImageToBuffer,
                                                 required_size, entries, log)) {
            return false;
        }
    }
    return true;
}

bool plan_prepared_resource_heap(id<MTLDevice> device, const MpsrtModel& model, id<MTLHeap>& heap, size_t& heap_size,
                                 size_t& heap_unaliased_size, size_t& heap_aliasable_size,
                                 size_t& transient_buffer_count, size_t& transient_image_count,
                                 std::vector<PreparedResourceHeapPlanEntry>& entries,
                                 std::vector<PreparedImageBridgeHeapPlanEntry>& image_bridge_entries,
                                 std::string* log) {
    heap = nil;
    heap_size = 0;
    heap_unaliased_size = 0;
    heap_aliasable_size = 0;
    transient_buffer_count = 0;
    transient_image_count = 0;
    entries.clear();
    image_bridge_entries.clear();

    NSUInteger required_size = 0;
    for (const auto& resource : model.resources) {
        if (resource.lifetime != MpsrtRuntimeResourceLifetime::Transient) {
            continue;
        }
        if (!resource.has_tensor_value) {
            return fail(log, "GFX MPSRT: transient resource is not a tensor");
        }
        if (tensor_requires_image_resource(resource.tensor_desc)) {
            MTLTextureDescriptor* descriptor = new_transient_image_texture_descriptor(resource.tensor_desc, log);
            if (!descriptor) {
                return false;
            }
            const MTLSizeAndAlign size_align = [device heapTextureSizeAndAlignWithDescriptor:descriptor];
            if (size_align.size == 0) {
                return fail(log, "GFX MPSRT: transient texture has invalid heap allocation size");
            }
            required_size = align_heap_size(required_size, size_align.align);
            required_size += size_align.size;
            const auto live_window = transient_resource_live_window(model, resource);
            entries.push_back({resource.resource_index, static_cast<size_t>(size_align.size),
                               static_cast<size_t>(size_align.align), live_window.first, live_window.second});
            ++transient_image_count;
            continue;
        }

        const uint64_t allocation_bytes = transient_buffer_allocation_bytes(resource.tensor_desc);
        if (allocation_bytes == 0) {
            return fail(log, "GFX MPSRT: transient buffer resource has invalid byte size");
        }
        const MTLSizeAndAlign size_align =
            [device heapBufferSizeAndAlignWithLength:static_cast<NSUInteger>(allocation_bytes)
                                             options:MTLResourceStorageModePrivate];
        if (size_align.size == 0) {
            return fail(log, "GFX MPSRT: transient buffer has invalid heap allocation size");
        }
        required_size = align_heap_size(required_size, size_align.align);
        required_size += size_align.size;
        const auto live_window = transient_resource_live_window(model, resource);
        entries.push_back({resource.resource_index, static_cast<size_t>(size_align.size),
                           static_cast<size_t>(size_align.align), live_window.first, live_window.second});
        ++transient_buffer_count;
    }

    if (!append_image_bridge_heap_plan_entries(device, model, required_size, image_bridge_entries, log)) {
        return false;
    }
    NSUInteger always_live_image_bridge_size = 0;
    for (const auto& entry : image_bridge_entries) {
        always_live_image_bridge_size =
            align_heap_size(always_live_image_bridge_size, static_cast<NSUInteger>(entry.alignment));
        always_live_image_bridge_size += static_cast<NSUInteger>(entry.allocation_size);
    }

    heap_unaliased_size = static_cast<size_t>(required_size);
    for (size_t stage_index = 0; stage_index < std::max<size_t>(model.stages.size(), 1); ++stage_index) {
        NSUInteger active_size = always_live_image_bridge_size;
        for (const auto& entry : entries) {
            if (entry.first_stage_index > stage_index || entry.last_stage_index < stage_index) {
                continue;
            }
            active_size = align_heap_size(active_size, static_cast<NSUInteger>(entry.alignment));
            active_size += static_cast<NSUInteger>(entry.allocation_size);
        }
        heap_aliasable_size = std::max(heap_aliasable_size, static_cast<size_t>(active_size));
    }
    if (entries.empty() && image_bridge_entries.empty()) {
        heap_aliasable_size = 0;
    } else if (heap_aliasable_size == 0) {
        heap_aliasable_size = heap_unaliased_size;
    }

    if (required_size == 0) {
        return true;
    }

    const NSUInteger planned_heap_size =
        heap_aliasable_size != 0 ? static_cast<NSUInteger>(heap_aliasable_size) : required_size;
    MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.size = planned_heap_size;
    heap = [device newHeapWithDescriptor:descriptor];
    [descriptor release];
    if (!heap) {
        return fail(log, "GFX MPSRT: failed to allocate prepared transient resource heap");
    }
    heap_size = static_cast<size_t>(planned_heap_size);
    return true;
}

MPSNNNeuronDescriptor* make_conv_fused_neuron_descriptor(uint32_t fused_activation, std::string* log) {
    switch (fused_activation) {
    case 0:
        return nil;
    case 1:
        return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLU a:0.0f];
    case 2:
        return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeSigmoid];
    case 3:
        return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeTanH a:1.0f b:1.0f];
    case 10:
        return [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeAbsolute];
    default:
        fail(log, "GFX MPSRT: unsupported MPS Conv2D fused activation code");
        return nil;
    }
}

uint32_t conv2d_kernel_height(const GfxMpsrtTensorAbiDesc& weights) {
    return weights.rank == 5 ? weights.dims[3] : weights.dims[2];
}

uint32_t conv2d_kernel_width(const GfxMpsrtTensorAbiDesc& weights) {
    return weights.rank == 5 ? weights.dims[4] : weights.dims[3];
}

uint32_t conv2d_output_channels(const GfxMpsrtTensorAbiDesc& weights) {
    return weights.rank == 5 ? weights.dims[0] * weights.dims[1] : weights.dims[0];
}

uint32_t conv2d_input_channels_per_group(const GfxMpsrtTensorAbiDesc& weights) {
    return weights.rank == 5 ? weights.dims[2] : weights.dims[1];
}

uint32_t conv2d_output_channels_per_group(const GfxMpsrtTensorAbiDesc& weights) {
    return weights.rank == 5 ? weights.dims[1] : weights.dims[0];
}

bool is_mps_cnn_depthwise_conv2d_candidate(const GfxMpsrtTensorAbiDesc& weights,
                                           const GfxMpsrtTensorAbiDesc& input,
                                           const GfxMpsrtTensorAbiDesc& output,
                                           const MpsrtRuntimeStage& stage,
                                           const GfxMpsrtTensorAbiDesc* bias) {
    if (stage.kind != GfxMpsrtStageKind::MPSGroupConv2D || weights.rank != 5 ||
        stage.conv2d_desc.groups == 0 || stage.conv2d_desc.fused_activation != 0) {
        return false;
    }
    const uint32_t groups = weights.dims[0];
    if (groups == 0 || groups != stage.conv2d_desc.groups ||
        weights.dims[1] != 1 || weights.dims[2] != 1 ||
        input.image_feature_channels != groups || output.image_feature_channels != groups) {
        return false;
    }
    if (bias && bias->dtype != output.dtype) {
        return false;
    }
    return true;
}

uint32_t mps_conv2d_descriptor_groups(const GfxMpsrtTensorAbiDesc& weights, const MpsrtRuntimeStage& stage) {
    if (weights.rank == 5 && stage.conv2d_desc.groups > 1 && conv2d_input_channels_per_group(weights) % 4u != 0u) {
        return 1u;
    }
    return stage.conv2d_desc.groups;
}

bool pack_depthwise_weights_to_mps_cnn(const GfxMpsrtTensorAbiDesc& weights,
                                       const uint8_t* source,
                                       size_t bytes,
                                       std::vector<uint8_t>& packed,
                                       std::string* log) {
    const uint64_t expected_bytes = tensor_dense_bytes(weights);
    if (!source || expected_bytes == 0 || expected_bytes != bytes) {
        return fail(log, "GFX MPSRT: MPS CNN DepthwiseConv2D cannot pack weights with invalid byte size");
    }
    if (weights.rank != 5 || weights.dims[1] != 1 || weights.dims[2] != 1) {
        return fail(log, "GFX MPSRT: MPS CNN DepthwiseConv2D expects GOIHW depthwise weights");
    }
    const auto dtype = static_cast<GfxMpsrtDType>(weights.dtype);
    const uint32_t element_bytes = gfx_mpsrt_element_size_bytes(dtype);
    if (element_bytes == 0) {
        return fail(log, "GFX MPSRT: MPS CNN DepthwiseConv2D cannot pack weights with unsupported dtype");
    }

    const uint32_t channels = weights.dims[0];
    const uint32_t kernel_h = weights.dims[3];
    const uint32_t kernel_w = weights.dims[4];
    packed.assign(static_cast<size_t>(channels) * kernel_h * kernel_w * element_bytes, 0u);
    for (uint32_t channel = 0; channel < channels; ++channel) {
        for (uint32_t kh = 0; kh < kernel_h; ++kh) {
            for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                const size_t src_index =
                    (((static_cast<size_t>(channel) * kernel_h + kh) * kernel_w + kw) * element_bytes);
                const size_t dst_index = src_index;
                std::memcpy(packed.data() + dst_index, source + src_index, element_bytes);
            }
        }
    }
    return true;
}

bool pack_conv_weights_to_mps_ohwi(const GfxMpsrtTensorAbiDesc& weights, const GfxMpsrtTensorAbiDesc& input,
                                   const GfxMpsrtTensorAbiDesc& output, const MpsrtRuntimeStage& stage,
                                   const uint8_t* source, size_t bytes, std::vector<uint8_t>& packed,
                                   std::string* log) {
    const uint64_t expected_bytes = tensor_dense_bytes(weights);
    if (!source || expected_bytes == 0 || expected_bytes != bytes) {
        return fail(log, "GFX MPSRT: MPS Conv2D cannot pack weights with invalid byte size");
    }
    const auto dtype = static_cast<GfxMpsrtDType>(weights.dtype);
    const uint32_t element_bytes = gfx_mpsrt_element_size_bytes(dtype);
    if (element_bytes == 0) {
        return fail(log, "GFX MPSRT: MPS Conv2D cannot pack weights with unsupported dtype");
    }

    if (weights.rank == 4) {
        const uint32_t output_channels = weights.dims[0];
        const uint32_t input_channels = weights.dims[1];
        const uint32_t kernel_h = weights.dims[2];
        const uint32_t kernel_w = weights.dims[3];
        const uint32_t mps_input_channels = input.image_feature_channels;
        const uint32_t mps_output_channels = output.image_feature_channels;
        if (stage.conv2d_desc.groups != 1) {
            return fail(log, "GFX MPSRT: MPS Conv2D OHWI pack supports only groups=1");
        }
        if (input_channels != input.image_feature_channels || output_channels != output.image_feature_channels) {
            return fail(log, "GFX MPSRT: MPS Conv2D weights do not match logical image channels");
        }
        packed.assign(static_cast<size_t>(mps_output_channels) * kernel_h * kernel_w * mps_input_channels *
                          element_bytes,
                      0u);
        for (uint32_t oc = 0; oc < output_channels; ++oc) {
            for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                    for (uint32_t ic = 0; ic < input_channels; ++ic) {
                        const size_t src_index =
                            (((static_cast<size_t>(oc) * input_channels + ic) * kernel_h + kh) * kernel_w + kw) *
                            element_bytes;
                        const size_t dst_index =
                            (((static_cast<size_t>(oc) * kernel_h + kh) * kernel_w + kw) * mps_input_channels + ic) *
                            element_bytes;
                        std::memcpy(packed.data() + dst_index, source + src_index, element_bytes);
                    }
                }
            }
        }
        return true;
    }
    if (weights.rank == 5) {
        const uint32_t groups = weights.dims[0];
        const uint32_t output_channels_per_group = weights.dims[1];
        const uint32_t input_channels_per_group = weights.dims[2];
        const uint32_t kernel_h = weights.dims[3];
        const uint32_t kernel_w = weights.dims[4];
        const uint32_t input_channels = input.image_feature_channels;
        const uint32_t output_channels = output.image_feature_channels;
        const uint32_t descriptor_groups = mps_conv2d_descriptor_groups(weights, stage);
        const bool dense_sparse_mps_group_workaround = descriptor_groups == 1u && groups > 1u;
        const uint32_t mps_input_channels =
            dense_sparse_mps_group_workaround ? input_channels : input_channels_per_group;
        const uint32_t mps_output_channels = output_channels;
        if (groups == 0 || stage.conv2d_desc.groups != groups || input_channels != groups * input_channels_per_group ||
            output_channels != groups * output_channels_per_group) {
            return fail(log, "GFX MPSRT: MPS GroupConv2D weights do not match "
                             "logical image channels");
        }
        packed.assign(static_cast<size_t>(mps_output_channels) * kernel_h * kernel_w * mps_input_channels *
                          element_bytes,
                      0u);
        for (uint32_t group = 0; group < groups; ++group) {
            for (uint32_t oc = 0; oc < output_channels_per_group; ++oc) {
                const uint32_t physical_oc = group * output_channels_per_group + oc;
                for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                    for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                        for (uint32_t ic = 0; ic < input_channels_per_group; ++ic) {
                            const uint32_t mps_ic =
                                dense_sparse_mps_group_workaround ? group * input_channels_per_group + ic : ic;
                            const size_t src_index = (((((static_cast<size_t>(group) * output_channels_per_group + oc) *
                                                             input_channels_per_group +
                                                         ic) *
                                                            kernel_h +
                                                        kh) *
                                                           kernel_w +
                                                      kw) *
                                                      element_bytes);
                            const size_t dst_index =
                                ((((static_cast<size_t>(physical_oc) * kernel_h + kh) * kernel_w + kw) *
                                      mps_input_channels +
                                  mps_ic) *
                                 element_bytes);
                            std::memcpy(packed.data() + dst_index, source + src_index, element_bytes);
                        }
                    }
                }
            }
        }
        return true;
    }
    return fail(log, "GFX MPSRT: MPS Conv2D weights must be OIHW or GOIHW");
}

id<MTLBuffer> new_private_buffer_with_bytes(id<MTLDevice> device,
                                            id<MTLCommandQueue> command_queue,
                                            const void* data,
                                            size_t bytes,
                                            const char* name,
                                            std::string* log) {
    if (!device || !command_queue || !data || bytes == 0) {
        (void)fail(log, std::string("GFX MPSRT: cannot upload ") + name + " with empty input");
        return nil;
    }
    id<MTLBuffer> staging = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    if (!staging) {
        (void)fail(log, std::string("GFX MPSRT: failed to create staging buffer for ") + name);
        return nil;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    if (!buffer) {
        [staging release];
        (void)fail(log, std::string("GFX MPSRT: failed to create private buffer for ") + name);
        return nil;
    }

    id<MTLCommandBuffer> command = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:buffer destinationOffset:0 size:bytes];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    [staging release];
    if ([command status] != MTLCommandBufferStatusCompleted) {
        [buffer release];
        (void)fail(log, std::string("GFX MPSRT: failed to upload ") + name + " into GPU-owned buffer");
        return nil;
    }
    return buffer;
}

bool unpack_conv_bias_to_float(const GfxMpsrtTensorAbiDesc& bias,
                               uint32_t output_channels,
                               const std::vector<uint8_t>& host_bytes,
                               std::vector<float>& out,
                               std::string* log) {
    out.clear();
    if (output_channels == 0) {
        return fail(log, "GFX MPSRT: MPS Conv2D bias output channel count is zero");
    }
    const uint64_t expected_bytes = tensor_dense_bytes(bias);
    if (expected_bytes == 0 || expected_bytes != host_bytes.size()) {
        return fail(log, "GFX MPSRT: MPS Conv2D bias byte size mismatch");
    }
    uint64_t elements = 1;
    for (uint32_t i = 0; i < bias.rank; ++i) {
        elements *= bias.dims[i];
    }
    if (elements != output_channels) {
        return fail(log, "GFX MPSRT: MPS Conv2D bias must be a channel vector");
    }

    out.resize(output_channels);
    const auto dtype = static_cast<GfxMpsrtDType>(bias.dtype);
    if (dtype == GfxMpsrtDType::F32) {
        std::memcpy(out.data(), host_bytes.data(), out.size() * sizeof(float));
        return true;
    }
    if (dtype == GfxMpsrtDType::F16) {
        const auto* values = reinterpret_cast<const ov::float16*>(host_bytes.data());
        for (uint32_t i = 0; i < output_channels; ++i) {
            out[i] = static_cast<float>(values[i]);
        }
        return true;
    }
    return fail(log, "GFX MPSRT: MPS Conv2D bias dtype is unsupported");
}

bool make_mps_conv2d_cache_key(const MpsrtRuntimeStage& stage, const GfxMpsrtTensorAbiDesc& input,
                               const GfxMpsrtTensorAbiDesc& weights, const GfxMpsrtTensorAbiDesc& output,
                               const std::string& weights_const_key,
                               const std::string& bias_const_key,
                               std::string& key, std::string* log) {
    if (input.dtype != weights.dtype || weights.dtype != output.dtype) {
        return fail(log, "GFX MPSRT: MPS Conv2D input, weights and output dtype "
                         "must match for this ABI");
    }
    if (stage.conv2d_desc.groups == 0) {
        return fail(log, "GFX MPSRT: MPS Conv2D group count is zero");
    }
    if (input.image_feature_channels % stage.conv2d_desc.groups != 0 ||
        output.image_feature_channels % stage.conv2d_desc.groups != 0) {
        return fail(log, "GFX MPSRT: MPS Conv2D channel counts must be divisible by groups");
    }
    if (conv2d_input_channels_per_group(weights) != input.image_feature_channels / stage.conv2d_desc.groups ||
        conv2d_output_channels(weights) != output.image_feature_channels) {
        return fail(log, "GFX MPSRT: MPS Conv2D weights do not match image channel contract");
    }
    const uint32_t descriptor_groups = mps_conv2d_descriptor_groups(weights, stage);
    std::ostringstream stream;
    stream << "mps_conv2d|" << static_cast<uint32_t>(stage.kind) << '|' << input.image_feature_channels << '|'
           << output.image_feature_channels << '|' << conv2d_kernel_width(weights) << '|'
           << conv2d_kernel_height(weights) << '|' << stage.conv2d_desc.groups << '|' << descriptor_groups << '|'
           << stage.conv2d_desc.strides[0] << '|' << stage.conv2d_desc.strides[1] << '|'
           << stage.conv2d_desc.dilations[0] << '|'
           << stage.conv2d_desc.dilations[1] << '|' << stage.conv2d_desc.pads[0] << '|' << stage.conv2d_desc.pads[1]
           << '|' << stage.conv2d_desc.pads[2] << '|' << stage.conv2d_desc.pads[3] << '|'
           << stage.conv2d_desc.fused_activation << '|' << output.dtype << '|'
           << weights_const_key << '|' << bias_const_key;
    key = stream.str();
    return true;
}

bool make_mps_pool2d_cache_key(const MpsrtRuntimeStage& stage, const GfxMpsrtTensorAbiDesc& input,
                               const GfxMpsrtTensorAbiDesc& output, std::string& key, std::string* log) {
    if (input.dtype != output.dtype) {
        return fail(log, "GFX MPSRT: MPS Pool2D input and output dtype must match");
    }
    if (stage.pool2d_desc.kernel[0] == 0 || stage.pool2d_desc.kernel[1] == 0 || stage.pool2d_desc.strides[0] == 0 ||
        stage.pool2d_desc.strides[1] == 0) {
        return fail(log, "GFX MPSRT: MPS Pool2D kernel and stride must be nonzero");
    }
    if (stage.pool2d_desc.dilations[0] != 1 || stage.pool2d_desc.dilations[1] != 1) {
        return fail(log, "GFX MPSRT: MPS Pool2D dilation is not supported by this ABI");
    }
    if (input.image_batch != output.image_batch || input.image_feature_channels != output.image_feature_channels) {
        return fail(log, "GFX MPSRT: MPS Pool2D input/output image channel or batch mismatch");
    }
    std::ostringstream stream;
    stream << "mps_pool2d|" << stage.pool2d_desc.is_avg << '|' << stage.pool2d_desc.kernel[0] << '|'
           << stage.pool2d_desc.kernel[1] << '|' << stage.pool2d_desc.strides[0] << '|' << stage.pool2d_desc.strides[1]
           << '|' << stage.pool2d_desc.pads[0] << '|' << stage.pool2d_desc.pads[1] << '|' << stage.pool2d_desc.pads[2]
           << '|' << stage.pool2d_desc.pads[3] << '|' << stage.pool2d_desc.exclude_pad << '|'
           << input.image_feature_channels << '|' << input.dtype;
    key = stream.str();
    return true;
}

bool make_mps_resize2d_cache_key(const MpsrtRuntimeStage& stage, const GfxMpsrtTensorAbiDesc& input,
                                 const GfxMpsrtTensorAbiDesc& output, std::string& key, std::string* log) {
    if (input.dtype != output.dtype) {
        return fail(log, "GFX MPSRT: MPS Resize2D input and output dtype must match");
    }
    if (stage.resize2d_desc.nearest != 0) {
        return fail(log, "GFX MPSRT: MPS Resize2D nearest mode is not supported by this ABI");
    }
    if (input.image_batch != output.image_batch || input.image_feature_channels != output.image_feature_channels) {
        return fail(log, "GFX MPSRT: MPS Resize2D input/output image channel or batch mismatch");
    }
    std::ostringstream stream;
    stream << "mps_resize2d|bilinear" << '|' << input.image_width << '|' << input.image_height << '|'
           << output.image_width << '|' << output.image_height << '|' << input.image_feature_channels << '|'
           << input.image_batch << '|' << input.dtype << '|' << stage.resize2d_desc.align_corners << '|'
           << stage.resize2d_desc.half_pixel_centers;
    key = stream.str();
    return true;
}

uint32_t matrix_count_or_one(const GfxMpsrtTensorAbiDesc& desc);

bool make_mps_softmax_cache_key(const MpsrtRuntimeStage& stage, const GfxMpsrtTensorAbiDesc& input,
                                const GfxMpsrtTensorAbiDesc& output, std::string& key, std::string* log) {
    if (input.dtype != output.dtype) {
        return fail(log, "GFX MPSRT: MPS Softmax input and output dtype must match");
    }
    if (input.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix) ||
        output.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
        return fail(log, "GFX MPSRT: MPS Softmax requires matrix storage");
    }
    if (input.matrix_rows == 0 || input.matrix_columns == 0 || input.matrix_row_bytes == 0 || output.matrix_rows == 0 ||
        output.matrix_columns == 0 || output.matrix_row_bytes == 0) {
        return fail(log, "GFX MPSRT: MPS Softmax matrix descriptor is incomplete");
    }
    if (input.matrix_rows != output.matrix_rows || input.matrix_columns != output.matrix_columns ||
        matrix_count_or_one(input) != matrix_count_or_one(output)) {
        return fail(log, "GFX MPSRT: MPS Softmax input/output matrix shape mismatch");
    }
    if (stage.softmax_desc.log_softmax != 0) {
        return fail(log, "GFX MPSRT: MPS Softmax runtime does not implement LogSoftmax");
    }
    std::ostringstream stream;
    stream << "mps_softmax|" << input.matrix_rows << '|' << input.matrix_columns << '|' << matrix_count_or_one(input)
           << '|' << input.dtype << "|axis" << stage.softmax_desc.axis;
    key = stream.str();
    return true;
}

bool make_mps_topk_cache_key(const MpsrtRuntimeStage& stage, const GfxMpsrtTensorAbiDesc& input,
                             const GfxMpsrtTensorAbiDesc& values_output, const GfxMpsrtTensorAbiDesc& indices_output,
                             std::string& key, std::string* log) {
    if (stage.topk_desc.mode_max == 0) {
        return fail(log, "GFX MPSRT: MPS TopK supports MAX mode only");
    }
    if (stage.topk_desc.k == 0) {
        return fail(log, "GFX MPSRT: MPS TopK k must be positive");
    }
    if (input.dtype != values_output.dtype) {
        return fail(log, "GFX MPSRT: MPS TopK input and values output dtype must match");
    }
    if (input.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix) ||
        values_output.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix) ||
        indices_output.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
        return fail(log, "GFX MPSRT: MPS TopK requires matrix storage");
    }
    if (input.matrix_rows == 0 || input.matrix_columns == 0 || input.matrix_row_bytes == 0 ||
        values_output.matrix_rows == 0 || values_output.matrix_columns == 0 || values_output.matrix_row_bytes == 0 ||
        indices_output.matrix_rows == 0 || indices_output.matrix_columns == 0 || indices_output.matrix_row_bytes == 0) {
        return fail(log, "GFX MPSRT: MPS TopK matrix descriptor is incomplete");
    }
    if (input.matrix_rows != values_output.matrix_rows || input.matrix_rows != indices_output.matrix_rows ||
        values_output.matrix_columns != stage.topk_desc.k || indices_output.matrix_columns != stage.topk_desc.k ||
        matrix_count_or_one(input) != matrix_count_or_one(values_output) ||
        matrix_count_or_one(input) != matrix_count_or_one(indices_output)) {
        return fail(log, "GFX MPSRT: MPS TopK input/output matrix shape mismatch");
    }
    if (indices_output.dtype != static_cast<uint32_t>(GfxMpsrtDType::I32) &&
        indices_output.dtype != static_cast<uint32_t>(GfxMpsrtDType::U32) &&
        indices_output.dtype != static_cast<uint32_t>(GfxMpsrtDType::I64)) {
        return fail(log, "GFX MPSRT: MPS TopK index output must be i32/u32/i64");
    }
    const bool uses_graph_topk = stage.topk_desc.k > 16;
    if (stage.topk_desc.k > input.matrix_columns) {
        return fail(log, "GFX MPSRT: MPS TopK k exceeds source columns");
    }
    if (uses_graph_topk && stage.topk_desc.sort_type == 2u) {
        return fail(log, "GFX MPSRT: MPSGraph TopK does not implement SORT_INDICES");
    }
    if (uses_graph_topk && mps_data_type_from_gfx(input.dtype) == MPSDataTypeInvalid) {
        return fail(log, "GFX MPSRT: MPSGraph TopK dtype is unsupported");
    }
    std::ostringstream stream;
    stream << "mps_topk|" << (uses_graph_topk ? "graph" : "matrix") << '|' << input.matrix_rows << '|'
           << input.matrix_columns << '|' << matrix_count_or_one(input)
           << '|' << input.dtype << "|axis" << stage.topk_desc.axis << "|k" << stage.topk_desc.k << "|sort"
           << stage.topk_desc.sort_type;
    key = stream.str();
    return true;
}

bool is_dense_ndarray_graph_tensor_desc(const GfxMpsrtTensorAbiDesc& desc) {
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::NDArray)) {
        return false;
    }
    const auto dtype = static_cast<GfxMpsrtDType>(desc.dtype);
    const uint32_t element_bytes = gfx_mpsrt_element_size_bytes(dtype);
    if (element_bytes == 0 || desc.byte_offset != 0 || desc.rank == 0 || desc.rank > 8) {
        return false;
    }
    uint64_t dense_elements = 1;
    for (uint32_t i = 0; i < desc.rank; ++i) {
        if (desc.dims[i] == 0) {
            return false;
        }
        dense_elements *= desc.dims[i];
    }
    return desc.byte_length == dense_elements * element_bytes;
}

bool make_mps_sdpa_cache_key(const MpsrtRuntimeStage& stage,
                             const GfxMpsrtTensorAbiDesc& query,
                             const GfxMpsrtTensorAbiDesc& key_desc,
                             const GfxMpsrtTensorAbiDesc& value,
                             const GfxMpsrtTensorAbiDesc& output,
                             std::string& key, std::string* log) {
    if (stage.sdpa_desc.has_mask != 0 || stage.sdpa_desc.causal != 0) {
        return fail(log, "GFX MPSRT: MPSGraph SDPA supports only unmasked non-causal stages in this ABI");
    }
    if (query.dtype != key_desc.dtype || query.dtype != value.dtype || query.dtype != output.dtype) {
        return fail(log, "GFX MPSRT: MPSGraph SDPA Q/K/V/output dtype mismatch");
    }
    if (mps_data_type_from_gfx(output.dtype) == MPSDataTypeInvalid) {
        return fail(log, "GFX MPSRT: MPSGraph SDPA dtype is unsupported");
    }
    if (query.rank != 4 || key_desc.rank != 4 || value.rank != 4 || output.rank != 4) {
        return fail(log, "GFX MPSRT: MPSGraph SDPA expects rank-4 Q/K/V/output tensors");
    }
    if (!is_dense_ndarray_graph_tensor_desc(query) ||
        !is_dense_ndarray_graph_tensor_desc(key_desc) ||
        !is_dense_ndarray_graph_tensor_desc(value) ||
        !is_dense_ndarray_graph_tensor_desc(output)) {
        return fail(log, "GFX MPSRT: MPSGraph SDPA requires dense zero-offset NDArray tensors");
    }
    if (stage.sdpa_desc.layout == GfxMpsrtSdpaLayoutNativeBHND) {
        if (query.dims[0] != key_desc.dims[0] || query.dims[0] != value.dims[0] ||
            query.dims[1] != key_desc.dims[1] || query.dims[1] != value.dims[1] ||
            query.dims[3] != key_desc.dims[3] || query.dims[3] != value.dims[3] ||
            key_desc.dims[2] != value.dims[2] ||
            output.dims[0] != query.dims[0] || output.dims[1] != query.dims[1] ||
            output.dims[2] != query.dims[2] || output.dims[3] != value.dims[3]) {
            return fail(log, "GFX MPSRT: MPSGraph SDPA native Q/K/V/output shape mismatch");
        }
    } else if (stage.sdpa_desc.layout == GfxMpsrtSdpaLayoutTransposedBHDN) {
        if (query.dims[0] != key_desc.dims[0] || query.dims[0] != value.dims[0] ||
            query.dims[1] != key_desc.dims[1] || query.dims[1] != value.dims[1] ||
            query.dims[2] != key_desc.dims[2] || key_desc.dims[3] != value.dims[3] ||
            output.dims[0] != query.dims[0] || output.dims[1] != query.dims[1] ||
            output.dims[2] != value.dims[2] || output.dims[3] != query.dims[3]) {
            return fail(log, "GFX MPSRT: MPSGraph SDPA transposed Q/K/V/output shape mismatch");
        }
    } else {
        return fail(log, "GFX MPSRT: MPSGraph SDPA layout is unsupported");
    }
    std::ostringstream stream;
    stream << "mps_sdpa|graph|layout" << stage.sdpa_desc.layout << '|' << query.dtype
           << "|scale" << stage.sdpa_desc.scale;
    for (uint32_t i = 0; i < 4; ++i) {
        stream << '|' << query.dims[i];
    }
    for (uint32_t i = 0; i < 4; ++i) {
        stream << '|' << key_desc.dims[i];
    }
    for (uint32_t i = 0; i < 4; ++i) {
        stream << '|' << value.dims[i];
    }
    key = stream.str();
    return true;
}

uint32_t matrix_count_or_one(const GfxMpsrtTensorAbiDesc& desc) {
    return desc.matrix_count == 0 ? 1 : desc.matrix_count;
}

void configure_mps_graph_full_precision_gpu_compile(MPSGraphCompilationDescriptor* descriptor) {
    if (!descriptor) {
        return;
    }
    descriptor.waitForCompilationCompletion = YES;
    if (@available(macOS 12.3, iOS 15.4, tvOS 15.4, *)) {
        descriptor.optimizationLevel = MPSGraphOptimizationLevel0;
    }
    if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
        descriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathNone;
    }
}

bool matrix_count_can_feed_output(uint32_t input_count, uint32_t output_count) {
    return input_count == output_count || input_count == 1;
}

bool validate_matrix_desc(const GfxMpsrtTensorAbiDesc& desc, const char* name, std::string* log) {
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
        return fail(log, std::string("GFX MPSRT: MPS GEMM ") + name + " tensor is not matrix storage");
    }
    if (desc.matrix_rows == 0 || desc.matrix_columns == 0 || desc.matrix_row_bytes == 0) {
        return fail(log, std::string("GFX MPSRT: MPS GEMM ") + name + " matrix descriptor is incomplete");
    }
    if (desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32) &&
        desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16)) {
        return fail(log, std::string("GFX MPSRT: MPS GEMM ") + name + " dtype is unsupported");
    }
    return true;
}

bool is_mps_conv2d_stage(GfxMpsrtStageKind kind) {
    return kind == GfxMpsrtStageKind::MPSConv2D || kind == GfxMpsrtStageKind::MPSGroupConv2D;
}

bool is_first_class_mps_stage(GfxMpsrtStageKind kind) {
    switch (kind) {
    case GfxMpsrtStageKind::MPSConv2D:
    case GfxMpsrtStageKind::MPSGroupConv2D:
    case GfxMpsrtStageKind::MPSPool2D:
    case GfxMpsrtStageKind::MPSResize2D:
    case GfxMpsrtStageKind::MPSGemm:
    case GfxMpsrtStageKind::MPSSoftmax:
    case GfxMpsrtStageKind::MPSTopK:
    case GfxMpsrtStageKind::MPSSdpa:
        return true;
    case GfxMpsrtStageKind::MSLDispatch:
    case GfxMpsrtStageKind::Alias:
    case GfxMpsrtStageKind::Unknown:
    default:
        return false;
    }
}

bool validate_image_desc(const GfxMpsrtTensorAbiDesc& desc, const char* stage_name, const char* tensor_name,
                         std::string* log) {
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Image)) {
        return fail(log, std::string("GFX MPSRT: ") + stage_name + " " + tensor_name + " tensor is not image storage");
    }
    if (desc.rank != 4 || desc.image_width == 0 || desc.image_height == 0 || desc.image_feature_channels == 0 ||
        desc.image_batch == 0) {
        return fail(log,
                    std::string("GFX MPSRT: ") + stage_name + " " + tensor_name + " image descriptor is incomplete");
    }
    if (desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32) &&
        desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16)) {
        return fail(log, std::string("GFX MPSRT: ") + stage_name + " " + tensor_name + " dtype is unsupported");
    }
    return true;
}

bool validate_conv_weights_desc(const GfxMpsrtTensorAbiDesc& desc, const MpsrtRuntimeStage& stage, std::string* log) {
    const char* stage_name = gfx_mpsrt_stage_kind_name(stage.kind);
    if (desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32) &&
        desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16)) {
        return fail(log, std::string("GFX MPSRT: ") + stage_name + " weights dtype is unsupported");
    }
    if (stage.kind == GfxMpsrtStageKind::MPSConv2D && desc.rank != 4) {
        return fail(log, "GFX MPSRT: MPS Conv2D weights must be OIHW rank-4");
    }
    if (stage.kind == GfxMpsrtStageKind::MPSGroupConv2D && desc.rank != 5) {
        return fail(log, "GFX MPSRT: MPS GroupConv2D weights must be GOIHW rank-5");
    }
    if (stage.kind == GfxMpsrtStageKind::MPSGroupConv2D && desc.dims[0] != stage.conv2d_desc.groups) {
        return fail(log, "GFX MPSRT: MPS GroupConv2D weights group count mismatch");
    }
    return true;
}

bool is_dense_matrix_graph_tensor_desc(const GfxMpsrtTensorAbiDesc& desc) {
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
        return false;
    }
    const auto dtype = static_cast<GfxMpsrtDType>(desc.dtype);
    const uint32_t element_bytes = gfx_mpsrt_element_size_bytes(dtype);
    if (element_bytes == 0 || desc.byte_offset != 0 || desc.matrix_rows == 0 ||
        desc.matrix_columns == 0 || desc.matrix_row_bytes == 0 ||
        desc.rank < 2) {
        return false;
    }
    const uint64_t dense_row_bytes =
        static_cast<uint64_t>(desc.matrix_columns) * element_bytes;
    if (desc.matrix_row_bytes != dense_row_bytes) {
        return false;
    }
    const uint64_t dense_matrix_bytes =
        dense_row_bytes * static_cast<uint64_t>(desc.matrix_rows);
    const uint64_t dense_total_bytes =
        dense_matrix_bytes * static_cast<uint64_t>(matrix_count_or_one(desc));
    return desc.byte_length == dense_total_bytes;
}

bool should_use_mps_graph_gemm(const MpsrtRuntimeStage& stage,
                               const GfxMpsrtTensorAbiDesc& lhs,
                               const GfxMpsrtTensorAbiDesc& rhs,
                               const GfxMpsrtTensorAbiDesc& output) {
    if (@available(macOS 11.0, iOS 14.0, tvOS 14.0, *)) {
        if (stage.gemm_desc.alpha != 1.0f || stage.gemm_desc.beta != 0.0f) {
            return false;
        }
        if (stage.gemm_desc.transpose_lhs == 0 &&
            stage.gemm_desc.transpose_rhs == 0) {
            return false;
        }
        if (matrix_count_or_one(output) <= 1) {
            return false;
        }
        if (lhs.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32) ||
            rhs.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32) ||
            output.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32)) {
            return false;
        }
        return is_dense_matrix_graph_tensor_desc(lhs) &&
               is_dense_matrix_graph_tensor_desc(rhs) &&
               is_dense_matrix_graph_tensor_desc(output);
    }
    return false;
}

bool make_mps_gemm_cache_key(const MpsrtRuntimeStage& stage, const GfxMpsrtTensorAbiDesc& lhs,
                             const GfxMpsrtTensorAbiDesc& rhs, const GfxMpsrtTensorAbiDesc& output,
                             uint32_t result_rows, uint32_t result_columns, uint32_t interior_columns,
                             bool uses_graph_gemm, std::string& key, std::string* log) {
    if (lhs.dtype != rhs.dtype || lhs.dtype != output.dtype) {
        return fail(log, "GFX MPSRT: MPS GEMM matrix dtype mismatch");
    }
    const uint32_t lhs_count = matrix_count_or_one(lhs);
    const uint32_t rhs_count = matrix_count_or_one(rhs);
    const uint32_t output_count = matrix_count_or_one(output);
    if (!matrix_count_can_feed_output(lhs_count, output_count) ||
        !matrix_count_can_feed_output(rhs_count, output_count)) {
        return fail(log, "GFX MPSRT: MPS GEMM batch matrix counts must be either 1 "
                         "or output matrix count");
    }

    const uint32_t lhs_rows = stage.gemm_desc.transpose_lhs ? lhs.matrix_columns : lhs.matrix_rows;
    const uint32_t lhs_columns = stage.gemm_desc.transpose_lhs ? lhs.matrix_rows : lhs.matrix_columns;
    const uint32_t rhs_rows = stage.gemm_desc.transpose_rhs ? rhs.matrix_columns : rhs.matrix_rows;
    const uint32_t rhs_columns = stage.gemm_desc.transpose_rhs ? rhs.matrix_rows : rhs.matrix_columns;
    if (lhs_rows != result_rows || rhs_columns != result_columns || lhs_columns != interior_columns ||
        rhs_rows != interior_columns) {
        return fail(log, "GFX MPSRT: MPS GEMM matrix dimensions do not match "
                         "transpose contract");
    }

    std::ostringstream stream;
    stream << "mps_gemm|" << (uses_graph_gemm ? "graph" : "matrix") << '|'
           << stage.gemm_desc.transpose_lhs << '|' << stage.gemm_desc.transpose_rhs << '|'
           << std::setprecision(9) << stage.gemm_desc.alpha << '|' << std::setprecision(9) << stage.gemm_desc.beta
           << '|' << result_rows << '|' << result_columns << '|' << interior_columns << '|' << lhs.dtype << '|'
           << lhs_count << '|' << rhs_count << '|' << output_count;
    key = stream.str();
    return true;
}

std::string unsupported_mps_stage_message(const MpsrtRuntimeStage& stage) {
    std::ostringstream stream;
    stream << "GFX MPSRT: " << gfx_mpsrt_stage_kind_name(stage.kind)
           << " is a first-class AppleMps vendor stage, but its runtime encoder "
              "is not implemented yet";
    return stream.str();
}

} // namespace

struct MpsrtContext::PipelineCacheEntry {
    std::string key;
    id<MTLComputePipelineState> pipeline = nil;
};

struct MpsrtContext::MpsGemmCacheEntry {
    std::string key;
    id kernel = nil;
    id graph_lhs_tensor = nil;
    id graph_rhs_tensor = nil;
    id graph_output_tensor = nil;
    id graph_executable = nil;
    bool uses_mps_graph_gemm = false;
};

struct MpsrtContext::MpsConv2DCacheEntry {
    std::string key;
    id kernel = nil;
    id data_source = nil;
};

struct MpsrtContext::MpsPool2DCacheEntry {
    std::string key;
    id kernel = nil;
};

struct MpsrtContext::MpsResize2DCacheEntry {
    std::string key;
    id kernel = nil;
};

struct MpsrtContext::MpsSoftmaxCacheEntry {
    std::string key;
    id kernel = nil;
};

struct MpsrtContext::MpsTopKCacheEntry {
    std::string key;
    id kernel = nil;
    id graph_input_tensor = nil;
    id graph_values_tensor = nil;
    id graph_indices_tensor = nil;
    id graph_executable = nil;
    bool uses_mps_graph_topk = false;
};

struct MpsrtContext::MpsSdpaCacheEntry {
    std::string key;
    id kernel = nil;
    id graph_query_tensor = nil;
    id graph_key_tensor = nil;
    id graph_value_tensor = nil;
    id graph_output_tensor = nil;
    id graph_executable = nil;
};

struct MpsrtContext::ConstTensorCacheEntry {
    std::string key;
    GfxMpsrtValue value = 0;
    GfxMpsrtTensorAbiDesc desc{};
    size_t bytes = 0;
    id<MTLBuffer> buffer = nil;
    std::vector<uint8_t> host_bytes;
};

MpsrtPreparedModel::~MpsrtPreparedModel() {
    release_owned_resources();
}

MpsrtPreparedModel::MpsrtPreparedModel(MpsrtPreparedModel&& other) noexcept {
    *this = std::move(other);
}

MpsrtPreparedModel& MpsrtPreparedModel::operator=(MpsrtPreparedModel&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    release_owned_resources();

    resource_heap = other.resource_heap;
    other.resource_heap = nil;
    resource_heap_size = other.resource_heap_size;
    resource_heap_unaliased_size = other.resource_heap_unaliased_size;
    resource_heap_aliasable_size = other.resource_heap_aliasable_size;
    resource_heap_alias_reuse_count = other.resource_heap_alias_reuse_count;
    transient_buffer_resource_count = other.transient_buffer_resource_count;
    transient_image_resource_count = other.transient_image_resource_count;
    image_bridge_resource_count = other.image_bridge_resource_count;
    resources = std::move(other.resources);
    image_bridge_resources = std::move(other.image_bridge_resources);
    msl_dispatches = std::move(other.msl_dispatches);
    mps_gemm_stages = std::move(other.mps_gemm_stages);
    mps_conv2d_stages = std::move(other.mps_conv2d_stages);
    mps_pool2d_stages = std::move(other.mps_pool2d_stages);
    mps_resize2d_stages = std::move(other.mps_resize2d_stages);
    mps_softmax_stages = std::move(other.mps_softmax_stages);
    mps_topk_stages = std::move(other.mps_topk_stages);
    mps_sdpa_stages = std::move(other.mps_sdpa_stages);
    skipped_non_msl_stages = other.skipped_non_msl_stages;

    other.resources.clear();
    other.image_bridge_resources.clear();
    other.msl_dispatches.clear();
    other.mps_gemm_stages.clear();
    other.mps_conv2d_stages.clear();
    other.mps_pool2d_stages.clear();
    other.mps_resize2d_stages.clear();
    other.mps_softmax_stages.clear();
    other.mps_topk_stages.clear();
    other.mps_sdpa_stages.clear();
    other.resource_heap_size = 0;
    other.resource_heap_unaliased_size = 0;
    other.resource_heap_aliasable_size = 0;
    other.resource_heap_alias_reuse_count = 0;
    other.transient_buffer_resource_count = 0;
    other.transient_image_resource_count = 0;
    other.image_bridge_resource_count = 0;
    other.skipped_non_msl_stages = 0;
    return *this;
}

void MpsrtPreparedModel::release_owned_resources() {
    for (auto& resource : resources) {
        if (resource.lifetime != MpsrtRuntimeResourceLifetime::Transient) {
            continue;
        }
        [resource.buffer release];
        [resource.texture release];
        resource.buffer = nil;
        resource.texture = nil;
    }
    for (auto& resource : image_bridge_resources) {
        [resource.texture release];
        resource.texture = nil;
    }
    [resource_heap release];
    resource_heap = nil;
    resource_heap_size = 0;
    resource_heap_unaliased_size = 0;
    resource_heap_aliasable_size = 0;
    resource_heap_alias_reuse_count = 0;
    transient_buffer_resource_count = 0;
    transient_image_resource_count = 0;
    image_bridge_resource_count = 0;
    resources.clear();
    image_bridge_resources.clear();
    msl_dispatches.clear();
    mps_gemm_stages.clear();
    mps_conv2d_stages.clear();
    mps_pool2d_stages.clear();
    mps_resize2d_stages.clear();
    mps_softmax_stages.clear();
    mps_topk_stages.clear();
    mps_sdpa_stages.clear();
    skipped_non_msl_stages = 0;
}

MpsrtContext::MpsrtContext(id<MTLDevice> device) : m_device([device retain]) {
    OPENVINO_ASSERT(m_device, "GFX MPSRT: Metal device is null");
    m_command_queue = [m_device newCommandQueue];
    OPENVINO_ASSERT(m_command_queue, "GFX MPSRT: failed to create Metal command queue");

    if (@available(macOS 11.0, iOS 14.0, *)) {
        NSError* error = nil;
        MTLBinaryArchiveDescriptor* descriptor = [[MTLBinaryArchiveDescriptor alloc] init];
        m_binary_archive = [m_device newBinaryArchiveWithDescriptor:descriptor error:&error];
        [descriptor release];
        if (!m_binary_archive) {
            increment_compile_counter("mpsrt_binary_archive_create_failed_count");
        }
    }
}

MpsrtContext::~MpsrtContext() {
    for (auto& entry : m_pipeline_cache) {
        [entry.pipeline release];
        entry.pipeline = nil;
    }
    for (auto& entry : m_mps_gemm_cache) {
        [entry.kernel release];
        [entry.graph_lhs_tensor release];
        [entry.graph_rhs_tensor release];
        [entry.graph_output_tensor release];
        [entry.graph_executable release];
        entry.kernel = nil;
        entry.graph_lhs_tensor = nil;
        entry.graph_rhs_tensor = nil;
        entry.graph_output_tensor = nil;
        entry.graph_executable = nil;
    }
    for (auto& entry : m_mps_conv2d_cache) {
        [entry.kernel release];
        [entry.data_source release];
        entry.kernel = nil;
        entry.data_source = nil;
    }
    for (auto& entry : m_mps_pool2d_cache) {
        [entry.kernel release];
        entry.kernel = nil;
    }
    for (auto& entry : m_mps_resize2d_cache) {
        [entry.kernel release];
        entry.kernel = nil;
    }
    for (auto& entry : m_mps_softmax_cache) {
        [entry.kernel release];
        entry.kernel = nil;
    }
    for (auto& entry : m_mps_topk_cache) {
        [entry.kernel release];
        [entry.graph_input_tensor release];
        [entry.graph_values_tensor release];
        [entry.graph_indices_tensor release];
        [entry.graph_executable release];
        entry.kernel = nil;
        entry.graph_input_tensor = nil;
        entry.graph_values_tensor = nil;
        entry.graph_indices_tensor = nil;
        entry.graph_executable = nil;
    }
    for (auto& entry : m_mps_sdpa_cache) {
        [entry.kernel release];
        [entry.graph_query_tensor release];
        [entry.graph_key_tensor release];
        [entry.graph_value_tensor release];
        [entry.graph_output_tensor release];
        [entry.graph_executable release];
        entry.kernel = nil;
        entry.graph_query_tensor = nil;
        entry.graph_key_tensor = nil;
        entry.graph_value_tensor = nil;
        entry.graph_output_tensor = nil;
        entry.graph_executable = nil;
    }
    for (auto& entry : m_const_tensor_cache) {
        [entry.buffer release];
        entry.buffer = nil;
    }
    [m_binary_archive release];
    [m_command_queue release];
    [m_device release];
}

size_t MpsrtContext::pipeline_cache_size() const {
    return m_pipeline_cache.size();
}

bool MpsrtContext::register_const_tensor_data(GfxMpsrtValue value, const GfxMpsrtTensorAbiDesc& desc, const void* data,
                                              size_t bytes, std::string* log) {
    if (value == 0 && (desc.flags & GfxMpsrtTensorFlagConst) == 0) {
        return fail(log, "GFX MPSRT: const tensor value must be nonzero unless "
                         "descriptor is explicitly const");
    }
    if ((desc.flags & GfxMpsrtTensorFlagConst) == 0) {
        return fail(log, "GFX MPSRT: registered tensor descriptor is not marked const");
    }
    if (!data || bytes == 0) {
        return fail(log, "GFX MPSRT: const tensor data is empty");
    }
    const uint64_t expected_bytes = tensor_dense_bytes(desc);
    if (expected_bytes == 0 || expected_bytes != bytes) {
        std::ostringstream stream;
        stream << "GFX MPSRT: const tensor byte size mismatch for value " << value << " expected " << expected_bytes
               << " got " << bytes;
        return fail(log, stream.str());
    }

    const uint64_t data_hash = gfx_hash_bytes(data, bytes);
    const std::string key = make_const_tensor_cache_key(value, desc, bytes, data_hash);
    for (const auto& entry : m_const_tensor_cache) {
        if (entry.value == value) {
            if (entry.key != key) {
                return fail(log, "GFX MPSRT: const tensor value was already registered "
                                 "with a different descriptor");
            }
            ++m_const_tensor_cache_hits;
            increment_compile_counter("mpsrt_const_tensor_cache_hit_count");
            return true;
        }
    }

    ++m_const_tensor_cache_misses;
    increment_compile_counter("mpsrt_const_tensor_cache_miss_count");

    id<MTLBuffer> staging = [m_device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    if (!staging) {
        return fail(log, "GFX MPSRT: failed to create staging buffer for const tensor");
    }
    id<MTLBuffer> buffer = [m_device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    if (!buffer) {
        [staging release];
        return fail(log, "GFX MPSRT: failed to create private buffer for const tensor");
    }

    id<MTLCommandBuffer> command = [m_command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:buffer destinationOffset:0 size:bytes];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    [staging release];
    if ([command status] != MTLCommandBufferStatusCompleted) {
        [buffer release];
        return fail(log, "GFX MPSRT: failed to upload const tensor into GPU-owned const pack");
    }

    std::vector<uint8_t> host_bytes(bytes);
    std::memcpy(host_bytes.data(), data, bytes);
    m_const_tensor_cache.push_back({key, value, desc, bytes, buffer, std::move(host_bytes)});
    increment_compile_counter("mpsrt_const_tensor_upload_count");
    return true;
}

bool MpsrtContext::has_const_tensor(GfxMpsrtValue value) const {
    for (const auto& entry : m_const_tensor_cache) {
        if (entry.value == value) {
            return true;
        }
    }
    return false;
}

bool MpsrtContext::prepare_model_resources(const MpsrtModel& model, MpsrtPreparedModel& out, std::string* log) const {
    out.release_owned_resources();
    out.resources.clear();
    out.resource_heap = nil;
    out.resource_heap_size = 0;
    out.resource_heap_unaliased_size = 0;
    out.resource_heap_aliasable_size = 0;
    out.resource_heap_alias_reuse_count = 0;
    out.transient_buffer_resource_count = 0;
    out.transient_image_resource_count = 0;
    out.image_bridge_resource_count = 0;
    out.image_bridge_resources.clear();
    if (model.resources.empty() && !model.tensors.empty()) {
        return fail(log, "GFX MPSRT: runtime resource table is required for model preparation");
    }
    std::vector<PreparedResourceHeapPlanEntry> heap_plan_entries;
    std::vector<PreparedImageBridgeHeapPlanEntry> image_bridge_heap_plan_entries;
    if (!plan_prepared_resource_heap(m_device, model, out.resource_heap, out.resource_heap_size,
                                     out.resource_heap_unaliased_size, out.resource_heap_aliasable_size,
                                     out.transient_buffer_resource_count, out.transient_image_resource_count,
                                     heap_plan_entries, image_bridge_heap_plan_entries, log)) {
        return false;
    }
    out.image_bridge_resource_count = image_bridge_heap_plan_entries.size();
    out.resources.reserve(model.resources.size());
    for (const auto& resource : model.resources) {
        MpsrtPreparedResource prepared{};
        prepared.resource_index = resource.resource_index;
        prepared.role = resource.role;
        prepared.lifetime = resource.lifetime;
        prepared.has_tensor_value = resource.has_tensor_value;
        prepared.value = resource.value;
        prepared.tensor_desc = resource.tensor_desc;
        if (const auto* heap_entry = find_heap_plan_entry(heap_plan_entries, resource.resource_index)) {
            prepared.heap_allocation_size = heap_entry->allocation_size;
            prepared.heap_alignment = heap_entry->alignment;
            prepared.first_stage_index = heap_entry->first_stage_index;
            prepared.last_stage_index = heap_entry->last_stage_index;
        }

        switch (resource.lifetime) {
        case MpsrtRuntimeResourceLifetime::External:
            if (resource.has_tensor_value) {
                prepared.byte_length = static_cast<size_t>(resource.tensor_desc.byte_length);
                prepared.offset = static_cast<size_t>(resource.tensor_desc.byte_offset);
            }
            break;
        case MpsrtRuntimeResourceLifetime::Transient: {
            if (!resource.has_tensor_value) {
                return fail(log, "GFX MPSRT: transient resource is not a tensor");
            }
            break;
        }
        case MpsrtRuntimeResourceLifetime::Model: {
            if (!resource.has_tensor_value) {
                return fail(log, "GFX MPSRT: model resource is not a tensor");
            }
            if (resource.role != GfxMpsrtExternalBufferRole::ConstBuffer ||
                (resource.tensor_desc.flags & GfxMpsrtTensorFlagConst) == 0) {
                return fail(log, "GFX MPSRT: model resource must be a const tensor buffer");
            }
            const ConstTensorCacheEntry* const_entry = nullptr;
            for (const auto& entry : m_const_tensor_cache) {
                if (entry.value == resource.value) {
                    const_entry = &entry;
                    break;
                }
            }
            if (!const_entry || !const_entry->buffer) {
                return fail(log, "GFX MPSRT: model resource is not materialized in the "
                                 "MPSRT prepared const resource table");
            }
            const uint64_t expected_bytes = tensor_dense_bytes(resource.tensor_desc);
            if (expected_bytes == 0 || expected_bytes != const_entry->bytes) {
                return fail(log, "GFX MPSRT: model resource prepared const byte size mismatch");
            }
            prepared.byte_length = const_entry->bytes;
            prepared.buffer = const_entry->buffer;
            prepared.offset = 0;
            prepared.cache_key = const_entry->key;
            prepared.host_bytes = const_entry->host_bytes;
            break;
        }
        case MpsrtRuntimeResourceLifetime::Unknown:
            return fail(log, "GFX MPSRT: cannot prepare resource with unknown lifetime");
        }
        out.resources.push_back(prepared);
    }
    out.image_bridge_resources.reserve(image_bridge_heap_plan_entries.size());
    for (const auto& heap_entry : image_bridge_heap_plan_entries) {
        MpsrtPreparedImageBridgeResource prepared{};
        prepared.value = heap_entry.value;
        prepared.direction = heap_entry.direction;
        prepared.tensor_desc = heap_entry.tensor_desc;
        prepared.heap_allocation_size = heap_entry.allocation_size;
        prepared.heap_alignment = heap_entry.alignment;
        prepared.texture = new_transient_image_texture(out.resource_heap, heap_entry.tensor_desc, log);
        if (!prepared.texture) {
            return false;
        }
        out.image_bridge_resources.push_back(prepared);
    }
    std::vector<PreparedResourceHeapPlanEntry> transient_allocation_entries = heap_plan_entries;
    std::sort(transient_allocation_entries.begin(), transient_allocation_entries.end(),
              [](const PreparedResourceHeapPlanEntry& lhs, const PreparedResourceHeapPlanEntry& rhs) {
                  if (lhs.first_stage_index != rhs.first_stage_index) {
                      return lhs.first_stage_index < rhs.first_stage_index;
                  }
                  if (lhs.last_stage_index != rhs.last_stage_index) {
                      return lhs.last_stage_index < rhs.last_stage_index;
                  }
                  return lhs.resource_index < rhs.resource_index;
              });

    std::vector<ActivePreparedHeapResource> active_heap_resources;
    for (const auto& heap_entry : transient_allocation_entries) {
        auto active_it = active_heap_resources.begin();
        while (active_it != active_heap_resources.end()) {
            if (active_it->last_stage_index < heap_entry.first_stage_index) {
                [active_it->resource makeAliasable];
                ++out.resource_heap_alias_reuse_count;
                active_it = active_heap_resources.erase(active_it);
                continue;
            }
            ++active_it;
        }

        auto* prepared = find_prepared_resource(out.resources, heap_entry.resource_index);
        if (!prepared) {
            return fail(log, "GFX MPSRT: transient heap plan references an unknown "
                             "prepared resource");
        }
        if (!allocate_prepared_transient_resource_from_heap(out.resource_heap, *prepared, log)) {
            return false;
        }
        id<MTLResource> heap_resource = prepared_heap_resource(*prepared);
        if (!heap_resource) {
            return fail(log, "GFX MPSRT: prepared transient resource allocation "
                             "returned no Metal resource");
        }
        active_heap_resources.push_back({heap_resource, heap_entry.last_stage_index});
    }
    return true;
}

id<MTLComputePipelineState> MpsrtContext::get_or_create_pipeline(const MpsrtRuntimeStage& stage,
                                                                 const std::string& msl_source, bool& cache_hit,
                                                                 std::string* log) {
    cache_hit = false;
    if (stage.dispatch_entry_point.empty()) {
        (void)fail(log, "GFX MPSRT: MSL dispatch entry point is empty");
        return nil;
    }
    if (msl_source.empty()) {
        (void)fail(log, "GFX MPSRT: MSL source is empty");
        return nil;
    }

    const std::string key = make_pipeline_cache_key(stage, msl_source);
    for (const auto& entry : m_pipeline_cache) {
        if (entry.key == key) {
            cache_hit = true;
            ++m_pipeline_cache_hits;
            increment_compile_counter("mpsrt_pso_cache_hit_count");
            return entry.pipeline;
        }
    }
    if (id<MTLComputePipelineState> shared_pipeline =
            lookup_shared_pipeline_cache(m_device, key)) {
        cache_hit = true;
        ++m_pipeline_cache_hits;
        increment_compile_counter("mpsrt_pso_shared_cache_hit_count");
        return shared_pipeline;
    }

    ++m_pipeline_cache_misses;
    increment_compile_counter("mpsrt_pso_cache_miss_count");

    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }

    const auto library_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLLibrary> library = [m_device newLibraryWithSource:[NSString stringWithUTF8String:msl_source.c_str()]
                                                    options:options
                                                      error:&error];
    [options release];
    if (current_compile_trace()) {
        increment_compile_counter("mpsrt_library_compile_count");
        add_compile_segment("mpsrt_library_compile",
                            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                      std::chrono::steady_clock::now() - library_start)
                                                      .count()));
    }
    if (!library || error) {
        const std::string message =
            "GFX MPSRT: failed to compile MSL library: " + ns_error_message(error, "library compile failed");
        [library release];
        (void)fail(log, message);
        return nil;
    }

    const auto function_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:stage.dispatch_entry_point.c_str()]];
    if (current_compile_trace()) {
        increment_compile_counter("mpsrt_function_lookup_count");
        add_compile_segment("mpsrt_function_lookup",
                            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                      std::chrono::steady_clock::now() - function_start)
                                                      .count()));
    }
    [library release];
    if (!function) {
        (void)fail(log, "GFX MPSRT: function " + stage.dispatch_entry_point + " not found in MSL library");
        return nil;
    }

    const auto pso_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLComputePipelineState> pipeline = nil;
    if (m_binary_archive) {
        MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
        descriptor.computeFunction = function;
        descriptor.binaryArchives = @[ m_binary_archive ];
        pipeline = [m_device newComputePipelineStateWithDescriptor:descriptor options:0 reflection:nil error:&error];
        [descriptor release];
    } else {
        pipeline = [m_device newComputePipelineStateWithFunction:function error:&error];
    }
    [function release];
    if (current_compile_trace()) {
        increment_compile_counter("mpsrt_pipeline_state_create_count");
        add_compile_segment("mpsrt_pipeline_state_create",
                            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                      std::chrono::steady_clock::now() - pso_start)
                                                      .count()));
    }
    if (!pipeline || error) {
        const std::string message =
            "GFX MPSRT: failed to create pipeline state: " + ns_error_message(error, "pipeline state creation failed");
        [pipeline release];
        (void)fail(log, message);
        return nil;
    }

    store_shared_pipeline_cache(m_device, key, pipeline);
    m_pipeline_cache.push_back({key, pipeline});
    return pipeline;
}

bool MpsrtContext::prepare_msl_dispatch(const MpsrtRuntimeStage& stage, const std::string& msl_source,
                                        MpsrtPreparedMslDispatch& out, std::string* log) {
    out = {};
    if (stage.kind != GfxMpsrtStageKind::MSLDispatch) {
        return fail(log, "GFX MPSRT: requested MSL preparation for non-MSL stage");
    }
    if (stage.dispatch_kernel_family_id == 0 || stage.msl_dispatch_desc.kernel_family == 0) {
        return fail(log, "GFX MPSRT: MSL dispatch kernel family is not set");
    }

    bool cache_hit = false;
    id<MTLComputePipelineState> pipeline = get_or_create_pipeline(stage, msl_source, cache_hit, log);
    if (!pipeline) {
        return false;
    }

    increment_compile_counter("mpsrt_prepare_msl_dispatch_count");
    out.stage_record_key = stage.stage_record_key;
    out.dispatch_entry_point = stage.dispatch_entry_point;
    out.dispatch_kernel_family_id = stage.dispatch_kernel_family_id;
    out.dispatch_threads_per_threadgroup = stage.dispatch_threads_per_threadgroup;
    out.thread_execution_width = static_cast<uint32_t>([pipeline threadExecutionWidth]);
    out.max_total_threads_per_threadgroup = static_cast<uint32_t>([pipeline maxTotalThreadsPerThreadgroup]);
    out.pipeline_cache_hit = cache_hit;
    out.pipeline = pipeline;
    return true;
}

bool MpsrtContext::prepare_mps_gemm(const MpsrtModel& model, const MpsrtRuntimeStage& stage, MpsrtPreparedMpsGemm& out,
                                    std::string* log) {
    out = {};
    if (stage.kind != GfxMpsrtStageKind::MPSGemm) {
        return fail(log, "GFX MPSRT: requested MPS GEMM preparation for non-GEMM stage");
    }
    if (!MPSSupportsMTLDevice(m_device)) {
        return fail(log, "GFX MPSRT: Metal device does not support MPS");
    }
    if (stage.inputs.size() != 2 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(log, "GFX MPSRT: MPS GEMM requires two inputs and one output");
    }

    const auto* lhs_tensor = find_tensor(model, stage.inputs[0]);
    const auto* rhs_tensor = find_tensor(model, stage.inputs[1]);
    if (!lhs_tensor || !rhs_tensor) {
        return fail(log, "GFX MPSRT: MPS GEMM input tensor descriptor is missing");
    }
    const auto& lhs = lhs_tensor->desc;
    const auto& rhs = rhs_tensor->desc;
    const auto& output = stage.output_descs.front();
    if (!validate_matrix_desc(lhs, "lhs", log) || !validate_matrix_desc(rhs, "rhs", log) ||
        !validate_matrix_desc(output, "output", log)) {
        return false;
    }

    const uint32_t result_rows = output.matrix_rows;
    const uint32_t result_columns = output.matrix_columns;
    const uint32_t interior_columns = stage.gemm_desc.transpose_lhs ? lhs.matrix_rows : lhs.matrix_columns;
    const bool uses_graph_gemm = should_use_mps_graph_gemm(stage, lhs, rhs, output);
    std::string key;
    if (!make_mps_gemm_cache_key(stage, lhs, rhs, output, result_rows, result_columns, interior_columns,
                                 uses_graph_gemm, key, log)) {
        return false;
    }

    auto fill_prepared = [&](bool kernel_cache_hit, const MpsGemmCacheEntry& entry) {
        out.stage_record_key = stage.stage_record_key;
        out.gemm_desc = stage.gemm_desc;
        out.result_rows = result_rows;
        out.result_columns = result_columns;
        out.interior_columns = interior_columns;
        out.batch_count = matrix_count_or_one(output);
        out.lhs_batch_broadcast = matrix_count_or_one(lhs) == 1 && out.batch_count > 1;
        out.rhs_batch_broadcast = matrix_count_or_one(rhs) == 1 && out.batch_count > 1;
        out.data_type = output.dtype;
        out.kernel_cache_hit = kernel_cache_hit;
        out.kernel = entry.kernel;
        out.graph_lhs_tensor = entry.graph_lhs_tensor;
        out.graph_rhs_tensor = entry.graph_rhs_tensor;
        out.graph_output_tensor = entry.graph_output_tensor;
        out.graph_executable = entry.graph_executable;
        out.uses_mps_graph_gemm = entry.uses_mps_graph_gemm;
    };

    for (const auto& entry : m_mps_gemm_cache) {
        if (entry.key == key) {
            ++m_mps_gemm_cache_hits;
            increment_compile_counter("mpsrt_mps_gemm_kernel_cache_hit_count");
            fill_prepared(true, entry);
            return true;
        }
    }

    ++m_mps_gemm_cache_misses;
    increment_compile_counter("mpsrt_mps_gemm_kernel_cache_miss_count");

    MpsGemmCacheEntry cache_entry;
    cache_entry.key = key;
    cache_entry.uses_mps_graph_gemm = uses_graph_gemm;
    if (uses_graph_gemm) {
        const MPSDataType data_type = mps_data_type_from_gfx(output.dtype);
        if (data_type == MPSDataTypeInvalid) {
            return fail(log, "GFX MPSRT: MPSGraph GEMM dtype is unsupported");
        }
        MPSGraph* graph = [[MPSGraph alloc] init];
        NSArray<NSNumber*>* lhs_shape = mps_shape_from_tensor_desc(lhs);
        NSArray<NSNumber*>* rhs_shape = mps_shape_from_tensor_desc(rhs);
        MPSGraphTensor* graph_lhs = [graph placeholderWithShape:lhs_shape
                                                       dataType:data_type
                                                           name:@"gfx_mps_gemm_lhs"];
        MPSGraphTensor* graph_rhs = [graph placeholderWithShape:rhs_shape
                                                       dataType:data_type
                                                           name:@"gfx_mps_gemm_rhs"];
        MPSGraphTensor* lhs_value = graph_lhs;
        MPSGraphTensor* rhs_value = graph_rhs;
        if (stage.gemm_desc.transpose_lhs != 0) {
            lhs_value = [graph transposeTensor:lhs_value
                                     dimension:lhs.rank - 2
                                 withDimension:lhs.rank - 1
                                          name:@"gfx_mps_gemm_lhs_t"];
        }
        if (stage.gemm_desc.transpose_rhs != 0) {
            rhs_value = [graph transposeTensor:rhs_value
                                     dimension:rhs.rank - 2
                                 withDimension:rhs.rank - 1
                                          name:@"gfx_mps_gemm_rhs_t"];
        }
        MPSGraphTensor* graph_output =
            [graph matrixMultiplicationWithPrimaryTensor:lhs_value
                                         secondaryTensor:rhs_value
                                                    name:@"gfx_mps_gemm"];
        if (!graph || !graph_lhs || !graph_rhs || !graph_output) {
            [graph release];
            return fail(log, "GFX MPSRT: failed to create MPSGraph GEMM tensors");
        }

        MPSGraphShapedType* lhs_type = [[MPSGraphShapedType alloc] initWithShape:lhs_shape
                                                                        dataType:data_type];
        MPSGraphShapedType* rhs_type = [[MPSGraphShapedType alloc] initWithShape:rhs_shape
                                                                        dataType:data_type];
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* executable_feeds =
            [NSMutableDictionary dictionaryWithObjectsAndKeys:lhs_type, graph_lhs, rhs_type, graph_rhs, nil];
        MPSGraphCompilationDescriptor* compilation_descriptor = [[MPSGraphCompilationDescriptor alloc] init];
        configure_mps_graph_full_precision_gpu_compile(compilation_descriptor);
        MPSGraphExecutable* graph_executable = nil;
        if (lhs_type && rhs_type) {
            MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:m_device];
            graph_executable = [[graph compileWithDevice:graph_device
                                                   feeds:executable_feeds
                                           targetTensors:@[ graph_output ]
                                        targetOperations:nil
                                   compilationDescriptor:compilation_descriptor] retain];
        }
        [compilation_descriptor release];
        [lhs_type release];
        [rhs_type release];

        cache_entry.kernel = graph;
        cache_entry.graph_lhs_tensor = [graph_lhs retain];
        cache_entry.graph_rhs_tensor = [graph_rhs retain];
        cache_entry.graph_output_tensor = [graph_output retain];
        cache_entry.graph_executable = graph_executable;
    } else {
        id kernel = [[MPSMatrixMultiplication alloc] initWithDevice:m_device
                                                      transposeLeft:(stage.gemm_desc.transpose_lhs != 0)
                                                     transposeRight:(stage.gemm_desc.transpose_rhs != 0)
                                                         resultRows:result_rows
                                                      resultColumns:result_columns
                                                    interiorColumns:interior_columns
                                                              alpha:stage.gemm_desc.alpha
                                                               beta:stage.gemm_desc.beta];
        if (!kernel) {
            return fail(log, "GFX MPSRT: failed to create MPSMatrixMultiplication kernel");
        }
        cache_entry.kernel = kernel;
    }

    m_mps_gemm_cache.push_back(cache_entry);
    fill_prepared(false, m_mps_gemm_cache.back());
    if (cache_entry.uses_mps_graph_gemm) {
        if (cache_entry.graph_executable) {
            increment_compile_counter("mpsrt_prepare_mps_graph_gemm_executable_count");
        }
        increment_compile_counter("mpsrt_prepare_mps_graph_gemm_count");
    }
    increment_compile_counter("mpsrt_prepare_mps_gemm_count");
    return true;
}

bool MpsrtContext::prepare_mps_conv2d(const MpsrtModel& model, const MpsrtRuntimeStage& stage,
                                      const MpsrtPreparedModel& prepared_model,
                                      MpsrtPreparedMpsConv2D& out, std::string* log) {
    out = {};
    if (!is_mps_conv2d_stage(stage.kind)) {
        return fail(log, "GFX MPSRT: requested MPS Conv2D preparation for non-Conv2D stage");
    }
    if (!MPSSupportsMTLDevice(m_device)) {
        return fail(log, "GFX MPSRT: Metal device does not support MPS");
    }
    if ((stage.inputs.size() != 2 && stage.inputs.size() != 3) ||
        stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(log, "GFX MPSRT: MPS Conv2D requires input, weights, optional bias and one output");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    const auto* weights_tensor = find_tensor(model, stage.inputs[1]);
    const auto* bias_tensor = stage.inputs.size() == 3 ? find_tensor(model, stage.inputs[2]) : nullptr;
    if (!input_tensor || !weights_tensor) {
        return fail(log, "GFX MPSRT: MPS Conv2D input or weights tensor descriptor is missing");
    }
    if (stage.inputs.size() == 3 && !bias_tensor) {
        return fail(log, "GFX MPSRT: MPS Conv2D bias tensor descriptor is missing");
    }
    const auto& input = input_tensor->desc;
    const auto& weights = weights_tensor->desc;
    const auto& output = stage.output_descs.front();
    const char* stage_name = gfx_mpsrt_stage_kind_name(stage.kind);
    if (!validate_image_desc(input, stage_name, "input", log) ||
        !validate_image_desc(output, stage_name, "output", log) || !validate_conv_weights_desc(weights, stage, log)) {
        return false;
    }
    if (input.dtype != output.dtype || weights.dtype != output.dtype) {
        return fail(log, "GFX MPSRT: MPS Conv2D input, weights and output dtype "
                         "must match for this ABI");
    }
    if (input.image_batch != output.image_batch) {
        return fail(log, "GFX MPSRT: MPS Conv2D input/output batch mismatch");
    }
    if (stage.conv2d_desc.groups == 0) {
        return fail(log, "GFX MPSRT: MPS Conv2D group count is zero");
    }
    const MpsrtPreparedResource* weights_resource =
        find_prepared_model_const_resource(prepared_model, stage.inputs[1], log);
    if (!weights_resource) {
        return false;
    }
    const MpsrtPreparedResource* bias_resource = nullptr;
    if (stage.inputs.size() == 3) {
        bias_resource = find_prepared_model_const_resource(prepared_model, stage.inputs[2], log);
        if (!bias_resource) {
            return false;
        }
    }
    const uint64_t expected_weight_bytes = tensor_dense_bytes(weights);
    if (expected_weight_bytes == 0 || expected_weight_bytes != weights_resource->byte_length ||
        expected_weight_bytes != weights_resource->host_bytes.size()) {
        return fail(log, "GFX MPSRT: MPS Conv2D prepared weight size mismatch");
    }
    if ((weights_resource->tensor_desc.flags & GfxMpsrtTensorFlagConst) == 0) {
        return fail(log, "GFX MPSRT: MPS Conv2D prepared weights are not marked const");
    }

    std::string key;
    const std::string bias_cache_key = bias_resource ? bias_resource->cache_key : std::string{};
    if (!make_mps_conv2d_cache_key(stage,
                                   input,
                                   weights,
                                   output,
                                   weights_resource->cache_key,
                                   bias_cache_key,
                                   key,
                                   log)) {
        return false;
    }
    const GfxMpsrtTensorAbiDesc* bias_desc = bias_tensor ? &bias_tensor->desc : nullptr;
    const bool use_mps_cnn_depthwise =
        is_mps_cnn_depthwise_conv2d_candidate(weights, input, output, stage, bias_desc);
    key += use_mps_cnn_depthwise ? "|mps_cnn_depthwise" : "|mps_cnn";

    auto fill_prepared = [&](bool kernel_cache_hit, const MpsConv2DCacheEntry& entry) {
        out.stage_record_key = stage.stage_record_key;
        out.conv2d_desc = stage.conv2d_desc;
        out.weights_value = stage.inputs[1];
        out.weights_byte_length = weights_resource->byte_length;
        out.input_feature_channels = input.image_feature_channels;
        out.output_feature_channels = output.image_feature_channels;
        out.output_width = output.image_width;
        out.output_height = output.image_height;
        out.output_batch = output.image_batch;
        out.data_type = output.dtype;
        out.weights_cache_hit = true;
        out.kernel_cache_hit = kernel_cache_hit;
        out.weights_buffer = weights_resource->buffer;
        out.kernel = entry.kernel;
    };

    for (const auto& entry : m_mps_conv2d_cache) {
        if (entry.key == key) {
            ++m_mps_conv2d_cache_hits;
            increment_compile_counter("mpsrt_mps_conv2d_kernel_cache_hit_count");
            fill_prepared(true, entry);
            return true;
        }
    }

    ++m_mps_conv2d_cache_misses;
    increment_compile_counter("mpsrt_mps_conv2d_kernel_cache_miss_count");

    const MPSDataType data_type = mps_data_type_from_gfx(output.dtype);
    if (data_type == MPSDataTypeInvalid) {
        return fail(log, "GFX MPSRT: MPS Conv2D dtype is unsupported");
    }

    MpsConv2DCacheEntry cache_entry;
    cache_entry.key = key;
    std::vector<uint8_t> mps_weights;
    if (use_mps_cnn_depthwise) {
        if (!pack_depthwise_weights_to_mps_cnn(weights,
                                               weights_resource->host_bytes.data(),
                                               weights_resource->host_bytes.size(),
                                               mps_weights,
                                               log)) {
            return false;
        }
    } else {
        if (!pack_conv_weights_to_mps_ohwi(weights, input, output, stage, weights_resource->host_bytes.data(),
                                           weights_resource->host_bytes.size(), mps_weights, log)) {
            return false;
        }
    }
    std::vector<float> mps_bias;
    if (bias_resource &&
        !unpack_conv_bias_to_float(bias_resource->tensor_desc,
                                   output.image_feature_channels,
                                   bias_resource->host_bytes,
                                   mps_bias,
                                   log)) {
        return false;
    }
    const uint32_t mps_input_feature_channels = input.image_feature_channels;
    const uint32_t mps_output_feature_channels = output.image_feature_channels;
    const uint32_t descriptor_groups =
        use_mps_cnn_depthwise ? 1u : mps_conv2d_descriptor_groups(weights, stage);
    if (weights.rank == 5) {
        if (use_mps_cnn_depthwise) {
            increment_compile_counter("mpsrt_prepare_mps_cnn_depthwise_conv2d_count");
        } else if (descriptor_groups == stage.conv2d_desc.groups) {
            increment_compile_counter("mpsrt_prepare_mps_group_conv2d_native_groups_count");
        } else {
            increment_compile_counter("mpsrt_prepare_mps_group_conv2d_dense_sparse_workaround_count");
        }
    }

    MPSCNNConvolutionDescriptor* descriptor = use_mps_cnn_depthwise
        ? [MPSCNNDepthWiseConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:conv2d_kernel_width(weights)
                                                                           kernelHeight:conv2d_kernel_height(weights)
                                                                   inputFeatureChannels:mps_input_feature_channels
                                                                  outputFeatureChannels:mps_output_feature_channels]
        : [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:conv2d_kernel_width(weights)
                                                                  kernelHeight:conv2d_kernel_height(weights)
                                                          inputFeatureChannels:mps_input_feature_channels
                                                         outputFeatureChannels:mps_output_feature_channels];
    if (!descriptor) {
        return fail(log, "GFX MPSRT: failed to create MPSCNNConvolutionDescriptor");
    }
    descriptor.groups = descriptor_groups;
    descriptor.strideInPixelsX = stage.conv2d_desc.strides[1] == 0 ? 1 : stage.conv2d_desc.strides[1];
    descriptor.strideInPixelsY = stage.conv2d_desc.strides[0] == 0 ? 1 : stage.conv2d_desc.strides[0];
    descriptor.dilationRateX = stage.conv2d_desc.dilations[1] == 0 ? 1 : stage.conv2d_desc.dilations[1];
    descriptor.dilationRateY = stage.conv2d_desc.dilations[0] == 0 ? 1 : stage.conv2d_desc.dilations[0];
    if (stage.conv2d_desc.fused_activation != 0) {
        MPSNNNeuronDescriptor* neuron = make_conv_fused_neuron_descriptor(stage.conv2d_desc.fused_activation, log);
        if (!neuron) {
            return false;
        }
        descriptor.fusedNeuronDescriptor = neuron;
    }

    NSString* label = [NSString stringWithUTF8String:stage.stage_record_key.c_str()];
    OVGfxMpsrtConv2DDataSource* data_source =
        [[OVGfxMpsrtConv2DDataSource alloc] initWithDescriptor:descriptor
                                                       weights:std::move(mps_weights)
                                                          bias:std::move(mps_bias)
                                                      dataType:data_type
                                                         label:label];
    id kernel = [[MPSCNNConvolution alloc] initWithDevice:m_device weights:data_source];
    if (!kernel) {
        [data_source release];
        return fail(log, "GFX MPSRT: failed to create MPSCNNConvolution kernel");
    }
    if (data_type == MPSDataTypeFloat32) {
        MPSCNNConvolution* conv_kernel = static_cast<MPSCNNConvolution*>(kernel);
        conv_kernel.accumulatorPrecisionOption = MPSNNConvolutionAccumulatorPrecisionOptionFloat;
    }
    cache_entry.kernel = kernel;
    cache_entry.data_source = data_source;
    m_mps_conv2d_cache.push_back(cache_entry);

    fill_prepared(false, m_mps_conv2d_cache.back());
    increment_compile_counter("mpsrt_prepare_mps_conv2d_count");
    return true;
}

bool MpsrtContext::prepare_mps_pool2d(const MpsrtModel& model, const MpsrtRuntimeStage& stage,
                                      MpsrtPreparedMpsPool2D& out, std::string* log) {
    out = {};
    if (stage.kind != GfxMpsrtStageKind::MPSPool2D) {
        return fail(log, "GFX MPSRT: requested MPS Pool2D preparation for non-Pool2D stage");
    }
    if (!MPSSupportsMTLDevice(m_device)) {
        return fail(log, "GFX MPSRT: Metal device does not support MPS");
    }
    if (stage.inputs.size() != 1 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(log, "GFX MPSRT: MPS Pool2D requires one input and one output");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    if (!input_tensor) {
        return fail(log, "GFX MPSRT: MPS Pool2D input tensor descriptor is missing");
    }
    const auto& input = input_tensor->desc;
    const auto& output = stage.output_descs.front();
    if (!validate_image_desc(input, "mps_pool2d", "input", log) ||
        !validate_image_desc(output, "mps_pool2d", "output", log)) {
        return false;
    }

    std::string key;
    if (!make_mps_pool2d_cache_key(stage, input, output, key, log)) {
        return false;
    }

    auto fill_prepared = [&](bool kernel_cache_hit, id kernel) {
        out.stage_record_key = stage.stage_record_key;
        out.pool2d_desc = stage.pool2d_desc;
        out.output_width = output.image_width;
        out.output_height = output.image_height;
        out.output_batch = output.image_batch;
        out.output_feature_channels = output.image_feature_channels;
        out.data_type = output.dtype;
        out.kernel_cache_hit = kernel_cache_hit;
        out.kernel = kernel;
    };

    for (const auto& entry : m_mps_pool2d_cache) {
        if (entry.key == key) {
            ++m_mps_pool2d_cache_hits;
            increment_compile_counter("mpsrt_mps_pool2d_kernel_cache_hit_count");
            fill_prepared(true, entry.kernel);
            return true;
        }
    }

    ++m_mps_pool2d_cache_misses;
    increment_compile_counter("mpsrt_mps_pool2d_kernel_cache_miss_count");

    id kernel = nil;
    if (stage.pool2d_desc.is_avg != 0) {
        kernel = [[MPSCNNPoolingAverage alloc] initWithDevice:m_device
                                                  kernelWidth:stage.pool2d_desc.kernel[1]
                                                 kernelHeight:stage.pool2d_desc.kernel[0]
                                              strideInPixelsX:stage.pool2d_desc.strides[1]
                                              strideInPixelsY:stage.pool2d_desc.strides[0]];
    } else {
        kernel = [[MPSCNNPoolingMax alloc] initWithDevice:m_device
                                              kernelWidth:stage.pool2d_desc.kernel[1]
                                             kernelHeight:stage.pool2d_desc.kernel[0]
                                          strideInPixelsX:stage.pool2d_desc.strides[1]
                                          strideInPixelsY:stage.pool2d_desc.strides[0]];
    }
    if (!kernel) {
        return fail(log, "GFX MPSRT: failed to create MPSCNNPooling kernel");
    }

    m_mps_pool2d_cache.push_back({key, kernel});
    fill_prepared(false, kernel);
    increment_compile_counter("mpsrt_prepare_mps_pool2d_count");
    return true;
}

bool MpsrtContext::prepare_mps_resize2d(const MpsrtModel& model, const MpsrtRuntimeStage& stage,
                                        MpsrtPreparedMpsResize2D& out, std::string* log) {
    out = {};
    if (stage.kind != GfxMpsrtStageKind::MPSResize2D) {
        return fail(log, "GFX MPSRT: requested MPS Resize2D preparation for non-Resize2D stage");
    }
    if (!MPSSupportsMTLDevice(m_device)) {
        return fail(log, "GFX MPSRT: Metal device does not support MPS");
    }
    if (stage.inputs.size() != 1 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(log, "GFX MPSRT: MPS Resize2D requires one input and one output");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    if (!input_tensor) {
        return fail(log, "GFX MPSRT: MPS Resize2D input tensor descriptor is missing");
    }
    const auto& input = input_tensor->desc;
    const auto& output = stage.output_descs.front();
    if (!validate_image_desc(input, "mps_resize2d", "input", log) ||
        !validate_image_desc(output, "mps_resize2d", "output", log)) {
        return false;
    }

    std::string key;
    if (!make_mps_resize2d_cache_key(stage, input, output, key, log)) {
        return false;
    }

    auto fill_prepared = [&](bool kernel_cache_hit, id kernel) {
        out.stage_record_key = stage.stage_record_key;
        out.resize2d_desc = stage.resize2d_desc;
        out.input_width = input.image_width;
        out.input_height = input.image_height;
        out.output_width = output.image_width;
        out.output_height = output.image_height;
        out.output_batch = output.image_batch;
        out.output_feature_channels = output.image_feature_channels;
        out.data_type = output.dtype;
        out.kernel_cache_hit = kernel_cache_hit;
        out.kernel = kernel;
    };

    for (const auto& entry : m_mps_resize2d_cache) {
        if (entry.key == key) {
            ++m_mps_resize2d_cache_hits;
            increment_compile_counter("mpsrt_mps_resize2d_kernel_cache_hit_count");
            fill_prepared(true, entry.kernel);
            return true;
        }
    }

    ++m_mps_resize2d_cache_misses;
    increment_compile_counter("mpsrt_mps_resize2d_kernel_cache_miss_count");

    MPSImageBilinearScale* kernel = [[MPSImageBilinearScale alloc] initWithDevice:m_device];
    if (!kernel) {
        return fail(log, "GFX MPSRT: failed to create MPSImageBilinearScale kernel");
    }

    m_mps_resize2d_cache.push_back({key, kernel});
    fill_prepared(false, kernel);
    increment_compile_counter("mpsrt_prepare_mps_resize2d_count");
    return true;
}

bool MpsrtContext::prepare_mps_softmax(const MpsrtModel& model, const MpsrtRuntimeStage& stage,
                                       MpsrtPreparedMpsSoftmax& out, std::string* log) {
    out = {};
    if (stage.kind != GfxMpsrtStageKind::MPSSoftmax) {
        return fail(log, "GFX MPSRT: requested MPS Softmax preparation for non-Softmax stage");
    }
    if (!MPSSupportsMTLDevice(m_device)) {
        return fail(log, "GFX MPSRT: Metal device does not support MPS");
    }
    if (stage.inputs.size() != 1 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(log, "GFX MPSRT: MPS Softmax requires one input and one output");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    if (!input_tensor) {
        return fail(log, "GFX MPSRT: MPS Softmax input tensor descriptor is missing");
    }
    const auto& input = input_tensor->desc;
    const auto& output = stage.output_descs.front();
    if (!validate_matrix_desc(input, "input", log) || !validate_matrix_desc(output, "output", log)) {
        return false;
    }

    std::string key;
    if (!make_mps_softmax_cache_key(stage, input, output, key, log)) {
        return false;
    }

    auto fill_prepared = [&](bool kernel_cache_hit, id kernel) {
        out.stage_record_key = stage.stage_record_key;
        out.softmax_desc = stage.softmax_desc;
        out.rows = output.matrix_rows;
        out.columns = output.matrix_columns;
        out.matrix_count = matrix_count_or_one(output);
        out.data_type = output.dtype;
        out.kernel_cache_hit = kernel_cache_hit;
        out.kernel = kernel;
    };

    for (const auto& entry : m_mps_softmax_cache) {
        if (entry.key == key) {
            ++m_mps_softmax_cache_hits;
            increment_compile_counter("mpsrt_mps_softmax_kernel_cache_hit_count");
            fill_prepared(true, entry.kernel);
            return true;
        }
    }

    ++m_mps_softmax_cache_misses;
    increment_compile_counter("mpsrt_mps_softmax_kernel_cache_miss_count");

    id kernel = [[MPSMatrixSoftMax alloc] initWithDevice:m_device];
    if (!kernel) {
        return fail(log, "GFX MPSRT: failed to create MPSMatrixSoftMax kernel");
    }

    m_mps_softmax_cache.push_back({key, kernel});
    fill_prepared(false, kernel);
    increment_compile_counter("mpsrt_prepare_mps_softmax_count");
    return true;
}

bool MpsrtContext::prepare_mps_topk(const MpsrtModel& model, const MpsrtRuntimeStage& stage, MpsrtPreparedMpsTopK& out,
                                    std::string* log) {
    out = {};
    if (stage.kind != GfxMpsrtStageKind::MPSTopK) {
        return fail(log, "GFX MPSRT: requested MPS TopK preparation for non-TopK stage");
    }
    if (!MPSSupportsMTLDevice(m_device)) {
        return fail(log, "GFX MPSRT: Metal device does not support MPS");
    }
    if (stage.inputs.size() != 1 || stage.outputs.size() != 2 || stage.output_descs.size() != 2) {
        return fail(log, "GFX MPSRT: MPS TopK requires one input and two outputs");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    if (!input_tensor) {
        return fail(log, "GFX MPSRT: MPS TopK input tensor descriptor is missing");
    }
    const auto& input = input_tensor->desc;
    const auto& values_output = stage.output_descs[0];
    const auto& indices_output = stage.output_descs[1];
    if (!validate_matrix_desc(input, "input", log) || !validate_matrix_desc(values_output, "values output", log)) {
        return false;
    }

    std::string key;
    if (!make_mps_topk_cache_key(stage, input, values_output, indices_output, key, log)) {
        return false;
    }

    auto fill_prepared = [&](bool kernel_cache_hit, const MpsTopKCacheEntry& entry) {
        out.stage_record_key = stage.stage_record_key;
        out.topk_desc = stage.topk_desc;
        out.rows = values_output.matrix_rows;
        out.source_columns = input.matrix_columns;
        out.k = stage.topk_desc.k;
        out.matrix_count = matrix_count_or_one(values_output);
        out.data_type = values_output.dtype;
        out.index_type = indices_output.dtype;
        out.kernel_cache_hit = kernel_cache_hit;
        out.kernel = entry.kernel;
        out.graph_input_tensor = entry.graph_input_tensor;
        out.graph_values_tensor = entry.graph_values_tensor;
        out.graph_indices_tensor = entry.graph_indices_tensor;
        out.graph_executable = entry.graph_executable;
        out.uses_mps_graph_topk = entry.uses_mps_graph_topk;
    };

    for (const auto& entry : m_mps_topk_cache) {
        if (entry.key == key) {
            ++m_mps_topk_cache_hits;
            increment_compile_counter("mpsrt_mps_topk_kernel_cache_hit_count");
            fill_prepared(true, entry);
            return true;
        }
    }

    ++m_mps_topk_cache_misses;
    increment_compile_counter("mpsrt_mps_topk_kernel_cache_miss_count");

    MpsTopKCacheEntry cache_entry;
    cache_entry.key = key;
    if (stage.topk_desc.k > 16) {
        const MPSDataType data_type = mps_data_type_from_gfx(input.dtype);
        if (data_type == MPSDataTypeInvalid) {
            return fail(log, "GFX MPSRT: MPSGraph TopK dtype is unsupported");
        }
        MPSGraph* graph = [[MPSGraph alloc] init];
        NSArray<NSNumber*>* input_shape = mps_shape_from_tensor_desc(input);
        MPSGraphTensor* graph_input = [graph placeholderWithShape:input_shape
                                                         dataType:data_type
                                                             name:@"gfx_mps_topk_input"];
        NSArray<MPSGraphTensor*>* graph_outputs =
            [graph topKWithSourceTensor:graph_input k:stage.topk_desc.k name:@"gfx_mps_topk"];
        if (!graph || !graph_input || graph_outputs.count < 2) {
            [graph release];
            return fail(log, "GFX MPSRT: failed to create MPSGraph TopK tensors");
        }
        MPSGraphTensor* graph_values = graph_outputs[0];
        MPSGraphTensor* graph_indices = graph_outputs[1];
        MPSGraphShapedType* input_type = [[MPSGraphShapedType alloc] initWithShape:input_shape dataType:data_type];
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* executable_feeds =
            [NSMutableDictionary dictionaryWithObjectsAndKeys:input_type, graph_input, nil];
        MPSGraphCompilationDescriptor* compilation_descriptor = [[MPSGraphCompilationDescriptor alloc] init];
        configure_mps_graph_full_precision_gpu_compile(compilation_descriptor);
        MPSGraphExecutable* graph_executable = nil;
        if (input_type) {
            MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:m_device];
            graph_executable = [[graph compileWithDevice:graph_device
                                                   feeds:executable_feeds
                                           targetTensors:@[ graph_values, graph_indices ]
                                        targetOperations:nil
                                   compilationDescriptor:compilation_descriptor] retain];
        }
        [compilation_descriptor release];
        [input_type release];

        cache_entry.kernel = graph;
        cache_entry.graph_input_tensor = [graph_input retain];
        cache_entry.graph_values_tensor = [graph_values retain];
        cache_entry.graph_indices_tensor = [graph_indices retain];
        cache_entry.graph_executable = graph_executable;
        cache_entry.uses_mps_graph_topk = true;
    } else {
        id kernel = [[MPSMatrixFindTopK alloc] initWithDevice:m_device numberOfTopKValues:stage.topk_desc.k];
        if (!kernel) {
            return fail(log, "GFX MPSRT: failed to create MPSMatrixFindTopK kernel");
        }
        cache_entry.kernel = kernel;
    }

    m_mps_topk_cache.push_back(cache_entry);
    fill_prepared(false, m_mps_topk_cache.back());
    if (cache_entry.uses_mps_graph_topk) {
        if (cache_entry.graph_executable) {
            increment_compile_counter("mpsrt_prepare_mps_graph_topk_executable_count");
        }
        increment_compile_counter("mpsrt_prepare_mps_graph_topk_count");
    }
    increment_compile_counter("mpsrt_prepare_mps_topk_count");
    return true;
}

bool MpsrtContext::prepare_mps_sdpa(const MpsrtModel& model, const MpsrtRuntimeStage& stage, MpsrtPreparedMpsSdpa& out,
                                    std::string* log) {
    out = {};
    if (stage.kind != GfxMpsrtStageKind::MPSSdpa) {
        return fail(log, "GFX MPSRT: requested MPS SDPA preparation for non-SDPA stage");
    }
    if (!MPSSupportsMTLDevice(m_device)) {
        return fail(log, "GFX MPSRT: Metal device does not support MPS");
    }
    if (stage.inputs.size() != 3 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(log, "GFX MPSRT: MPS SDPA requires Q, K, V and one output");
    }

    const auto* query_tensor = find_tensor(model, stage.inputs[0]);
    const auto* key_tensor = find_tensor(model, stage.inputs[1]);
    const auto* value_tensor = find_tensor(model, stage.inputs[2]);
    if (!query_tensor || !key_tensor || !value_tensor) {
        return fail(log, "GFX MPSRT: MPS SDPA input tensor descriptor is missing");
    }
    const auto& query = query_tensor->desc;
    const auto& key_desc = key_tensor->desc;
    const auto& value = value_tensor->desc;
    const auto& output = stage.output_descs.front();
    std::string key;
    if (!make_mps_sdpa_cache_key(stage, query, key_desc, value, output, key, log)) {
        return false;
    }

    auto fill_prepared = [&](bool kernel_cache_hit, const MpsSdpaCacheEntry& entry) {
        out.stage_record_key = stage.stage_record_key;
        out.sdpa_desc = stage.sdpa_desc;
        out.data_type = output.dtype;
        out.kernel_cache_hit = kernel_cache_hit;
        out.kernel = entry.kernel;
        out.graph_query_tensor = entry.graph_query_tensor;
        out.graph_key_tensor = entry.graph_key_tensor;
        out.graph_value_tensor = entry.graph_value_tensor;
        out.graph_output_tensor = entry.graph_output_tensor;
        out.graph_executable = entry.graph_executable;
    };

    for (const auto& entry : m_mps_sdpa_cache) {
        if (entry.key == key) {
            ++m_mps_sdpa_cache_hits;
            increment_compile_counter("mpsrt_mps_sdpa_kernel_cache_hit_count");
            fill_prepared(true, entry);
            return true;
        }
    }

    ++m_mps_sdpa_cache_misses;
    increment_compile_counter("mpsrt_mps_sdpa_kernel_cache_miss_count");

    const bool transposed_layout = stage.sdpa_desc.layout == GfxMpsrtSdpaLayoutTransposedBHDN;
    if (transposed_layout) {
        const MPSDataType data_type = mps_data_type_from_gfx(output.dtype);
        if (data_type == MPSDataTypeInvalid) {
            return fail(log, "GFX MPSRT: MPSGraph SDPA dtype is unsupported");
        }
        MPSGraph* graph = [[MPSGraph alloc] init];
        NSArray<NSNumber*>* query_shape = mps_shape_from_tensor_desc(query);
        NSArray<NSNumber*>* key_shape = mps_shape_from_tensor_desc(key_desc);
        NSArray<NSNumber*>* value_shape = mps_shape_from_tensor_desc(value);
        MPSGraphTensor* graph_query = [graph placeholderWithShape:query_shape
                                                         dataType:data_type
                                                             name:@"gfx_mps_sdpa_query"];
        MPSGraphTensor* graph_key = [graph placeholderWithShape:key_shape
                                                       dataType:data_type
                                                           name:@"gfx_mps_sdpa_key"];
        MPSGraphTensor* graph_value = [graph placeholderWithShape:value_shape
                                                         dataType:data_type
                                                             name:@"gfx_mps_sdpa_value"];
        MPSGraphTensor* query_bhnd = [graph transposeTensor:graph_query
                                                  dimension:2
                                              withDimension:3
                                                       name:@"gfx_mps_sdpa_query_bhnd"];
        MPSGraphTensor* scores =
            [graph matrixMultiplicationWithPrimaryTensor:query_bhnd
                                         secondaryTensor:graph_key
                                                    name:@"gfx_mps_sdpa_scores"];
        MPSGraphTensor* scale =
            [graph constantWithScalar:stage.sdpa_desc.scale
                              dataType:data_type];
        MPSGraphTensor* scaled_scores =
            [graph multiplicationWithPrimaryTensor:scores
                                   secondaryTensor:scale
                                              name:@"gfx_mps_sdpa_scaled_scores"];
        MPSGraphTensor* weights = [graph softMaxWithTensor:scaled_scores
                                                      axis:-1
                                                      name:@"gfx_mps_sdpa_weights"];
        MPSGraphTensor* value_bhnv = [graph transposeTensor:graph_value
                                                  dimension:2
                                              withDimension:3
                                                       name:@"gfx_mps_sdpa_value_bhnv"];
        MPSGraphTensor* output_bhnv =
            [graph matrixMultiplicationWithPrimaryTensor:weights
                                         secondaryTensor:value_bhnv
                                                    name:@"gfx_mps_sdpa_output_bhnv"];
        MPSGraphTensor* graph_output = [graph transposeTensor:output_bhnv
                                                    dimension:2
                                                withDimension:3
                                                         name:@"gfx_mps_sdpa"];
        if (!graph || !graph_query || !graph_key || !graph_value || !graph_output) {
            [graph release];
            return fail(log, "GFX MPSRT: failed to create MPSGraph SDPA tensors");
        }

        MPSGraphShapedType* query_type = [[MPSGraphShapedType alloc] initWithShape:query_shape
                                                                          dataType:data_type];
        MPSGraphShapedType* key_type = [[MPSGraphShapedType alloc] initWithShape:key_shape
                                                                        dataType:data_type];
        MPSGraphShapedType* value_type = [[MPSGraphShapedType alloc] initWithShape:value_shape
                                                                          dataType:data_type];
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* executable_feeds =
            [NSMutableDictionary dictionaryWithObjectsAndKeys:query_type, graph_query, key_type, graph_key,
                                                               value_type, graph_value, nil];
        MPSGraphCompilationDescriptor* compilation_descriptor = [[MPSGraphCompilationDescriptor alloc] init];
        configure_mps_graph_full_precision_gpu_compile(compilation_descriptor);
        MPSGraphExecutable* graph_executable = nil;
        if (query_type && key_type && value_type) {
            MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:m_device];
            graph_executable = [[graph compileWithDevice:graph_device
                                                   feeds:executable_feeds
                                           targetTensors:@[ graph_output ]
                                        targetOperations:nil
                                   compilationDescriptor:compilation_descriptor] retain];
        }
        [compilation_descriptor release];
        [query_type release];
        [key_type release];
        [value_type release];

        MpsSdpaCacheEntry cache_entry;
        cache_entry.key = key;
        cache_entry.kernel = graph;
        cache_entry.graph_query_tensor = [graph_query retain];
        cache_entry.graph_key_tensor = [graph_key retain];
        cache_entry.graph_value_tensor = [graph_value retain];
        cache_entry.graph_output_tensor = [graph_output retain];
        cache_entry.graph_executable = graph_executable;
        m_mps_sdpa_cache.push_back(cache_entry);
        fill_prepared(false, m_mps_sdpa_cache.back());
        if (graph_executable) {
            increment_compile_counter("mpsrt_prepare_mps_graph_sdpa_executable_count");
        }
        increment_compile_counter("mpsrt_prepare_mps_graph_sdpa_count");
        increment_compile_counter("mpsrt_prepare_mps_sdpa_count");
        return true;
    }

    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
        const MPSDataType data_type = mps_data_type_from_gfx(output.dtype);
        if (data_type == MPSDataTypeInvalid) {
            return fail(log, "GFX MPSRT: MPSGraph SDPA dtype is unsupported");
        }
        MPSGraph* graph = [[MPSGraph alloc] init];
        NSArray<NSNumber*>* query_shape = mps_shape_from_tensor_desc(query);
        NSArray<NSNumber*>* key_shape = mps_shape_from_tensor_desc(key_desc);
        NSArray<NSNumber*>* value_shape = mps_shape_from_tensor_desc(value);
        MPSGraphTensor* graph_query = [graph placeholderWithShape:query_shape
                                                         dataType:data_type
                                                             name:@"gfx_mps_sdpa_query"];
        MPSGraphTensor* graph_key = [graph placeholderWithShape:key_shape
                                                       dataType:data_type
                                                           name:@"gfx_mps_sdpa_key"];
        MPSGraphTensor* graph_value = [graph placeholderWithShape:value_shape
                                                         dataType:data_type
                                                             name:@"gfx_mps_sdpa_value"];
        MPSGraphTensor* graph_output =
            [graph scaledDotProductAttentionWithQueryTensor:graph_query
                                                  keyTensor:graph_key
                                                valueTensor:graph_value
                                                      scale:stage.sdpa_desc.scale
                                                       name:@"gfx_mps_sdpa"];
        if (!graph || !graph_query || !graph_key || !graph_value || !graph_output) {
            [graph release];
            return fail(log, "GFX MPSRT: failed to create MPSGraph SDPA tensors");
        }

        MPSGraphShapedType* query_type = [[MPSGraphShapedType alloc] initWithShape:query_shape
                                                                          dataType:data_type];
        MPSGraphShapedType* key_type = [[MPSGraphShapedType alloc] initWithShape:key_shape
                                                                        dataType:data_type];
        MPSGraphShapedType* value_type = [[MPSGraphShapedType alloc] initWithShape:value_shape
                                                                          dataType:data_type];
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* executable_feeds =
            [NSMutableDictionary dictionaryWithObjectsAndKeys:query_type, graph_query, key_type, graph_key,
                                                               value_type, graph_value, nil];
        MPSGraphCompilationDescriptor* compilation_descriptor = [[MPSGraphCompilationDescriptor alloc] init];
        configure_mps_graph_full_precision_gpu_compile(compilation_descriptor);
        MPSGraphExecutable* graph_executable = nil;
        if (query_type && key_type && value_type) {
            MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:m_device];
            graph_executable = [[graph compileWithDevice:graph_device
                                                   feeds:executable_feeds
                                           targetTensors:@[ graph_output ]
                                        targetOperations:nil
                                   compilationDescriptor:compilation_descriptor] retain];
        }
        [compilation_descriptor release];
        [query_type release];
        [key_type release];
        [value_type release];

        MpsSdpaCacheEntry cache_entry;
        cache_entry.key = key;
        cache_entry.kernel = graph;
        cache_entry.graph_query_tensor = [graph_query retain];
        cache_entry.graph_key_tensor = [graph_key retain];
        cache_entry.graph_value_tensor = [graph_value retain];
        cache_entry.graph_output_tensor = [graph_output retain];
        cache_entry.graph_executable = graph_executable;
        m_mps_sdpa_cache.push_back(cache_entry);
        fill_prepared(false, m_mps_sdpa_cache.back());
        if (graph_executable) {
            increment_compile_counter("mpsrt_prepare_mps_graph_sdpa_executable_count");
        }
        increment_compile_counter("mpsrt_prepare_mps_graph_sdpa_count");
        increment_compile_counter("mpsrt_prepare_mps_sdpa_count");
        return true;
    }

    return fail(log, "GFX MPSRT: MPSGraph SDPA requires macOS 15.0/iOS 18.0/tvOS 18.0 or newer");
}

bool MpsrtContext::prepare_model(const MpsrtModel& model, const std::string& msl_source, MpsrtPreparedModel& out,
                                 std::string* log) {
    out = {};
    if (!prepare_model_resources(model, out, log)) {
        return false;
    }
    for (size_t i = 0; i < model.stages.size(); ++i) {
        const auto& stage = model.stages[i];
        if (stage.kind == GfxMpsrtStageKind::MSLDispatch) {
            MpsrtPreparedMslDispatch prepared;
            if (!prepare_msl_dispatch(stage, msl_source, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.msl_dispatches.push_back(prepared);
        } else if (stage.kind == GfxMpsrtStageKind::MPSGemm) {
            MpsrtPreparedMpsGemm prepared;
            if (!prepare_mps_gemm(model, stage, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.mps_gemm_stages.push_back(prepared);
        } else if (is_mps_conv2d_stage(stage.kind)) {
            MpsrtPreparedMpsConv2D prepared;
            if (!prepare_mps_conv2d(model, stage, out, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.mps_conv2d_stages.push_back(prepared);
        } else if (stage.kind == GfxMpsrtStageKind::MPSPool2D) {
            MpsrtPreparedMpsPool2D prepared;
            if (!prepare_mps_pool2d(model, stage, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.mps_pool2d_stages.push_back(prepared);
        } else if (stage.kind == GfxMpsrtStageKind::MPSResize2D) {
            MpsrtPreparedMpsResize2D prepared;
            if (!prepare_mps_resize2d(model, stage, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.mps_resize2d_stages.push_back(prepared);
        } else if (stage.kind == GfxMpsrtStageKind::MPSSoftmax) {
            MpsrtPreparedMpsSoftmax prepared;
            if (!prepare_mps_softmax(model, stage, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.mps_softmax_stages.push_back(prepared);
        } else if (stage.kind == GfxMpsrtStageKind::MPSTopK) {
            MpsrtPreparedMpsTopK prepared;
            if (!prepare_mps_topk(model, stage, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.mps_topk_stages.push_back(prepared);
        } else if (stage.kind == GfxMpsrtStageKind::MPSSdpa) {
            MpsrtPreparedMpsSdpa prepared;
            if (!prepare_mps_sdpa(model, stage, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.mps_sdpa_stages.push_back(prepared);
        } else if (is_first_class_mps_stage(stage.kind)) {
            return fail(log, unsupported_mps_stage_message(stage));
        } else {
            ++out.skipped_non_msl_stages;
        }
    }
    return true;
}

} // namespace mpsrt
} // namespace metal
} // namespace gfx_plugin
} // namespace ov
