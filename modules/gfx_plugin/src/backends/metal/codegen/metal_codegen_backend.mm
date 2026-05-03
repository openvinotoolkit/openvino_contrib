// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/codegen/metal_codegen_backend.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backends/metal/codegen/metal_compiler.hpp"
#include "backends/metal/runtime/metal_command_encoder.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_request.hpp"
#include "mlir/mlir_passes.hpp"
#include "openvino/core/except.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/gfx_mpsrt_runtime_abi_pipeline.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_mpsrt_storage_bridge.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

namespace ov {
namespace gfx_plugin {

class MetalBindingSchema final {
public:
    explicit MetalBindingSchema(uint32_t arg_count) : m_arg_count(arg_count) {}

    uint32_t arg_count() const {
        return m_arg_count;
    }

private:
    uint32_t m_arg_count = 0;
};

class MetalDeviceReuseContext final {
public:
    explicit MetalDeviceReuseContext(MetalDeviceHandle device) : m_device(device) {}

    std::shared_ptr<MetalBindingSchema> acquire_binding_schema(uint32_t arg_count) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (auto it = m_binding_schemas.find(arg_count); it != m_binding_schemas.end()) {
            if (auto schema = it->second.lock()) {
                return schema;
            }
        }
        auto schema = std::make_shared<MetalBindingSchema>(arg_count);
        m_binding_schemas[arg_count] = schema;
        return schema;
    }

private:
    MetalDeviceHandle m_device = nullptr;
    std::mutex m_mutex;
    std::unordered_map<uint32_t, std::weak_ptr<MetalBindingSchema>> m_binding_schemas;
};

class MetalDeviceReuseRegistry final {
public:
    static MetalDeviceReuseRegistry& instance() {
        static MetalDeviceReuseRegistry registry;
        return registry;
    }

    std::shared_ptr<MetalDeviceReuseContext> acquire(MetalDeviceHandle device) {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto key = reinterpret_cast<uintptr_t>(device);
        if (auto it = m_contexts.find(key); it != m_contexts.end()) {
            if (auto context = it->second.lock()) {
                return context;
            }
        }
        auto context = std::make_shared<MetalDeviceReuseContext>(device);
        m_contexts[key] = context;
        return context;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<uintptr_t, std::weak_ptr<MetalDeviceReuseContext>> m_contexts;
};

namespace {
inline id<MTLBuffer> to_mtl(const GpuBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

std::string make_resolved_msl_cache_key(const KernelSource& source) {
    if (!source.module) {
        return {};
    }
    std::string module_text;
    llvm::raw_string_ostream os(module_text);
    auto module = source.module;
    module.print(os);
    os.flush();

    std::ostringstream key;
    key << source.entry_point << '\n'
        << source.signature.arg_count << ':'
        << source.signature.output_arg_count << '\n'
        << module_text;
    return key.str();
}

uint32_t resolve_kernel_output_arg_count(const KernelSource& source) {
    if (source.signature.output_arg_count != 0) {
        return source.signature.output_arg_count;
    }
    if (!source.module) {
        return 0;
    }
    if (auto attr = source.module->getAttrOfType<mlir::IntegerAttr>("gfx.kernel_output_arg_count")) {
        return static_cast<uint32_t>(std::max<int64_t>(attr.getInt(), 0));
    }
    return 0;
}

bool set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
    return false;
}

bool make_mpsrt_external_io_bindings(const metal::mpsrt::MpsrtModel& model,
                                     const std::vector<void*>& buffer_ptrs,
                                     const std::vector<size_t>& offsets,
                                     uint32_t output_arg_count,
                                     std::vector<metal::mpsrt::MpsrtBoundBuffer>& input_buffers,
                                     std::vector<metal::mpsrt::MpsrtBoundBuffer>& output_buffers,
                                     std::string* error) {
    const auto bound_buffers = metal::mpsrt::make_mpsrt_bound_buffers(buffer_ptrs, offsets);
    const size_t input_count = model.input_values.size();
    const size_t output_count = model.output_values.size();
    if (bound_buffers.size() < input_count) {
        return set_error(error, "GFX MPSRT: runtime buffers do not cover model inputs");
    }

    input_buffers.assign(bound_buffers.begin(), bound_buffers.begin() + input_count);
    output_buffers.clear();
    if (output_count == 0) {
        return true;
    }

    if (output_arg_count != 0) {
        if (output_arg_count != output_count) {
            return set_error(error, "GFX MPSRT: output arg count does not match model outputs");
        }
        if (bound_buffers.size() < output_count) {
            return set_error(error, "GFX MPSRT: runtime buffers do not cover model outputs");
        }
        output_buffers.assign(bound_buffers.end() - output_count, bound_buffers.end());
        return true;
    }

    if (bound_buffers.size() >= input_count + output_count) {
        output_buffers.assign(bound_buffers.begin() + input_count,
                              bound_buffers.begin() + input_count + output_count);
        return true;
    }

    if (bound_buffers.size() == input_count && input_count == output_count) {
        output_buffers = input_buffers;
        return true;
    }

    return set_error(error, "GFX MPSRT: cannot infer model output bindings from runtime buffers");
}

MTLPixelFormat mpsrt_texture_pixel_format_from_dtype(uint32_t dtype) {
    switch (static_cast<GfxMpsrtDType>(dtype)) {
        case GfxMpsrtDType::F16:
            return MTLPixelFormatRGBA16Float;
        case GfxMpsrtDType::F32:
            return MTLPixelFormatRGBA32Float;
        default:
            return MTLPixelFormatInvalid;
    }
}

uint32_t mpsrt_image_slice_count(uint32_t feature_channels) {
    return (feature_channels + 3) / 4;
}

id<MTLTexture> new_mpsrt_image_texture(MetalDeviceHandle device,
                                       const GfxMpsrtTensorAbiDesc& desc,
                                       std::string* error) {
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Image)) {
        set_error(error, "GFX MPSRT: cannot allocate MTLTexture for non-image tensor");
        return nil;
    }
    if (desc.image_width == 0 || desc.image_height == 0 || desc.image_feature_channels == 0 ||
        desc.image_batch == 0) {
        set_error(error, "GFX MPSRT: cannot allocate MTLTexture for incomplete image tensor descriptor");
        return nil;
    }

    const MTLPixelFormat pixel_format = mpsrt_texture_pixel_format_from_dtype(desc.dtype);
    if (pixel_format == MTLPixelFormatInvalid) {
        set_error(error, "GFX MPSRT: cannot allocate MTLTexture for unsupported image dtype");
        return nil;
    }

    const uint32_t slices = mpsrt_image_slice_count(desc.image_feature_channels);
    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                                                          width:desc.image_width
                                                         height:desc.image_height
                                                      mipmapped:false];
    descriptor.textureType = MTLTextureType2DArray;
    descriptor.arrayLength = static_cast<NSUInteger>(desc.image_batch * slices);
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    descriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texture = [static_cast<id<MTLDevice>>(device) newTextureWithDescriptor:descriptor];
    if (!texture) {
        set_error(error, "GFX MPSRT: failed to allocate transient image MTLTexture");
    }
    return texture;
}

struct MpsrtImageBridgeCopy {
    GfxMpsrtStorageBridgeDirection direction = GfxMpsrtStorageBridgeDirection::BufferToImage;
    GfxMpsrtValue value = 0;
    GfxMpsrtTensorAbiDesc desc{};
    metal::mpsrt::MpsrtBoundBuffer buffer_binding{};
    metal::mpsrt::MpsrtBoundBuffer image_binding{};
};

struct MpsrtImageBridgeParams {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    uint32_t batch = 0;
};

const metal::mpsrt::MpsrtRuntimeTensor* find_mpsrt_tensor(const metal::mpsrt::MpsrtModel& model,
                                                         GfxMpsrtValue value) {
    for (const auto& tensor : model.tensors) {
        if (tensor.value == value) {
            return &tensor;
        }
    }
    return nullptr;
}

bool has_mpsrt_value(const std::vector<GfxMpsrtValue>& values, GfxMpsrtValue value) {
    for (const auto known : values) {
        if (known == value) {
            return true;
        }
    }
    return false;
}

const GfxMpsrtStorageBridgeDesc* find_mpsrt_storage_bridge(const metal::mpsrt::MpsrtModel& model,
                                                           GfxMpsrtValue value) {
    for (const auto& bridge : model.storage_bridges) {
        if (bridge.value == value) {
            return &bridge;
        }
    }
    return nullptr;
}

GfxMpsrtStorageBridgeDirection mpsrt_external_image_bridge_direction_for_value(
    const metal::mpsrt::MpsrtModel& model,
    GfxMpsrtValue value,
    GfxMpsrtStorageBridgeDirection legacy_fallback) {
    if (model.storage_bridges.empty()) {
        return legacy_fallback;
    }
    if (const auto* bridge = find_mpsrt_storage_bridge(model, value)) {
        return bridge->direction;
    }
    return GfxMpsrtStorageBridgeDirection::Unknown;
}

bool register_mpsrt_const_tensor_sources(MetalCompiledKernel& kernel,
                                         const metal::mpsrt::MpsrtModel& model,
                                         const std::vector<MpsrtConstTensorSource>& const_tensors,
                                         std::string* log) {
    for (const auto& payload : const_tensors) {
        if (payload.bytes.empty()) {
            return set_error(log, "GFX MPSRT: const tensor source payload is empty");
        }
        const auto* tensor = find_mpsrt_tensor(model, payload.value);
        if (!tensor) {
            std::ostringstream stream;
            stream << "GFX MPSRT: const tensor source references unknown value " << payload.value;
            return set_error(log, stream.str());
        }
        if (!kernel.register_mpsrt_const_tensor_data(payload.value,
                                                     tensor->desc,
                                                     payload.bytes.data(),
                                                     payload.bytes.size(),
                                                     log)) {
            return false;
        }
    }
    return true;
}

bool materialize_mpsrt_image_bridge_binding(const metal::mpsrt::MpsrtModel& model,
                                            MetalDeviceHandle device,
                                            GfxMpsrtValue value,
                                            GfxMpsrtStorageBridgeDirection direction,
                                            metal::mpsrt::MpsrtBoundBuffer& binding,
                                            std::vector<id<MTLTexture>>& transient_textures,
                                            std::vector<MpsrtImageBridgeCopy>& bridge_copies,
                                            std::string* error) {
    const auto* tensor = find_mpsrt_tensor(model, value);
    if (!tensor) {
        return set_error(error, "GFX MPSRT: image bridge tensor descriptor is missing");
    }
    if (!gfx_mpsrt_tensor_is_image(tensor->desc)) {
        return true;
    }
    if (binding.texture) {
        return true;
    }
    if (direction == GfxMpsrtStorageBridgeDirection::Unknown) {
        return true;
    }
    if (!binding.buffer) {
        return set_error(error, "GFX MPSRT: image bridge external buffer binding is null");
    }
    if (!gfx_mpsrt_image_bridge_supported(tensor->desc)) {
        std::ostringstream stream;
        stream << "GFX MPSRT: image bridge supports only static rank-4 f16/f32 image tensors"
               << " value=" << value
               << " rank=" << tensor->desc.rank
               << " storage=" << tensor->desc.storage
               << " flags=" << tensor->desc.flags;
        return set_error(error, stream.str());
    }
    GfxMpsrtStorageBridgeDesc bridge_desc{};
    if (!gfx_mpsrt_make_image_bridge_desc(value, tensor->desc, direction, bridge_desc)) {
        return set_error(error, "GFX MPSRT: image bridge storage contract is invalid");
    }

    id<MTLTexture> texture = new_mpsrt_image_texture(device, tensor->desc, error);
    if (!texture) {
        return false;
    }
    transient_textures.push_back(texture);
    const auto image_binding = metal::mpsrt::make_mpsrt_bound_image((__bridge void*)texture);
    bridge_copies.push_back({bridge_desc.direction, bridge_desc.value, bridge_desc.tensor, binding, image_binding});
    binding = image_binding;
    return true;
}

bool materialize_mpsrt_image_bridge_bindings(const metal::mpsrt::MpsrtModel& model,
                                             MetalDeviceHandle device,
                                             const std::vector<GfxMpsrtValue>& values,
                                             GfxMpsrtStorageBridgeDirection legacy_fallback_direction,
                                             std::vector<metal::mpsrt::MpsrtBoundBuffer>& bindings,
                                             std::vector<id<MTLTexture>>& transient_textures,
                                             std::vector<MpsrtImageBridgeCopy>& bridge_copies,
                                             std::string* error) {
    if (values.size() != bindings.size()) {
        return set_error(error, "GFX MPSRT: image bridge value count does not match binding count");
    }
    for (size_t i = 0; i < values.size(); ++i) {
        const auto direction =
            mpsrt_external_image_bridge_direction_for_value(model, values[i], legacy_fallback_direction);
        if (!materialize_mpsrt_image_bridge_binding(model,
                                                    device,
                                                    values[i],
                                                    direction,
                                                    bindings[i],
                                                    transient_textures,
                                                    bridge_copies,
                                                    error)) {
            return false;
        }
    }
    return true;
}

const char* mpsrt_image_bridge_entry_point(const GfxMpsrtTensorAbiDesc& desc,
                                           GfxMpsrtStorageBridgeDirection direction) {
    const bool f16 = desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::F16);
    if (direction == GfxMpsrtStorageBridgeDirection::BufferToImage) {
        return f16 ? "gfx_mpsrt_buffer_to_image_f16" : "gfx_mpsrt_buffer_to_image_f32";
    }
    return f16 ? "gfx_mpsrt_image_to_buffer_f16" : "gfx_mpsrt_image_to_buffer_f32";
}

const char* mpsrt_image_bridge_source() {
    return R"MSL(
#include <metal_stdlib>
using namespace metal;

struct GfxMpsrtImageBridgeParams {
    uint width;
    uint height;
    uint channels;
    uint batch;
};

inline uint gfx_mpsrt_image_bridge_slices(uint channels) {
    return (channels + 3u) / 4u;
}

inline uint gfx_mpsrt_image_bridge_nchw_index(constant GfxMpsrtImageBridgeParams& p,
                                              uint n,
                                              uint c,
                                              uint y,
                                              uint x) {
    return ((n * p.channels + c) * p.height + y) * p.width + x;
}

kernel void gfx_mpsrt_buffer_to_image_f32(device const float* src [[buffer(0)]],
                                          texture2d_array<float, access::write> dst [[texture(0)]],
                                          constant GfxMpsrtImageBridgeParams& p [[buffer(1)]],
                                          uint3 gid [[thread_position_in_grid]]) {
    const uint x = gid.x;
    const uint y = gid.y;
    const uint plane = gid.z;
    const uint slices = gfx_mpsrt_image_bridge_slices(p.channels);
    const uint n = plane / slices;
    const uint slice = plane - n * slices;
    if (x >= p.width || y >= p.height || n >= p.batch) {
        return;
    }
    float4 value = float4(0.0f);
    const uint c0 = slice * 4u;
    if (c0 + 0u < p.channels) value.x = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 0u, y, x)];
    if (c0 + 1u < p.channels) value.y = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 1u, y, x)];
    if (c0 + 2u < p.channels) value.z = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 2u, y, x)];
    if (c0 + 3u < p.channels) value.w = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 3u, y, x)];
    dst.write(value, uint2(x, y), plane);
}

kernel void gfx_mpsrt_buffer_to_image_f16(device const half* src [[buffer(0)]],
                                          texture2d_array<half, access::write> dst [[texture(0)]],
                                          constant GfxMpsrtImageBridgeParams& p [[buffer(1)]],
                                          uint3 gid [[thread_position_in_grid]]) {
    const uint x = gid.x;
    const uint y = gid.y;
    const uint plane = gid.z;
    const uint slices = gfx_mpsrt_image_bridge_slices(p.channels);
    const uint n = plane / slices;
    const uint slice = plane - n * slices;
    if (x >= p.width || y >= p.height || n >= p.batch) {
        return;
    }
    half4 value = half4(0.0h);
    const uint c0 = slice * 4u;
    if (c0 + 0u < p.channels) value.x = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 0u, y, x)];
    if (c0 + 1u < p.channels) value.y = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 1u, y, x)];
    if (c0 + 2u < p.channels) value.z = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 2u, y, x)];
    if (c0 + 3u < p.channels) value.w = src[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 3u, y, x)];
    dst.write(value, uint2(x, y), plane);
}

kernel void gfx_mpsrt_image_to_buffer_f32(texture2d_array<float, access::read> src [[texture(0)]],
                                          device float* dst [[buffer(0)]],
                                          constant GfxMpsrtImageBridgeParams& p [[buffer(1)]],
                                          uint3 gid [[thread_position_in_grid]]) {
    const uint x = gid.x;
    const uint y = gid.y;
    const uint plane = gid.z;
    const uint slices = gfx_mpsrt_image_bridge_slices(p.channels);
    const uint n = plane / slices;
    const uint slice = plane - n * slices;
    if (x >= p.width || y >= p.height || n >= p.batch) {
        return;
    }
    const float4 value = src.read(uint2(x, y), plane);
    const uint c0 = slice * 4u;
    if (c0 + 0u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 0u, y, x)] = value.x;
    if (c0 + 1u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 1u, y, x)] = value.y;
    if (c0 + 2u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 2u, y, x)] = value.z;
    if (c0 + 3u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 3u, y, x)] = value.w;
}

kernel void gfx_mpsrt_image_to_buffer_f16(texture2d_array<half, access::read> src [[texture(0)]],
                                          device half* dst [[buffer(0)]],
                                          constant GfxMpsrtImageBridgeParams& p [[buffer(1)]],
                                          uint3 gid [[thread_position_in_grid]]) {
    const uint x = gid.x;
    const uint y = gid.y;
    const uint plane = gid.z;
    const uint slices = gfx_mpsrt_image_bridge_slices(p.channels);
    const uint n = plane / slices;
    const uint slice = plane - n * slices;
    if (x >= p.width || y >= p.height || n >= p.batch) {
        return;
    }
    const half4 value = src.read(uint2(x, y), plane);
    const uint c0 = slice * 4u;
    if (c0 + 0u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 0u, y, x)] = value.x;
    if (c0 + 1u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 1u, y, x)] = value.y;
    if (c0 + 2u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 2u, y, x)] = value.z;
    if (c0 + 3u < p.channels) dst[gfx_mpsrt_image_bridge_nchw_index(p, n, c0 + 3u, y, x)] = value.w;
}
)MSL";
}

bool encode_mpsrt_image_bridge_copy(GpuCommandBufferHandle command_buffer,
                                    metal::mpsrt::MpsrtContext& context,
                                    const MpsrtImageBridgeCopy& copy,
                                    const KernelExecutionHooks* hooks,
                                    std::string* error) {
    if (!copy.buffer_binding.buffer || !copy.image_binding.texture) {
        return set_error(error, "GFX MPSRT: image bridge copy has incomplete resources");
    }

    metal::mpsrt::MpsrtRuntimeStage stage;
    stage.kind = GfxMpsrtStageKind::MSLDispatch;
    stage.stage_record_key = copy.direction == GfxMpsrtStorageBridgeDirection::BufferToImage
                                 ? "mpsrt_bridge_buffer_to_image"
                                 : "mpsrt_bridge_image_to_buffer";
    stage.dispatch_entry_point = mpsrt_image_bridge_entry_point(copy.desc, copy.direction);
    stage.dispatch_threads_per_threadgroup = 64;
    stage.dispatch_flags = GfxMpsrtMslDispatchFlagNone;
    bool cache_hit = false;
    id<MTLComputePipelineState> pipeline =
        context.get_or_create_pipeline(stage, mpsrt_image_bridge_source(), cache_hit, error);
    if (!pipeline) {
        return false;
    }

    bool encoder_created = false;
    id<MTLComputeCommandEncoder> encoder =
        static_cast<id<MTLComputeCommandEncoder>>(metal_get_or_create_compute_encoder(command_buffer, &encoder_created));
    if (!encoder) {
        return set_error(error, "GFX MPSRT: failed to create image bridge compute encoder");
    }
    metal_set_compute_pipeline_if_needed(command_buffer,
                                         reinterpret_cast<GpuCommandEncoderHandle>(encoder),
                                         (__bridge void*)pipeline);

    id<MTLBuffer> buffer = static_cast<id<MTLBuffer>>(copy.buffer_binding.buffer);
    id<MTLTexture> texture = static_cast<id<MTLTexture>>(copy.image_binding.texture);
    MpsrtImageBridgeParams params{};
    params.width = copy.desc.image_width;
    params.height = copy.desc.image_height;
    params.channels = copy.desc.image_feature_channels;
    params.batch = copy.desc.image_batch;
    [encoder setBuffer:buffer offset:copy.buffer_binding.offset atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    [encoder setTexture:texture atIndex:0];

    const NSUInteger slices = static_cast<NSUInteger>(mpsrt_image_slice_count(copy.desc.image_feature_channels));
    const MTLSize grid = MTLSizeMake(copy.desc.image_width,
                                     copy.desc.image_height,
                                     copy.desc.image_batch * slices);
    const MTLSize threads = MTLSizeMake(8, 8, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];

    if (hooks && hooks->on_counter) {
        hooks->on_counter(copy.direction == GfxMpsrtStorageBridgeDirection::BufferToImage
                              ? "mpsrt_image_bridge_buffer_to_image_encode_count"
                              : "mpsrt_image_bridge_image_to_buffer_encode_count",
                          1);
        hooks->on_counter(cache_hit ? "mpsrt_image_bridge_pipeline_cache_hit_count"
                                    : "mpsrt_image_bridge_pipeline_cache_miss_count",
                          1);
        if (encoder_created) {
            hooks->on_counter("mpsrt_image_bridge_encoder_create_count", 1);
        }
    }
    return true;
}

bool encode_mpsrt_image_bridge_copies(GpuCommandBufferHandle command_buffer,
                                      metal::mpsrt::MpsrtContext& context,
                                      const std::vector<MpsrtImageBridgeCopy>& copies,
                                      GfxMpsrtStorageBridgeDirection direction,
                                      const KernelExecutionHooks* hooks,
                                      std::string* error) {
    for (const auto& copy : copies) {
        if (copy.direction != direction) {
            continue;
        }
        if (!encode_mpsrt_image_bridge_copy(command_buffer, context, copy, hooks, error)) {
            return false;
        }
    }
    return true;
}

struct ResolvedMpsrtProgramPlan {
    bool valid = false;
    GfxMpsrtProgram program{};
    GfxMpsrtBuilderPlan builder_plan{};
};

bool module_has_mpsrt_program(mlir::ModuleOp module) {
    GfxMpsrtProgram program{};
    return module && read_module_mpsrt_program(module, program);
}

bool module_has_generated_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module) {
    if (!module) {
        return false;
    }
    module.getContext()->loadDialect<mlir::func::FuncDialect>();
    std::string plan_symbol = "gfx_mpsrt_runtime_abi_plan";
    if (auto attr = module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.runtime_abi.call_plan_symbol")) {
        plan_symbol = attr.str();
    }
    auto plan_func = module.lookupSymbol<mlir::func::FuncOp>(plan_symbol);
    return plan_func &&
           static_cast<bool>(plan_func->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.runtime_abi.generated"));
}

bool ensure_mpsrt_runtime_abi_call_plan(mlir::ModuleOp module) {
    if (!module_has_mpsrt_program(module)) {
        return false;
    }
    if (module_has_generated_mpsrt_runtime_abi_call_plan(module)) {
        return true;
    }

    mlir::PassManager pm(module.getContext());
    populate_gfx_apple_mpsrt_runtime_abi_pipeline(pm);
    if (mlir::failed(pm.run(module))) {
        OPENVINO_THROW("GFX Metal MPSRT: failed to materialize runtime ABI call plan");
    }
    return module_has_generated_mpsrt_runtime_abi_call_plan(module);
}

bool equal_mpsrt_u32_vectors(const std::vector<GfxMpsrtValue>& lhs,
                             const std::vector<GfxMpsrtValue>& rhs) {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool equal_mpsrt_external_roles(const std::vector<GfxMpsrtExternalBufferRole>& lhs,
                                const std::vector<GfxMpsrtExternalBufferRole>& rhs) {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool validate_mpsrt_runtime_abi_call_plan_matches_builder_plan(const GfxMpsrtBuilderPlan& builder_plan,
                                                               const GfxMpsrtBuilderPlan& call_plan,
                                                               std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error) {
            *error = message;
        }
        return false;
    };

    if (!builder_plan.valid || !call_plan.valid) {
        return fail("invalid builder or runtime ABI call plan");
    }
    if (builder_plan.stage_record_key != call_plan.stage_record_key) {
        return fail("runtime ABI call plan record key mismatch");
    }
    if (!equal_mpsrt_u32_vectors(builder_plan.input_values, call_plan.input_values) ||
        !equal_mpsrt_u32_vectors(builder_plan.output_values, call_plan.output_values)) {
        return fail("runtime ABI call plan external IO value mismatch");
    }
    if (builder_plan.external_buffer_abi_valid != call_plan.external_buffer_abi_valid ||
        builder_plan.external_buffer_count != call_plan.external_buffer_count ||
        builder_plan.external_output_buffer_count != call_plan.external_output_buffer_count ||
        !equal_mpsrt_external_roles(builder_plan.external_buffer_roles, call_plan.external_buffer_roles)) {
        return fail("runtime ABI call plan external buffer ABI mismatch");
    }
    if (builder_plan.records.size() != call_plan.records.size()) {
        return fail("runtime ABI call plan record count mismatch");
    }

    for (size_t i = 0; i < builder_plan.records.size(); ++i) {
        const auto& expected = builder_plan.records[i];
        const auto& actual = call_plan.records[i];
        if (expected.kind != actual.kind || expected.symbol != actual.symbol) {
            return fail("runtime ABI call plan record kind/symbol mismatch at " + std::to_string(i));
        }
        if (expected.kind == GfxMpsrtBuilderRecordKind::AddTensor && expected.value != actual.value) {
            return fail("runtime ABI call plan tensor value mismatch at " + std::to_string(i));
        }
        if (expected.kind != GfxMpsrtBuilderRecordKind::EncodeStage) {
            continue;
        }
        if (expected.stage_kind != actual.stage_kind ||
            expected.kernel_name != actual.kernel_name ||
            !equal_mpsrt_u32_vectors(expected.inputs, actual.inputs) ||
            !equal_mpsrt_u32_vectors(expected.outputs, actual.outputs) ||
            !equal_mpsrt_u32_vectors(expected.kernel_buffer_order, actual.kernel_buffer_order)) {
            return fail("runtime ABI call plan encode-stage contract mismatch at " + std::to_string(i));
        }
    }
    return true;
}

bool resolve_module_mpsrt_program_plan(mlir::ModuleOp module,
                                       ResolvedMpsrtProgramPlan& program_plan) {
    program_plan = {};
    if (!module) {
        return false;
    }

    GfxMpsrtProgram program{};
    if (!read_module_mpsrt_program(module, program)) {
        return false;
    }

    GfxMpsrtBuilderPlan builder_plan{};
    if (!gfx_mpsrt_build_builder_plan_from_program(program, builder_plan)) {
        return false;
    }

    const bool has_call_plan = ensure_mpsrt_runtime_abi_call_plan(module);
    GfxMpsrtBuilderPlan call_plan;
    const bool read_call_plan =
        has_call_plan && read_gfx_apple_mpsrt_runtime_abi_call_plan(module, call_plan);

    if (read_call_plan) {
        std::string error;
        if (!validate_mpsrt_runtime_abi_call_plan_matches_builder_plan(builder_plan,
                                                                       call_plan,
                                                                       &error)) {
            OPENVINO_THROW("GFX Metal MPSRT: runtime ABI call plan diverged from stage manifest: ", error);
        }
        builder_plan = std::move(call_plan);
        if (current_compile_trace()) {
            increment_compile_counter("mpsrt_runtime_abi_call_plan_consumed_count");
        }
    } else if (has_call_plan) {
        OPENVINO_THROW("GFX Metal MPSRT: generated runtime ABI call plan is not readable");
    }

    program_plan.valid = true;
    program_plan.program = std::move(program);
    program_plan.builder_plan = std::move(builder_plan);
    return program_plan.valid;
}

void record_mpsrt_plan_counters(mlir::ModuleOp module) {
    if (!module || !current_compile_trace()) {
        return;
    }

    ResolvedMpsrtProgramPlan program_plan;
    if (!resolve_module_mpsrt_program_plan(module, program_plan)) {
        return;
    }
    const auto& builder_plan = program_plan.builder_plan;
    const auto& program = program_plan.program;
    auto record_stage_counters = [](const GfxMpsrtStageDesc& stage) {
        switch (stage.domain) {
            case GfxStageBackendDomain::AppleMps:
                increment_compile_counter("mpsrt_plan_apple_mps_count");
                break;
            case GfxStageBackendDomain::AppleMsl:
                increment_compile_counter("mpsrt_plan_apple_msl_count");
                break;
            case GfxStageBackendDomain::Spirv:
                increment_compile_counter("mpsrt_plan_spirv_count");
                break;
            case GfxStageBackendDomain::Unknown:
            default:
                break;
        }
        increment_compile_counter(std::string("mpsrt_stage_kind_") + gfx_mpsrt_stage_kind_name(stage.kind));
        increment_compile_counter(std::string("mpsrt_builder_symbol_") + stage.builder_symbol + "_count");
        if (!stage.dispatch_kernel_family.empty()) {
            increment_compile_counter(std::string("mpsrt_dispatch_family_") +
                                      stage.dispatch_kernel_family +
                                      "_count");
        }
        if (stage.dispatch_kernel_family_id != 0) {
            increment_compile_counter(std::string("mpsrt_dispatch_family_id_") +
                                      std::to_string(stage.dispatch_kernel_family_id) +
                                      "_count");
        }
        if (!stage.dispatch_entry_point.empty()) {
            increment_compile_counter(std::string("mpsrt_dispatch_entry_") +
                                      stage.dispatch_entry_point +
                                      "_count");
        }
        if (stage.dispatch_threads_per_threadgroup != 0) {
            increment_compile_counter(std::string("mpsrt_dispatch_tg_") +
                                      std::to_string(stage.dispatch_threads_per_threadgroup) +
                                      "_count");
        }
        if (stage.dispatch_flags != GfxMpsrtMslDispatchFlagNone) {
            increment_compile_counter("mpsrt_dispatch_flags",
                                      static_cast<uint64_t>(stage.dispatch_flags));
        }
        if (stage.dispatch_precompiled_kernel_required) {
            increment_compile_counter("mpsrt_dispatch_precompiled_kernel_required_count");
        }
        increment_compile_counter(std::string("mpsrt_storage_") +
                                  gfx_mpsrt_storage_name(stage.output_storage) +
                                  "_count");
        if (stage.uses_vendor_primitive) {
            increment_compile_counter("mpsrt_vendor_primitive_stage_count");
        }
        if (stage.uses_custom_kernel) {
            increment_compile_counter("mpsrt_custom_kernel_stage_count");
        }
    };

    increment_compile_counter("mpsrt_builder_record_count", static_cast<uint64_t>(builder_plan.records.size()));
    increment_compile_counter("mpsrt_builder_storage_bridge_count",
                              static_cast<uint64_t>(builder_plan.storage_bridges.size()));
    uint64_t encode_record_count = 0;
    for (const auto& record : builder_plan.records) {
        if (record.kind == GfxMpsrtBuilderRecordKind::EncodeStage) {
            ++encode_record_count;
        }
        if (record.stage_kind == GfxMpsrtStageKind::MSLDispatch &&
            record.msl_dispatch_desc.kernel_family != 0) {
            increment_compile_counter("mpsrt_msl_dispatch_descriptor_count");
            increment_compile_counter(std::string("mpsrt_msl_dispatch_descriptor_family_id_") +
                                      std::to_string(record.msl_dispatch_desc.kernel_family) +
                                      "_count");
        }
    }
    increment_compile_counter("mpsrt_builder_encode_record_count", encode_record_count);

    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t output_descriptor_count = 0;
    if (program.multi_stage) {
        increment_compile_counter("mpsrt_multi_stage_module_plan_count");
        increment_compile_counter("mpsrt_multi_stage_module_stage_count",
                                  static_cast<uint64_t>(program.stages.size()));
        for (const auto& desc : program.inputs) {
            input_bytes += desc.byte_length;
        }
        for (const auto& stage : program.stages) {
            record_stage_counters(stage.stage);
            output_descriptor_count += stage.output_descs.size();
            for (const auto& desc : stage.output_descs) {
                output_bytes += desc.byte_length;
            }
        }
    } else if (!program.stages.empty()) {
        const auto& stage = program.stages.front();
        record_stage_counters(stage.stage);
        for (const auto& desc : program.inputs) {
            input_bytes += desc.byte_length;
        }
        output_descriptor_count = stage.output_descs.size();
        for (const auto& desc : stage.output_descs) {
            output_bytes += desc.byte_length;
        }
    }
    increment_compile_counter("mpsrt_input_descriptor_count",
                              static_cast<uint64_t>(builder_plan.input_values.size()));
    increment_compile_counter("mpsrt_output_descriptor_count", output_descriptor_count);
    increment_compile_counter("mpsrt_input_byte_length", input_bytes);
    increment_compile_counter("mpsrt_output_byte_length", output_bytes);
}

std::shared_ptr<const metal::mpsrt::MpsrtModel> build_metal_mpsrt_runtime_model(mlir::ModuleOp module,
                                                                               uint32_t arg_count,
                                                                               uint32_t output_arg_count) {
    if (!module) {
        return nullptr;
    }

    ResolvedMpsrtProgramPlan program_plan;
    if (!resolve_module_mpsrt_program_plan(module, program_plan)) {
        return nullptr;
    }

    metal::mpsrt::MpsrtModel model;
    std::string error;
    if (!metal::mpsrt::build_mpsrt_model_from_builder_plan(program_plan.builder_plan, model, &error)) {
        OPENVINO_THROW("GFX Metal MPSRT: failed to build runtime model: ", error);
    }
    const uint32_t mpsrt_arg_count = program_plan.builder_plan.external_buffer_count != 0
                                         ? program_plan.builder_plan.external_buffer_count
                                         : arg_count;
    const uint32_t mpsrt_output_arg_count = program_plan.builder_plan.external_buffer_abi_valid
                                                ? program_plan.builder_plan.external_output_buffer_count
                                                : output_arg_count;
    if (!metal::mpsrt::adapt_mpsrt_model_to_external_buffer_abi(model,
                                                                mpsrt_arg_count,
                                                                mpsrt_output_arg_count,
                                                                &error)) {
        OPENVINO_THROW("GFX Metal MPSRT: failed to adapt runtime model ABI: ", error);
    }

    if (current_compile_trace()) {
        increment_compile_counter("mpsrt_runtime_model_prepare_count");
        if (program_plan.builder_plan.external_buffer_abi_valid) {
            increment_compile_counter("mpsrt_runtime_model_mlir_external_buffer_abi_count");
        }
        increment_compile_counter("mpsrt_runtime_model_stage_count", static_cast<uint64_t>(model.stages.size()));
        increment_compile_counter("mpsrt_runtime_model_tensor_count", static_cast<uint64_t>(model.tensors.size()));
        for (const auto& stage : model.stages) {
            increment_compile_counter(std::string("mpsrt_runtime_model_stage_kind_") +
                                      gfx_mpsrt_stage_kind_name(stage.kind) +
                                      "_count");
            if (stage.kind == GfxMpsrtStageKind::MSLDispatch) {
                increment_compile_counter("mpsrt_runtime_model_msl_dispatch_stage_count");
            }
        }
    }

    return std::make_shared<metal::mpsrt::MpsrtModel>(std::move(model));
}

class MetalResolvedMslCache final {
public:
    static MetalResolvedMslCache& instance() {
        static MetalResolvedMslCache cache;
        return cache;
    }

    bool lookup(const std::string& key, std::string& msl) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache.find(key);
        if (it == m_cache.end()) {
            return false;
        }
        msl = it->second;
        return true;
    }

    void store(std::string key, std::string msl) {
        if (key.empty() || msl.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.emplace(std::move(key), std::move(msl));
    }

private:
    std::mutex m_mutex;
    std::unordered_map<std::string, std::string> m_cache;
};

class MetalPreparedState final {
public:
    explicit MetalPreparedState(const KernelBindingTable& table) {
        const auto& bindings = table.buffers;
        buffers.reserve(bindings.size());
        buffer_ptrs.reserve(bindings.size());
        offsets.reserve(bindings.size());
        for (const auto& binding : bindings) {
            auto* buffer = to_mtl(binding.buffer);
            buffers.push_back(buffer);
            buffer_ptrs.push_back(buffer);
            offsets.push_back(binding.offset);
        }
    }

    std::vector<id<MTLBuffer>> buffers;
    std::vector<void*> buffer_ptrs;
    std::vector<size_t> offsets;
};

bool mpsrt_model_has_msl_dispatch(const std::shared_ptr<const metal::mpsrt::MpsrtModel>& model) {
    if (!model) {
        return false;
    }
    return std::any_of(model->stages.begin(), model->stages.end(), [](const auto& stage) {
        return stage.kind == GfxMpsrtStageKind::MSLDispatch;
    });
}

bool mpsrt_conv2d_stage_supported_by_image_bridge(const metal::mpsrt::MpsrtModel& model,
                                                  const metal::mpsrt::MpsrtRuntimeStage& stage) {
    if (stage.inputs.size() != 2 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return false;
    }
    const auto* input = find_mpsrt_tensor(model, stage.inputs[0]);
    const auto* weights = find_mpsrt_tensor(model, stage.inputs[1]);
    if (!input || !weights) {
        return false;
    }
    const auto& input_desc = input->desc;
    const auto& weights_desc = weights->desc;
    const auto& output_desc = stage.output_descs.front();
    if (!gfx_mpsrt_tensor_is_image(input_desc) ||
        !gfx_mpsrt_tensor_is_image(output_desc) ||
        !gfx_mpsrt_image_bridge_supported(input_desc) ||
        !gfx_mpsrt_image_bridge_supported(output_desc)) {
        return false;
    }
    if (input_desc.dtype != output_desc.dtype || weights_desc.dtype != output_desc.dtype) {
        return false;
    }
    if (stage.conv2d_desc.groups == 0 ||
        input_desc.image_feature_channels % stage.conv2d_desc.groups != 0 ||
        output_desc.image_feature_channels % stage.conv2d_desc.groups != 0) {
        return false;
    }
    return true;
}

bool mpsrt_pool2d_stage_supported_by_image_bridge(const metal::mpsrt::MpsrtModel& model,
                                                  const metal::mpsrt::MpsrtRuntimeStage& stage) {
    if (stage.inputs.size() != 1 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return false;
    }
    const auto* input = find_mpsrt_tensor(model, stage.inputs[0]);
    if (!input) {
        return false;
    }
    const auto& input_desc = input->desc;
    const auto& output_desc = stage.output_descs.front();
    if (!gfx_mpsrt_tensor_is_image(input_desc) ||
        !gfx_mpsrt_tensor_is_image(output_desc) ||
        !gfx_mpsrt_image_bridge_supported(input_desc) ||
        !gfx_mpsrt_image_bridge_supported(output_desc)) {
        return false;
    }
    if (input_desc.dtype != output_desc.dtype ||
        input_desc.image_batch != output_desc.image_batch ||
        input_desc.image_feature_channels != output_desc.image_feature_channels) {
        return false;
    }
    if ((input_desc.image_feature_channels % 4u) != 0) {
        return false;
    }
    if (stage.pool2d_desc.kernel[0] == 0 || stage.pool2d_desc.kernel[1] == 0 ||
        stage.pool2d_desc.strides[0] == 0 || stage.pool2d_desc.strides[1] == 0 ||
        stage.pool2d_desc.dilations[0] != 1 || stage.pool2d_desc.dilations[1] != 1) {
        return false;
    }
    return true;
}

bool mpsrt_softmax_stage_supported_by_matrix_bridge(const metal::mpsrt::MpsrtModel& model,
                                                    const metal::mpsrt::MpsrtRuntimeStage& stage) {
    if (stage.inputs.size() != 1 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return false;
    }
    const auto* input = find_mpsrt_tensor(model, stage.inputs[0]);
    if (!input) {
        return false;
    }
    const auto& input_desc = input->desc;
    const auto& output_desc = stage.output_descs.front();
    if (input_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix) ||
        output_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
        return false;
    }
    if (stage.softmax_desc.log_softmax != 0) {
        return false;
    }
    if (input_desc.dtype != output_desc.dtype ||
        (input_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16) &&
         input_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32))) {
        return false;
    }
    if (input_desc.matrix_rows == 0 || input_desc.matrix_columns == 0 ||
        input_desc.matrix_row_bytes == 0 ||
        output_desc.matrix_rows != input_desc.matrix_rows ||
        output_desc.matrix_columns != input_desc.matrix_columns ||
        output_desc.matrix_row_bytes == 0) {
        return false;
    }
    const uint32_t input_count = input_desc.matrix_count == 0 ? 1 : input_desc.matrix_count;
    const uint32_t output_count = output_desc.matrix_count == 0 ? 1 : output_desc.matrix_count;
    return input_count == output_count;
}

bool mpsrt_topk_stage_supported_by_matrix_bridge(const metal::mpsrt::MpsrtModel& model,
                                                 const metal::mpsrt::MpsrtRuntimeStage& stage) {
    if (stage.inputs.size() != 1 || stage.outputs.size() != 2 || stage.output_descs.size() != 2) {
        return false;
    }
    const auto* input = find_mpsrt_tensor(model, stage.inputs[0]);
    if (!input) {
        return false;
    }
    const auto& input_desc = input->desc;
    const auto& values_desc = stage.output_descs[0];
    const auto& indices_desc = stage.output_descs[1];
    if (input_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix) ||
        values_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix) ||
        indices_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
        return false;
    }
    if (stage.topk_desc.mode_max == 0 || stage.topk_desc.k == 0 || stage.topk_desc.k > 16) {
        return false;
    }
    if (input_desc.dtype != values_desc.dtype ||
        (input_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16) &&
         input_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32))) {
        return false;
    }
    if (indices_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::I32) &&
        indices_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::U32)) {
        return false;
    }
    if (input_desc.matrix_rows == 0 || input_desc.matrix_columns == 0 ||
        input_desc.matrix_row_bytes == 0 ||
        values_desc.matrix_rows != input_desc.matrix_rows ||
        values_desc.matrix_columns != stage.topk_desc.k ||
        values_desc.matrix_row_bytes == 0 ||
        indices_desc.matrix_rows != input_desc.matrix_rows ||
        indices_desc.matrix_columns != stage.topk_desc.k ||
        indices_desc.matrix_row_bytes == 0) {
        return false;
    }
    const uint32_t input_count = input_desc.matrix_count == 0 ? 1 : input_desc.matrix_count;
    const uint32_t values_count = values_desc.matrix_count == 0 ? 1 : values_desc.matrix_count;
    const uint32_t indices_count = indices_desc.matrix_count == 0 ? 1 : indices_desc.matrix_count;
    return input_count == values_count && input_count == indices_count;
}

bool mpsrt_stage_supported_by_current_runtime(const metal::mpsrt::MpsrtModel& model,
                                              const metal::mpsrt::MpsrtRuntimeStage& stage) {
    switch (stage.kind) {
        case GfxMpsrtStageKind::MPSGemm:
            return true;
        case GfxMpsrtStageKind::MPSConv2D:
        case GfxMpsrtStageKind::MPSGroupConv2D:
            return mpsrt_conv2d_stage_supported_by_image_bridge(model, stage);
        case GfxMpsrtStageKind::MPSPool2D:
            return mpsrt_pool2d_stage_supported_by_image_bridge(model, stage);
        case GfxMpsrtStageKind::MPSSoftmax:
            return mpsrt_softmax_stage_supported_by_matrix_bridge(model, stage);
        case GfxMpsrtStageKind::MPSTopK:
            return mpsrt_topk_stage_supported_by_matrix_bridge(model, stage);
        default:
            return false;
    }
}

bool mpsrt_model_has_supported_vendor_stage(const std::shared_ptr<const metal::mpsrt::MpsrtModel>& model) {
    if (!model) {
        return false;
    }
    return std::any_of(model->stages.begin(), model->stages.end(), [&](const auto& stage) {
        return mpsrt_stage_supported_by_current_runtime(*model, stage);
    });
}

bool mpsrt_model_is_executable_by_mpsrt(const std::shared_ptr<const metal::mpsrt::MpsrtModel>& model,
                                        const std::string& msl_source) {
    if (!model || model->stages.empty()) {
        return false;
    }

    bool has_msl_dispatch = false;
    for (const auto& stage : model->stages) {
        switch (stage.kind) {
            case GfxMpsrtStageKind::MPSGemm:
            case GfxMpsrtStageKind::MPSConv2D:
            case GfxMpsrtStageKind::MPSGroupConv2D:
            case GfxMpsrtStageKind::MPSPool2D:
            case GfxMpsrtStageKind::MPSSoftmax:
            case GfxMpsrtStageKind::MPSTopK:
                if (!mpsrt_stage_supported_by_current_runtime(*model, stage)) {
                    return false;
                }
                break;
            case GfxMpsrtStageKind::MSLDispatch:
                has_msl_dispatch = true;
                break;
            default:
                return false;
        }
    }
    return !has_msl_dispatch || !msl_source.empty();
}

bool mpsrt_model_should_use_context_execution(const std::shared_ptr<const metal::mpsrt::MpsrtModel>& model,
                                              const std::string& msl_source) {
    if (!mpsrt_model_is_executable_by_mpsrt(model, msl_source)) {
        return false;
    }

    const bool has_vendor_stage = mpsrt_model_has_supported_vendor_stage(model);
    const bool has_msl_dispatch = mpsrt_model_has_msl_dispatch(model);
    if (!has_vendor_stage) {
        return false;
    }

    return !has_msl_dispatch || model->stages.size() > 1;
}

bool build_mpsrt_bindings_from_prepared_state(const metal::mpsrt::MpsrtModel& model,
                                              const KernelBindingPlan& binding_plan,
                                              MetalDeviceHandle device,
                                              const MetalPreparedState& prepared,
                                              metal::mpsrt::MpsrtTensorBindings& bindings,
                                              metal::mpsrt::MpsrtBindingBuildResult& binding_result,
                                              std::vector<id<MTLBuffer>>& transient_buffers,
                                              std::vector<id<MTLTexture>>& transient_textures,
                                              std::vector<MpsrtImageBridgeCopy>& image_bridge_copies,
                                              std::string& error) {
    std::vector<metal::mpsrt::MpsrtBoundBuffer> input_buffers;
    std::vector<metal::mpsrt::MpsrtBoundBuffer> output_buffers;
    auto external_buffers = metal::mpsrt::make_mpsrt_bound_buffers(prepared.buffer_ptrs,
                                                                   prepared.offsets);

    auto transient_allocator = [&](const metal::mpsrt::MpsrtRuntimeTensor& tensor) {
        if (tensor.desc.storage == static_cast<uint32_t>(GfxMpsrtStorage::Image)) {
            id<MTLTexture> texture = new_mpsrt_image_texture(device, tensor.desc, &error);
            if (!texture) {
                return metal::mpsrt::MpsrtBoundBuffer{};
            }
            transient_textures.push_back(texture);
            return metal::mpsrt::make_mpsrt_bound_image((__bridge void*)texture);
        }
        const auto byte_length = static_cast<NSUInteger>(tensor.desc.byte_length);
        if (byte_length == 0) {
            return metal::mpsrt::MpsrtBoundBuffer{};
        }
        id<MTLBuffer> buffer =
            [static_cast<id<MTLDevice>>(device) newBufferWithLength:byte_length
                                                            options:MTLResourceStorageModePrivate];
        transient_buffers.push_back(buffer);
        return metal::mpsrt::MpsrtBoundBuffer{(__bridge void*)buffer,
                                              static_cast<size_t>(tensor.desc.byte_offset)};
    };

    if (external_buffers.size() == model.external_values.size()) {
        std::vector<GfxMpsrtValue> external_output_values = model.external_output_values;
        if (external_output_values.empty()) {
            external_output_values = model.output_values;
        }
        for (size_t i = 0; i < model.external_values.size(); ++i) {
            const auto fallback_direction =
                gfx_mpsrt_external_image_bridge_direction(has_mpsrt_value(external_output_values,
                                                                          model.external_values[i]));
            const auto direction =
                mpsrt_external_image_bridge_direction_for_value(model,
                                                               model.external_values[i],
                                                               fallback_direction);
            if (!materialize_mpsrt_image_bridge_binding(model,
                                                        device,
                                                        model.external_values[i],
                                                        direction,
                                                        external_buffers[i],
                                                        transient_textures,
                                                        image_bridge_copies,
                                                        &error)) {
                return false;
            }
        }
        return metal::mpsrt::build_mpsrt_external_tensor_bindings(model,
                                                                  external_buffers,
                                                                  transient_allocator,
                                                                  bindings,
                                                                  &binding_result,
                                                                  &error);
    }

    if (!make_mpsrt_external_io_bindings(model,
                                         prepared.buffer_ptrs,
                                         prepared.offsets,
                                         binding_plan.output_arg_count(),
                                         input_buffers,
                                         output_buffers,
                                         &error)) {
        return false;
    }
    if (!materialize_mpsrt_image_bridge_bindings(model,
                                                device,
                                                model.input_values,
                                                GfxMpsrtStorageBridgeDirection::BufferToImage,
                                                input_buffers,
                                                transient_textures,
                                                image_bridge_copies,
                                                &error) ||
        !materialize_mpsrt_image_bridge_bindings(model,
                                                device,
                                                model.output_values,
                                                GfxMpsrtStorageBridgeDirection::ImageToBuffer,
                                                output_buffers,
                                                transient_textures,
                                                image_bridge_copies,
                                                &error)) {
        return false;
    }
    return metal::mpsrt::build_mpsrt_tensor_bindings(model,
                                                     input_buffers,
                                                     output_buffers,
                                                     transient_allocator,
                                                     bindings,
                                                     &binding_result,
                                                     &error);
}

}  // namespace

MetalCodegenBackend::MetalCodegenBackend(MetalDeviceHandle device)
    : m_device(device),
      m_reuse_context(MetalDeviceReuseRegistry::instance().acquire(device)) {}

std::shared_ptr<ICompiledKernel> MetalCodegenBackend::compile(const KernelSource& source,
                                                              std::string* log) {
    std::string local_log;
    std::string* log_ptr = log ? log : &local_log;
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MetalCodegen") << "compile entry=" << source.entry_point
                                       << " arg_count=" << source.signature.arg_count
                                       << " has_module=" << (source.module ? "yes" : "no")
                                       << " has_msl=" << (!source.msl_source.empty() ? "yes" : "no")
                                       << " has_generator=" << (source.msl_generator ? "yes" : "no");
    }
    std::string msl;
    std::string resolved_msl_cache_key;
    const bool can_cache_resolved_msl = source.module && source.msl_source.empty() && source.msl_generator;
    record_mpsrt_plan_counters(source.module);
    const uint32_t arg_count = source.signature.arg_count;
    const uint32_t output_arg_count = resolve_kernel_output_arg_count(source);
    auto mpsrt_model = build_metal_mpsrt_runtime_model(source.module, arg_count, output_arg_count);
    const uintptr_t device_key = reinterpret_cast<uintptr_t>(m_device);
    auto shared_prepared_cache = acquire_shared_prepared_binding_cache(GpuBackend::Metal, device_key, arg_count);
    auto binding_schema = m_reuse_context->acquire_binding_schema(arg_count);
    const bool vendor_only_mpsrt_model = mpsrt_model &&
                                         mpsrt_model_has_supported_vendor_stage(mpsrt_model) &&
                                         !mpsrt_model_has_msl_dispatch(mpsrt_model);
    if (vendor_only_mpsrt_model && source.msl_source.empty() && !source.msl_generator) {
        if (current_compile_trace()) {
            increment_compile_counter("metal_mpsrt_vendor_only_kernel_count");
        }
        auto binding_plan = std::make_shared<KernelBindingPlan>(arg_count, output_arg_count);
        auto kernel = std::make_shared<MetalCompiledKernel>(m_device,
                                                            nullptr,
                                                            std::move(binding_plan),
                                                            shared_prepared_cache,
                                                            binding_schema);
        kernel->set_mpsrt_model(mpsrt_model);
        if (!source.mpsrt_const_tensors.empty() &&
            !register_mpsrt_const_tensor_sources(*kernel, *mpsrt_model, source.mpsrt_const_tensors, log_ptr)) {
            return nullptr;
        }
        return kernel;
    }
    if (can_cache_resolved_msl) {
        const auto cache_key_start = current_compile_trace() ? std::chrono::steady_clock::now()
                                                             : std::chrono::steady_clock::time_point{};
        resolved_msl_cache_key = make_resolved_msl_cache_key(source);
        if (current_compile_trace()) {
            add_compile_segment(
                "metal_resolved_msl_cache_key",
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - cache_key_start)
                                          .count()));
        }
        if (MetalResolvedMslCache::instance().lookup(resolved_msl_cache_key, msl)) {
            if (current_compile_trace()) {
                increment_compile_counter("metal_resolved_msl_cache_hit_count");
                add_compile_segment("metal_resolved_msl_cache_hit", 0);
            }
        }
    }

    if (msl.empty() && source.module && source.msl_source.empty() && source.msl_generator) {
        const auto mlir_preprocess_start = current_compile_trace() ? std::chrono::steady_clock::now()
                                                                   : std::chrono::steady_clock::time_point{};
        try {
            if (gfx_log_debug_enabled()) {
                gfx_log_debug("MetalCodegen") << "before run_mlir_pipeline entry=" << source.entry_point;
            }
            run_mlir_pipeline(source.module, /*use_alloca=*/true, /*use_parallel_loops=*/false);
            if (gfx_log_debug_enabled()) {
                gfx_log_debug("MetalCodegen") << "after run_mlir_pipeline entry=" << source.entry_point;
            }
            if (current_compile_trace()) {
                increment_compile_counter("metal_mlir_preprocess_count");
                add_compile_segment(
                    "metal_mlir_preprocess",
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - mlir_preprocess_start)
                                              .count()));
            }
        } catch (const std::exception& e) {
            if (log_ptr) {
                *log_ptr = std::string("MLIR preprocessing failed: ") + e.what();
            }
            return nullptr;
        }
    }
    if (msl.empty()) {
        const auto resolve_msl_start =
            current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MetalCodegen") << "before resolve_msl_source entry=" << source.entry_point;
        }
        msl = resolve_msl_source(source, log_ptr);
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MetalCodegen") << "after resolve_msl_source entry=" << source.entry_point
                                           << " msl_size=" << msl.size();
        }
        if (current_compile_trace()) {
            increment_compile_counter("metal_resolve_msl_count");
            add_compile_segment(
                "metal_resolve_msl",
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - resolve_msl_start)
                                          .count()));
        }
        if (can_cache_resolved_msl && !resolved_msl_cache_key.empty()) {
            MetalResolvedMslCache::instance().store(std::move(resolved_msl_cache_key), msl);
        }
    }
    OPENVINO_ASSERT(!msl.empty(), "MetalCodegenBackend: missing MSL source");
    OPENVINO_ASSERT(!source.entry_point.empty(), "MetalCodegenBackend: missing entry point");

    auto kernel = lookup_or_compile_kernel(GpuBackend::Metal,
                                           device_key,
                                           msl.data(),
                                           msl.size(),
                                           source.entry_point,
                                           arg_count,
                                           [&]() -> std::shared_ptr<ICompiledKernel> {
                                               MetalKernelCompiler compiler((id<MTLDevice>)m_device);
                                               std::string local_log;
                                               std::string& compile_log = log ? *log : local_log;
                                               const auto backend_compile_start =
                                                   current_compile_trace()
                                                       ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point{};
                                               id<MTLComputePipelineState> pipeline =
                                                   compiler.compile_msl_from_source(msl,
                                                                                    source.entry_point.c_str(),
                                                                                    compile_log);
                                               if (current_compile_trace()) {
                                                   increment_compile_counter("metal_backend_compile_count");
                                                   add_compile_segment(
                                                       "metal_backend_compile",
                                                       static_cast<uint64_t>(
                                                           std::chrono::duration_cast<std::chrono::microseconds>(
                                                               std::chrono::steady_clock::now() -
                                                               backend_compile_start)
                                                               .count()));
                                               }
                                               if (!pipeline) {
                                                   return nullptr;
                                               }
                                               auto binding_plan = std::make_shared<KernelBindingPlan>(
                                                   arg_count,
                                                   output_arg_count);
                                               return std::make_shared<MetalCompiledKernel>(m_device,
                                                                                            (void*)pipeline,
                                                                                            std::move(binding_plan),
                                                                                            shared_prepared_cache,
                                                                                            binding_schema);
                                           });
    const bool compiled_model_has_msl_dispatch = mpsrt_model_has_msl_dispatch(mpsrt_model);
    if (auto metal_kernel = std::dynamic_pointer_cast<MetalCompiledKernel>(kernel)) {
        metal_kernel->set_mpsrt_model(mpsrt_model);
        if (mpsrt_model &&
            !source.mpsrt_const_tensors.empty() &&
            !register_mpsrt_const_tensor_sources(*metal_kernel,
                                                 *mpsrt_model,
                                                 source.mpsrt_const_tensors,
                                                 log_ptr)) {
            return nullptr;
        }
        if (compiled_model_has_msl_dispatch) {
            metal_kernel->set_mpsrt_msl_source(msl);
        }
    }
    return kernel;
}

MetalCompiledKernel::MetalCompiledKernel(MetalDeviceHandle device, void* pipeline, uint32_t arg_count)
    : CompiledKernelBase(arg_count), m_device(device), m_pipeline(pipeline) {}

MetalCompiledKernel::MetalCompiledKernel(MetalDeviceHandle device,
                                         void* pipeline,
                                         std::shared_ptr<const KernelBindingPlan> binding_plan)
    : CompiledKernelBase(std::move(binding_plan)), m_device(device), m_pipeline(pipeline) {}

MetalCompiledKernel::MetalCompiledKernel(MetalDeviceHandle device,
                                         void* pipeline,
                                         std::shared_ptr<const KernelBindingPlan> binding_plan,
                                         std::shared_ptr<void> prepared_binding_cache,
                                         std::shared_ptr<MetalBindingSchema> binding_schema)
    : CompiledKernelBase(std::move(binding_plan), std::move(prepared_binding_cache)),
      m_device(device),
      m_pipeline(pipeline),
      m_binding_schema(std::move(binding_schema)) {}

size_t MetalCompiledKernel::clamp_threadgroup_size(size_t desired) const {
    return metal_clamp_tg_size(m_pipeline, desired);
}

std::shared_ptr<ICompiledKernel> MetalCompiledKernel::fork() const {
    auto kernel = std::make_shared<MetalCompiledKernel>(m_device,
                                                        m_pipeline,
                                                        binding_plan(),
                                                        prepared_binding_cache(),
                                                        m_binding_schema);
    kernel->set_mpsrt_model(m_mpsrt_model);
    kernel->set_mpsrt_msl_source(m_mpsrt_msl_source);
    return kernel;
}

const void* MetalCompiledKernel::shared_binding_schema_identity() const {
    return m_binding_schema.get();
}

void MetalCompiledKernel::set_mpsrt_model(std::shared_ptr<const metal::mpsrt::MpsrtModel> model) {
    m_mpsrt_model = std::move(model);
}

void MetalCompiledKernel::set_mpsrt_msl_source(std::string msl_source) {
    m_mpsrt_msl_source = std::move(msl_source);
}

const metal::mpsrt::MpsrtModel* MetalCompiledKernel::mpsrt_model() const {
    return m_mpsrt_model.get();
}

bool MetalCompiledKernel::register_mpsrt_const_tensor_data(GfxMpsrtValue value,
                                                           GfxMpsrtTensorAbiDesc desc,
                                                           const void* data,
                                                           size_t bytes,
                                                           std::string* log) {
    if (!m_mpsrt_model) {
        return set_error(log, "GFX MPSRT: cannot register const tensor without an MPSRT model");
    }
    if (!m_mpsrt_context) {
        m_mpsrt_context = std::make_shared<metal::mpsrt::MpsrtContext>(static_cast<id<MTLDevice>>(m_device));
    }
    desc.flags |= GfxMpsrtTensorFlagConst;
    return m_mpsrt_context->register_const_tensor_data(value, desc, data, bytes, log);
}

void MetalCompiledKernel::prewarm_bindings(const std::vector<KernelArg>& args) {
    auto prepared_base = get_or_create_prepared_bindings(args, "MetalCompiledKernel prewarm");
    (void)prepared_base->get_or_create_backend_state<MetalPreparedState>(
        reinterpret_cast<uintptr_t>(m_binding_schema.get() ? m_binding_schema.get() : m_device),
        [&]() {
            return std::make_shared<MetalPreparedState>(prepared_base->binding_table());
        });
}

void MetalCompiledKernel::execute(GpuCommandBufferHandle command_buffer,
                                  const KernelDispatch& dispatch,
                                  const std::vector<KernelArg>& args,
                                  const KernelExecutionHooks* hooks) {
    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(command_buffer);
    OPENVINO_ASSERT(cb, "MetalCompiledKernel: command buffer is null");
    auto prepared_base = get_or_create_prepared_bindings(args, "MetalCompiledKernel");
    const bool trace_bindings = hooks && (hooks->on_segment || hooks->on_counter);
    const auto binding_start = trace_bindings ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool prepared_state_created = false;
    auto prepared = prepared_base->get_or_create_backend_state<MetalPreparedState>(
        reinterpret_cast<uintptr_t>(m_binding_schema.get() ? m_binding_schema.get() : m_device),
        [&]() {
            prepared_state_created = true;
            return std::make_shared<MetalPreparedState>(prepared_base->binding_table());
        });
    if (trace_bindings && prepared_state_created) {
        const auto binding_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - binding_start);
        if (hooks->on_counter) {
            hooks->on_counter("binding_prepare_count", 1);
        }
        if (hooks->on_segment) {
            hooks->on_segment("binding_prepare",
                              "metal_prepared_state",
                              binding_cpu_us,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -1,
                              0,
                              reinterpret_cast<uint64_t>(cb));
        }
    } else if (hooks && hooks->on_counter) {
        hooks->on_counter("prepared_binding_cache_hit_count", 1);
    }

    if (mpsrt_model_should_use_context_execution(m_mpsrt_model, m_mpsrt_msl_source)) {
        std::string mpsrt_error;
        std::vector<id<MTLBuffer>> transient_buffers;
        std::vector<id<MTLTexture>> transient_textures;
        std::vector<MpsrtImageBridgeCopy> image_bridge_copies;
        metal::mpsrt::MpsrtTensorBindings bindings;
        metal::mpsrt::MpsrtBindingBuildResult binding_result;
        const bool bindings_built =
            build_mpsrt_bindings_from_prepared_state(*m_mpsrt_model,
                                                     *binding_plan(),
                                                     m_device,
                                                     *prepared,
                                                     bindings,
                                                     binding_result,
                                                     transient_buffers,
                                                     transient_textures,
                                                     image_bridge_copies,
                                                     mpsrt_error);
        OPENVINO_ASSERT(bindings_built, mpsrt_error);
        if (hooks && hooks->on_counter) {
            hooks->on_counter("mpsrt_binding_external_input_count",
                              static_cast<uint64_t>(binding_result.external_inputs_bound));
            hooks->on_counter("mpsrt_binding_external_output_count",
                              static_cast<uint64_t>(binding_result.external_outputs_bound));
            hooks->on_counter("mpsrt_binding_transient_alloc_count",
                              static_cast<uint64_t>(binding_result.transient_buffers_allocated));
            hooks->on_counter("mpsrt_binding_transient_image_alloc_count",
                              static_cast<uint64_t>(binding_result.transient_images_allocated));
            hooks->on_counter("mpsrt_image_bridge_copy_count",
                              static_cast<uint64_t>(image_bridge_copies.size()));
        }

        if (!m_mpsrt_context) {
            m_mpsrt_context = std::make_shared<metal::mpsrt::MpsrtContext>(static_cast<id<MTLDevice>>(m_device));
        }
        metal::mpsrt::MpsrtPreparedModel prepared_model;
        OPENVINO_ASSERT(m_mpsrt_context->prepare_model(*m_mpsrt_model,
                                                       m_mpsrt_msl_source,
                                                       prepared_model,
                                                       &mpsrt_error),
                        mpsrt_error);
        std::vector<KernelDispatch> stage_dispatches(m_mpsrt_model->stages.size(), dispatch);
        metal::mpsrt::MpsrtRequest request;
        metal::mpsrt::MpsrtModelEncodeResult encode_result;
        OPENVINO_ASSERT(encode_mpsrt_image_bridge_copies(command_buffer,
                                                         *m_mpsrt_context,
                                                         image_bridge_copies,
                                                         GfxMpsrtStorageBridgeDirection::BufferToImage,
                                                         hooks,
                                                         &mpsrt_error),
                        mpsrt_error);
        const bool encoded =
            request.encode_prepared_model(command_buffer,
                                          *m_mpsrt_model,
                                          prepared_model,
                                          stage_dispatches,
                                          bindings,
                                          hooks,
                                          &encode_result,
                                          &mpsrt_error);
        OPENVINO_ASSERT(encoded, mpsrt_error);
        OPENVINO_ASSERT(encode_mpsrt_image_bridge_copies(command_buffer,
                                                         *m_mpsrt_context,
                                                         image_bridge_copies,
                                                         GfxMpsrtStorageBridgeDirection::ImageToBuffer,
                                                         hooks,
                                                         &mpsrt_error),
                        mpsrt_error);
        return;
    }

    if (m_mpsrt_model && m_mpsrt_model->stages.size() == 1 &&
        m_mpsrt_model->stages.front().kind == GfxMpsrtStageKind::MSLDispatch) {
        OPENVINO_ASSERT(m_pipeline, "MetalCompiledKernel: MSL MPSRT pipeline is null");
        const auto prepared_mpsrt =
            metal::mpsrt::make_prepared_msl_dispatch_from_pipeline(m_mpsrt_model->stages.front(),
                                                                    0,
                                                                    static_cast<id<MTLComputePipelineState>>(m_pipeline));
        std::string mpsrt_error;
        std::vector<id<MTLBuffer>> transient_buffers;
        std::vector<id<MTLTexture>> transient_textures;
        std::vector<MpsrtImageBridgeCopy> image_bridge_copies;
        metal::mpsrt::MpsrtTensorBindings bindings;
        metal::mpsrt::MpsrtBindingBuildResult binding_result;
        const bool bindings_built =
            build_mpsrt_bindings_from_prepared_state(*m_mpsrt_model,
                                                     *binding_plan(),
                                                     m_device,
                                                     *prepared,
                                                     bindings,
                                                     binding_result,
                                                     transient_buffers,
                                                     transient_textures,
                                                     image_bridge_copies,
                                                     mpsrt_error);
        OPENVINO_ASSERT(bindings_built, mpsrt_error);
        if (hooks && hooks->on_counter) {
            hooks->on_counter("mpsrt_binding_external_input_count",
                              static_cast<uint64_t>(binding_result.external_inputs_bound));
            hooks->on_counter("mpsrt_binding_external_output_count",
                              static_cast<uint64_t>(binding_result.external_outputs_bound));
            hooks->on_counter("mpsrt_binding_transient_alloc_count",
                              static_cast<uint64_t>(binding_result.transient_buffers_allocated));
            hooks->on_counter("mpsrt_binding_transient_image_alloc_count",
                              static_cast<uint64_t>(binding_result.transient_images_allocated));
            hooks->on_counter("mpsrt_image_bridge_copy_count",
                              static_cast<uint64_t>(image_bridge_copies.size()));
        }

        if (!m_mpsrt_context) {
            m_mpsrt_context = std::make_shared<metal::mpsrt::MpsrtContext>(static_cast<id<MTLDevice>>(m_device));
        }
        metal::mpsrt::MpsrtPreparedModel prepared_model;
        prepared_model.msl_dispatches.push_back(prepared_mpsrt);
        std::vector<KernelDispatch> stage_dispatches = {dispatch};
        metal::mpsrt::MpsrtRequest request;
        metal::mpsrt::MpsrtModelEncodeResult encode_result;
        OPENVINO_ASSERT(encode_mpsrt_image_bridge_copies(command_buffer,
                                                         *m_mpsrt_context,
                                                         image_bridge_copies,
                                                         GfxMpsrtStorageBridgeDirection::BufferToImage,
                                                         hooks,
                                                         &mpsrt_error),
                        mpsrt_error);
        const bool encoded =
            request.encode_prepared_model(command_buffer,
                                          *m_mpsrt_model,
                                          prepared_model,
                                          stage_dispatches,
                                          bindings,
                                          hooks,
                                          &encode_result,
                                          &mpsrt_error);
        OPENVINO_ASSERT(encoded, mpsrt_error);
        OPENVINO_ASSERT(encode_mpsrt_image_bridge_copies(command_buffer,
                                                         *m_mpsrt_context,
                                                         image_bridge_copies,
                                                         GfxMpsrtStorageBridgeDirection::ImageToBuffer,
                                                         hooks,
                                                         &mpsrt_error),
                        mpsrt_error);
        return;
    }

    OPENVINO_ASSERT(m_pipeline, "MetalCompiledKernel: pipeline is null");
    const bool trace_encoder_setup = hooks && (hooks->on_segment || hooks->on_counter);
    const auto encoder_setup_start =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool encoder_created = false;
    id<MTLComputeCommandEncoder> enc =
        static_cast<id<MTLComputeCommandEncoder>>(metal_get_or_create_compute_encoder(command_buffer, &encoder_created));
    OPENVINO_ASSERT(enc, "MetalCompiledKernel: failed to create compute encoder");
    const auto pipeline_bind_start =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool pipeline_bound =
        metal_set_compute_pipeline_if_needed(command_buffer,
                                             reinterpret_cast<GpuCommandEncoderHandle>(enc),
                                             m_pipeline);
    const auto after_pipeline_bind =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    const auto buffer_bind_start =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const size_t bound_buffers =
        metal_bind_compute_buffers_if_needed(command_buffer,
                                             reinterpret_cast<GpuCommandEncoderHandle>(enc),
                                             prepared->buffer_ptrs,
                                             prepared->offsets);
    if (trace_encoder_setup) {
        const auto encoder_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encoder_setup_start);
        const auto pipeline_bind_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after_pipeline_bind - pipeline_bind_start);
        const auto buffer_bind_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - buffer_bind_start);
        if (hooks->on_counter) {
            if (encoder_created) {
                hooks->on_counter("encoder_setup_count", 1);
            }
            if (pipeline_bound) {
                hooks->on_counter("pipeline_bind_count", 1);
            }
            hooks->on_counter("buffer_bind_count", static_cast<uint64_t>(bound_buffers));
        }
        if (hooks->on_segment) {
            hooks->on_segment("descriptor_update",
                              "metal_pipeline_bind",
                              pipeline_bind_cpu_us,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -1,
                              0,
                              reinterpret_cast<uint64_t>(cb));
            hooks->on_segment("descriptor_update",
                              "metal_buffer_bind",
                              buffer_bind_cpu_us,
                              0,
                              static_cast<uint32_t>(bound_buffers),
                              0,
                              0,
                              0,
                              0,
                              -1,
                              0,
                              reinterpret_cast<uint64_t>(cb));
            if (encoder_created && encoder_cpu_us > pipeline_bind_cpu_us + buffer_bind_cpu_us) {
                hooks->on_segment("descriptor_update",
                                  "metal_encoder_setup_overhead",
                                  encoder_cpu_us - pipeline_bind_cpu_us - buffer_bind_cpu_us,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  -1,
                                  0,
                                  reinterpret_cast<uint64_t>(cb));
            }
        }
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
        return;
    }

    MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);
    MTLSize tg = MTLSizeMake(dispatch.threads_per_group[0],
                             dispatch.threads_per_group[1],
                             dispatch.threads_per_group[2]);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];

    if (hooks && hooks->on_end) {
        hooks->on_end(enc);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
