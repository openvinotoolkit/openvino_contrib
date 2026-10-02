// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/mpsrt/mpsrt_request.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <chrono>
#include <sstream>

#include "backends/metal/runtime/metal_command_encoder.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_msl_kernel_loader.hpp"
#include "kernel_ir/metal_kernels/mpsrt_image_bridge_kernels.hpp"
#include "kernel_ir/metal_kernels/mpsrt_topk_kernels.hpp"
#include "openvino/core/except.hpp"
#include "backends/metal/common/mpsrt/gfx_mpsrt_kernel_manifest_adapter.hpp"

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {

namespace runtime_mpsrt = ::ov::gfx_plugin::mpsrt;
using runtime_mpsrt::MpsrtModel;
using runtime_mpsrt::MpsrtRuntimeResourceLifetime;
using runtime_mpsrt::MpsrtRuntimeStage;
using runtime_mpsrt::MpsrtRuntimeTensor;
using runtime_mpsrt::MpsrtTensorBindingPlanEntry;

namespace {

bool fail(std::string *error, const std::string &message) {
  if (error) {
    *error = message;
  }
  return false;
}

const MpsrtPreparedMslDispatch *
find_prepared_msl_dispatch(const MpsrtPreparedModel &prepared_model,
                           size_t stage_index) {
  for (const auto &dispatch : prepared_model.msl_dispatches) {
    if (dispatch.stage_index == stage_index) {
      return &dispatch;
    }
  }
  return nullptr;
}

const MpsrtPreparedMpsGemm *
find_prepared_mps_gemm(const MpsrtPreparedModel &prepared_model,
                       size_t stage_index) {
  for (const auto &gemm : prepared_model.mps_gemm_stages) {
    if (gemm.stage_index == stage_index) {
      return &gemm;
    }
  }
  return nullptr;
}

const MpsrtPreparedMpsConv2D *
find_prepared_mps_conv2d(const MpsrtPreparedModel &prepared_model,
                         size_t stage_index) {
  for (const auto &conv : prepared_model.mps_conv2d_stages) {
    if (conv.stage_index == stage_index) {
      return &conv;
    }
  }
  return nullptr;
}

const MpsrtPreparedMpsPool2D *
find_prepared_mps_pool2d(const MpsrtPreparedModel &prepared_model,
                         size_t stage_index) {
  for (const auto &pool : prepared_model.mps_pool2d_stages) {
    if (pool.stage_index == stage_index) {
      return &pool;
    }
  }
  return nullptr;
}

const MpsrtPreparedMpsResize2D *
find_prepared_mps_resize2d(const MpsrtPreparedModel &prepared_model,
                           size_t stage_index) {
  for (const auto &resize : prepared_model.mps_resize2d_stages) {
    if (resize.stage_index == stage_index) {
      return &resize;
    }
  }
  return nullptr;
}

const MpsrtPreparedMpsSoftmax *
find_prepared_mps_softmax(const MpsrtPreparedModel &prepared_model,
                          size_t stage_index) {
  for (const auto &softmax : prepared_model.mps_softmax_stages) {
    if (softmax.stage_index == stage_index) {
      return &softmax;
    }
  }
  return nullptr;
}

const MpsrtPreparedMpsTopK *
find_prepared_mps_topk(const MpsrtPreparedModel &prepared_model,
                       size_t stage_index) {
  for (const auto &topk : prepared_model.mps_topk_stages) {
    if (topk.stage_index == stage_index) {
      return &topk;
    }
  }
  return nullptr;
}

const MpsrtPreparedMpsSdpa *
find_prepared_mps_sdpa(const MpsrtPreparedModel &prepared_model,
                       size_t stage_index) {
  for (const auto &sdpa : prepared_model.mps_sdpa_stages) {
    if (sdpa.stage_index == stage_index) {
      return &sdpa;
    }
  }
  return nullptr;
}

bool is_mps_conv2d_stage(GfxMpsrtStageKind kind) {
  return kind == GfxMpsrtStageKind::MPSConv2D ||
         kind == GfxMpsrtStageKind::MPSGroupConv2D;
}

uint64_t tensor_element_count(const GfxMpsrtTensorAbiDesc &desc) {
  if (desc.rank == 0 || desc.rank > 8) {
    return 0;
  }
  uint64_t total = 1;
  for (uint32_t i = 0; i < desc.rank; ++i) {
    if (desc.dims[i] == 0) {
      return 0;
    }
    total *= desc.dims[i];
  }
  return total;
}

KernelDispatch make_msl_stage_dispatch(
    const MpsrtRuntimeStage &stage, const MpsrtPreparedMslDispatch &prepared,
    const KernelDispatch &fallback) {
  if (stage.kind != GfxMpsrtStageKind::MSLDispatch ||
      stage.output_descs.empty()) {
    return fallback;
  }

  uint64_t total = 0;
  for (const auto &desc : stage.output_descs) {
    total = std::max(total, tensor_element_count(desc));
  }
  if (total == 0) {
    return fallback;
  }

  uint32_t threads = prepared.dispatch_threads_per_threadgroup;
  if (threads == 0) {
    threads = static_cast<uint32_t>(
        std::max<size_t>(fallback.threads_per_group[0], 1));
  }
  if (prepared.max_total_threads_per_threadgroup != 0) {
    threads = std::min(threads, prepared.max_total_threads_per_threadgroup);
  }
  threads = std::max<uint32_t>(threads, 1u);

  KernelDispatch dispatch{};
  dispatch.grid[0] = static_cast<size_t>(std::max<uint64_t>(total, 1));
  dispatch.grid[1] = 1;
  dispatch.grid[2] = 1;
  dispatch.threads_per_group[0] = threads;
  dispatch.threads_per_group[1] = 1;
  dispatch.threads_per_group[2] = 1;
  return dispatch;
}

struct MpsrtImageBridgeParams {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  uint32_t batch = 0;
};

struct MpsrtBindingBuildResult {
  size_t external_inputs_bound = 0;
  size_t external_outputs_bound = 0;
  size_t external_resources_bound = 0;
  size_t model_resources_bound = 0;
  size_t transient_buffers_allocated = 0;
  size_t transient_images_allocated = 0;
};

uint32_t mpsrt_image_slice_count(uint32_t feature_channels) {
  return (feature_channels + 3) / 4;
}

MpsrtImageBridgeKernelKind
mpsrt_image_bridge_kernel_kind(const GfxMpsrtTensorAbiDesc &desc,
                               GfxMpsrtStorageBridgeDirection direction) {
  const bool f16 = desc.dtype == static_cast<uint32_t>(GfxMpsrtDType::F16);
  if (direction == GfxMpsrtStorageBridgeDirection::BufferToImage) {
    return f16 ? MpsrtImageBridgeKernelKind::BufferToImageF16
               : MpsrtImageBridgeKernelKind::BufferToImageF32;
  }
  return f16 ? MpsrtImageBridgeKernelKind::ImageToBufferF16
             : MpsrtImageBridgeKernelKind::ImageToBufferF32;
}

bool encode_mpsrt_image_bridge_copy(GpuCommandBufferHandle command_buffer,
                                    MpsrtContext &context,
                                    const MpsrtImageBridgeCopy &copy,
                                    const KernelExecutionHooks *hooks,
                                    std::string *error) {
  if (!copy.buffer_binding.buffer || !copy.image_binding.texture) {
    return fail(error, "GFX MPSRT: image bridge copy has incomplete resources");
  }

  const auto &kernel = mpsrt_image_bridge_kernel_source(
      mpsrt_image_bridge_kernel_kind(copy.desc, copy.direction));
  bool cache_hit = false;
  id<MTLComputePipelineState> pipeline =
      MpsrtMslKernelLoader::load_pipeline(context, kernel, 64, cache_hit,
                                          error);
  if (!pipeline) {
    return false;
  }

  bool encoder_created = false;
  id<MTLComputeCommandEncoder> encoder =
      static_cast<id<MTLComputeCommandEncoder>>(
          metal_get_or_create_compute_encoder(command_buffer,
                                              &encoder_created));
  if (!encoder) {
    return fail(error,
                "GFX MPSRT: failed to create image bridge compute encoder");
  }
  metal_set_compute_pipeline_if_needed(
      command_buffer, reinterpret_cast<GpuCommandEncoderHandle>(encoder),
      (__bridge void *)pipeline);

  id<MTLBuffer> buffer = static_cast<id<MTLBuffer>>(copy.buffer_binding.buffer);
  id<MTLTexture> texture =
      static_cast<id<MTLTexture>>(copy.image_binding.texture);
  MpsrtImageBridgeParams params{};
  params.width = copy.desc.image_width;
  params.height = copy.desc.image_height;
  params.channels = copy.desc.image_feature_channels;
  params.batch = copy.desc.image_batch;
  [encoder setBuffer:buffer offset:copy.buffer_binding.offset atIndex:0];
  [encoder setBytes:&params length:sizeof(params) atIndex:1];
  [encoder setTexture:texture atIndex:0];

  const NSUInteger slices = static_cast<NSUInteger>(
      mpsrt_image_slice_count(copy.desc.image_feature_channels));
  const MTLSize grid =
      MTLSizeMake(copy.desc.image_width, copy.desc.image_height,
                  copy.desc.image_batch * slices);
  const MTLSize threads = MTLSizeMake(8, 8, 1);
  [encoder dispatchThreads:grid threadsPerThreadgroup:threads];

  if (hooks && hooks->on_counter) {
    hooks->on_counter(copy.direction ==
                              GfxMpsrtStorageBridgeDirection::BufferToImage
                          ? "mpsrt_image_bridge_buffer_to_image_encode_count"
                          : "mpsrt_image_bridge_image_to_buffer_encode_count",
                      1);
    hooks->on_counter(cache_hit
                          ? "mpsrt_image_bridge_pipeline_cache_hit_count"
                          : "mpsrt_image_bridge_pipeline_cache_miss_count",
                      1);
    if (encoder_created) {
      hooks->on_counter("mpsrt_image_bridge_encoder_create_count", 1);
    }
  }
  return true;
}

bool encode_mpsrt_image_bridge_copies(
    GpuCommandBufferHandle command_buffer, MpsrtContext &context,
    const std::vector<MpsrtImageBridgeCopy> &copies,
    GfxMpsrtStorageBridgeDirection direction, const KernelExecutionHooks *hooks,
    std::string *error) {
  for (const auto &copy : copies) {
    if (copy.direction != direction) {
      continue;
    }
    if (!encode_mpsrt_image_bridge_copy(command_buffer, context, copy, hooks,
                                        error)) {
      return false;
    }
  }
  return true;
}

const MpsrtRuntimeTensor *find_tensor(const MpsrtModel &model,
                                      GfxMpsrtValue value) {
  for (const auto &tensor : model.tensors) {
    if (tensor.value == value) {
      return &tensor;
    }
  }
  return nullptr;
}

const MpsrtPreparedImageBridgeResource *
find_prepared_image_bridge_resource(const MpsrtPreparedModel *prepared_model,
                                    GfxMpsrtValue value,
                                    GfxMpsrtStorageBridgeDirection direction) {
  if (!prepared_model) {
    return nullptr;
  }
  for (const auto &resource : prepared_model->image_bridge_resources) {
    if (resource.value == value && resource.direction == direction) {
      return &resource;
    }
  }
  return nullptr;
}

bool tensor_requires_image_binding(const GfxMpsrtTensorAbiDesc &desc) {
  return desc.storage == static_cast<uint32_t>(GfxMpsrtStorage::Image);
}

bool materialize_mpsrt_image_bridge_binding(
    const MpsrtModel &model, const MpsrtPreparedModel *prepared_model,
    GfxMpsrtValue value, const GfxMpsrtTensorAbiDesc *fallback_desc,
    GfxMpsrtStorageBridgeDirection direction, MpsrtBoundBuffer &binding,
    std::vector<MpsrtImageBridgeCopy> &bridge_copies, std::string *error) {
  const auto *tensor = find_tensor(model, value);
  const auto *desc = tensor ? &tensor->desc : fallback_desc;
  if (!desc) {
    return true;
  }
  if (!gfx_mpsrt_tensor_is_image(*desc)) {
    return true;
  }
  if (binding.texture) {
    return true;
  }
  if (direction == GfxMpsrtStorageBridgeDirection::Unknown) {
    return true;
  }
  if (!binding.buffer) {
    return fail(error,
                "GFX MPSRT: image bridge external buffer binding is null");
  }
  if (!gfx_mpsrt_image_bridge_supported(*desc)) {
    std::ostringstream stream;
    stream << "GFX MPSRT: image bridge supports only static rank-4 f16/f32 "
              "image tensors"
           << " value=" << value << " rank=" << desc->rank
           << " storage=" << desc->storage << " flags=" << desc->flags;
    return fail(error, stream.str());
  }
  GfxMpsrtStorageBridgeDesc bridge_desc{};
  if (!gfx_mpsrt_make_image_bridge_desc(value, *desc, direction, bridge_desc)) {
    return fail(error, "GFX MPSRT: image bridge storage contract is invalid");
  }

  const auto *prepared_bridge = find_prepared_image_bridge_resource(
      prepared_model, bridge_desc.value, bridge_desc.direction);
  if (!prepared_bridge || !prepared_bridge->texture) {
    return fail(error, "GFX MPSRT: prepared image bridge resource is missing");
  }
  MpsrtBoundBuffer image_binding{nullptr, 0,
                                 (__bridge void *)prepared_bridge->texture};
  bridge_copies.push_back({bridge_desc.direction, bridge_desc.value,
                           bridge_desc.tensor, binding, image_binding});
  binding = image_binding;
  return true;
}

bool materialize_mpsrt_image_bridge_bindings(
    const MpsrtModel &model, const MpsrtPreparedModel *prepared_model,
    const std::vector<GfxMpsrtValue> &values,
    GfxMpsrtStorageBridgeDirection fallback_direction,
    std::vector<MpsrtBoundBuffer> &bindings,
    std::vector<MpsrtImageBridgeCopy> &bridge_copies, std::string *error) {
  if (values.size() != bindings.size()) {
    return fail(
        error,
        "GFX MPSRT: image bridge value count does not match binding count");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    const auto direction =
        runtime_mpsrt::mpsrt_model_external_bridge_direction_for_value(
            model, values[i], fallback_direction);
    if (!materialize_mpsrt_image_bridge_binding(
            model, prepared_model, values[i], nullptr, direction, bindings[i],
            bridge_copies, error)) {
      return false;
    }
  }
  return true;
}

bool materialize_mpsrt_image_bridge_bindings_from_plan(
    const MpsrtModel &model, const MpsrtPreparedModel *prepared_model,
    std::vector<MpsrtBoundBuffer> &external_buffers,
    std::vector<MpsrtImageBridgeCopy> &bridge_copies, std::string *error) {
  std::vector<MpsrtTensorBindingPlanEntry> binding_plan;
  if (!runtime_mpsrt::mpsrt_model_tensor_binding_plan(model, binding_plan,
                                                      error)) {
    return false;
  }
  for (const auto &binding : binding_plan) {
    if (binding.lifetime !=
            runtime_mpsrt::MpsrtRuntimeResourceLifetime::External ||
        !binding.has_tensor_value) {
      continue;
    }
    if (binding.arg_index >= external_buffers.size()) {
      return fail(error,
                  "GFX MPSRT: external buffer ABI index is out of range");
    }
    if (!materialize_mpsrt_image_bridge_binding(
            model, prepared_model, binding.value, &binding.tensor_desc,
            binding.bridge_direction, external_buffers[binding.arg_index],
            bridge_copies, error)) {
      return false;
    }
  }
  return true;
}

bool validate_bound_resource(const GfxMpsrtTensorAbiDesc &desc,
                             const MpsrtBoundBuffer &bound,
                             const std::string &name, std::string *error) {
  if (tensor_requires_image_binding(desc)) {
    if (!bound.texture) {
      return fail(error,
                  "GFX MPSRT: " + name + " image texture binding is null");
    }
    if (bound.offset != 0) {
      return fail(error, "GFX MPSRT: " + name +
                             " image binding must have zero byte offset");
    }
    return true;
  }
  if (!bound.buffer) {
    return fail(error, "GFX MPSRT: " + name + " buffer binding is null");
  }
  return true;
}

void count_transient_resource(const GfxMpsrtTensorAbiDesc &desc,
                              MpsrtBindingBuildResult *result) {
  if (!result) {
    return;
  }
  if (tensor_requires_image_binding(desc)) {
    ++result->transient_images_allocated;
  } else {
    ++result->transient_buffers_allocated;
  }
}

void count_external_buffer_tensor_resource(GfxMpsrtExternalBufferRole role,
                                           MpsrtBindingBuildResult *result) {
  if (!result) {
    return;
  }
  if (gfx_mpsrt_is_external_output_buffer_role(role)) {
    ++result->external_outputs_bound;
  } else {
    ++result->external_inputs_bound;
  }
}

void count_external_buffer_resource(GfxMpsrtExternalBufferRole role,
                                    MpsrtBindingBuildResult *result) {
  if (!result) {
    return;
  }
  if (gfx_mpsrt_is_external_output_buffer_role(role)) {
    ++result->external_outputs_bound;
  } else {
    ++result->external_inputs_bound;
  }
  ++result->external_resources_bound;
}

const MpsrtPreparedResource *
find_prepared_resource(const MpsrtPreparedModel *prepared_model,
                       uint32_t resource_index) {
  if (!prepared_model) {
    return nullptr;
  }
  for (const auto &resource : prepared_model->resources) {
    if (resource.resource_index == resource_index) {
      return &resource;
    }
  }
  return nullptr;
}

bool validate_bound_resource(const MpsrtTensorBindingPlanEntry &resource,
                             const MpsrtBoundBuffer &bound,
                             const std::string &name, std::string *error) {
  if (!resource.has_tensor_value) {
    return fail(error, "GFX MPSRT: " + name + " resource is not a tensor");
  }
  return validate_bound_resource(resource.tensor_desc, bound, name, error);
}

bool bind_external_buffer_tensor_resource(
    const MpsrtTensorBindingPlanEntry &resource, const MpsrtBoundBuffer &bound,
    const std::string &name, MpsrtTensorBindings &bindings,
    MpsrtBindingBuildResult *result, std::string *error) {
  if (resource.lifetime != MpsrtRuntimeResourceLifetime::External) {
    return fail(error, "GFX MPSRT: " + name + " resource is not external");
  }
  if (!validate_bound_resource(resource, bound, name, error)) {
    return false;
  }
  bindings.bind(resource.value, bound);
  count_external_buffer_tensor_resource(resource.role, result);
  return true;
}

bool bind_external_buffer_resource(const MpsrtTensorBindingPlanEntry &resource,
                                   const MpsrtBoundBuffer &bound,
                                   const std::string &name,
                                   MpsrtTensorBindings &bindings,
                                   MpsrtBindingBuildResult *result,
                                   std::string *error) {
  if (resource.lifetime != MpsrtRuntimeResourceLifetime::External) {
    return fail(error, "GFX MPSRT: " + name + " resource is not external");
  }
  if (resource.has_tensor_value) {
    return bind_external_buffer_tensor_resource(resource, bound, name, bindings,
                                                result, error);
  }
  if (!bound.buffer) {
    if (resource.role == GfxMpsrtExternalBufferRole::RuntimeParams ||
        resource.role == GfxMpsrtExternalBufferRole::Metadata) {
      return true;
    }
    return fail(error, "GFX MPSRT: " + name + " buffer binding is null");
  }
  bindings.bind(resource.value, bound);
  count_external_buffer_resource(resource.role, result);
  return true;
}

bool bind_model_owned_resource(const MpsrtTensorBindingPlanEntry &resource,
                               const MpsrtPreparedModel *prepared_model,
                               MpsrtTensorBindings &bindings,
                               MpsrtBindingBuildResult *result,
                               std::string *error) {
  if (!resource.has_tensor_value) {
    return fail(
        error, "GFX MPSRT: model-owned non-tensor resources are not supported");
  }
  if (resource.role != GfxMpsrtExternalBufferRole::ConstBuffer) {
    return fail(error,
                "GFX MPSRT: model-owned resource must be a const buffer");
  }
  const auto *prepared =
      find_prepared_resource(prepared_model, resource.resource_index);
  if (!prepared) {
    return fail(error, "GFX MPSRT: model-owned resource " +
                           std::to_string(resource.resource_index) +
                           " is missing from prepared resources");
  }
  if (prepared->lifetime != MpsrtRuntimeResourceLifetime::Model ||
      !prepared->has_tensor_value || prepared->value != resource.value ||
      !prepared->buffer) {
    return fail(error, "GFX MPSRT: prepared model resource is not bindable");
  }
  MpsrtBoundBuffer bound{(__bridge void *)prepared->buffer, prepared->offset};
  if (!validate_bound_resource(resource, bound,
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

bool bind_prepared_transient_resource(
    const MpsrtTensorBindingPlanEntry &resource,
    const MpsrtPreparedModel *prepared_model, MpsrtTensorBindings &bindings,
    MpsrtBindingBuildResult *result, std::string *error) {
  if (!resource.has_tensor_value) {
    return fail(error,
                "GFX MPSRT: transient non-tensor resources are not supported");
  }
  const auto *prepared =
      find_prepared_resource(prepared_model, resource.resource_index);
  if (!prepared) {
    return fail(error, "GFX MPSRT: transient resource " +
                           std::to_string(resource.resource_index) +
                           " is missing from prepared resources");
  }
  if (prepared->lifetime != MpsrtRuntimeResourceLifetime::Transient ||
      !prepared->has_tensor_value || prepared->value != resource.value) {
    return fail(error,
                "GFX MPSRT: prepared transient resource is not bindable");
  }
  MpsrtBoundBuffer allocated{};
  if (tensor_requires_image_binding(resource.tensor_desc)) {
    allocated = make_mpsrt_bound_image((__bridge void *)prepared->texture);
  } else {
    allocated = {(__bridge void *)prepared->buffer, prepared->offset};
  }
  if (!validate_bound_resource(resource, allocated,
                               "transient binding for resource " +
                                   std::to_string(resource.resource_index),
                               error)) {
    return false;
  }
  bindings.bind(resource.value, allocated);
  count_transient_resource(resource.tensor_desc, result);
  return true;
}

bool bind_tensor_binding_plan(
    const MpsrtModel &model,
    const std::vector<MpsrtBoundBuffer> &external_buffers,
    const MpsrtPreparedModel *prepared_model, MpsrtTensorBindings &bindings,
    MpsrtBindingBuildResult *result, std::string *error) {
  std::vector<MpsrtTensorBindingPlanEntry> plan;
  if (!runtime_mpsrt::mpsrt_model_tensor_binding_plan(model, plan, error)) {
    return false;
  }
  for (const auto &resource : plan) {
    switch (resource.lifetime) {
    case MpsrtRuntimeResourceLifetime::External:
      if (resource.arg_index >= external_buffers.size()) {
        return fail(
            error,
            "GFX MPSRT: external buffer ABI index is out of range"
            " arg_index=" +
                std::to_string(resource.arg_index) + " external_buffer_count=" +
                std::to_string(external_buffers.size()) +
                " resource_index=" + std::to_string(resource.resource_index) +
                " value=" + std::to_string(resource.value) + " role=" +
                std::to_string(static_cast<uint32_t>(resource.role)));
      }
      if (!bind_external_buffer_resource(resource,
                                         external_buffers[resource.arg_index],
                                         "external binding at ABI index " +
                                             std::to_string(resource.arg_index),
                                         bindings, result, error)) {
        return false;
      }
      break;
    case MpsrtRuntimeResourceLifetime::Model:
      if (resource.has_tensor_value && bindings.lookup(resource.value)) {
        break;
      }
      if (!bind_model_owned_resource(resource, prepared_model, bindings, result,
                                     error)) {
        return false;
      }
      break;
    case MpsrtRuntimeResourceLifetime::Transient:
      if (resource.has_tensor_value && bindings.lookup(resource.value)) {
        break;
      }
      if (!bind_prepared_transient_resource(resource, prepared_model, bindings,
                                            result, error)) {
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

NSArray<NSNumber *> *mps_shape_from_tensor_desc(
    const GfxMpsrtTensorAbiDesc &desc) {
  NSMutableArray<NSNumber *> *shape =
      [NSMutableArray arrayWithCapacity:desc.rank];
  for (uint32_t i = 0; i < desc.rank && i < 8; ++i) {
    [shape addObject:@(desc.dims[i])];
  }
  return shape;
}

uint32_t matrix_count_or_one(const GfxMpsrtTensorAbiDesc &desc) {
  return desc.matrix_count == 0 ? 1 : desc.matrix_count;
}

NSUInteger matrix_bytes_for_desc(const GfxMpsrtTensorAbiDesc &desc) {
  return static_cast<NSUInteger>(desc.matrix_rows) *
         static_cast<NSUInteger>(desc.matrix_row_bytes);
}

size_t matrix_batch_offset(const GfxMpsrtTensorAbiDesc &desc,
                           uint32_t batch_index) {
  if (matrix_count_or_one(desc) == 1) {
    return static_cast<size_t>(desc.byte_offset);
  }
  return static_cast<size_t>(desc.byte_offset) +
         static_cast<size_t>(batch_index) *
             static_cast<size_t>(matrix_bytes_for_desc(desc));
}

bool make_mps_matrix_descriptor(const GfxMpsrtTensorAbiDesc &desc,
                                MPSMatrixDescriptor *&out, const char *name,
                                std::string *error,
                                uint32_t matrix_count_override = 0) {
  out = nil;
  if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
    return fail(error, std::string("GFX MPSRT: MPS GEMM ") + name +
                           " tensor is not matrix storage");
  }
  if (desc.matrix_rows == 0 || desc.matrix_columns == 0 ||
      desc.matrix_row_bytes == 0) {
    return fail(error, std::string("GFX MPSRT: MPS GEMM ") + name +
                           " matrix descriptor is incomplete");
  }
  const MPSDataType data_type = mps_data_type_from_gfx(desc.dtype);
  if (data_type == MPSDataTypeInvalid) {
    return fail(error, std::string("GFX MPSRT: MPS GEMM ") + name +
                           " dtype is unsupported");
  }
  const uint32_t matrix_count = matrix_count_override == 0
                                    ? matrix_count_or_one(desc)
                                    : matrix_count_override;
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
    return fail(error, std::string("GFX MPSRT: failed to create MPS GEMM ") +
                           name + " descriptor");
  }
  return true;
}

bool make_mps_topk_index_matrix_descriptor(const GfxMpsrtTensorAbiDesc &desc,
                                           MPSMatrixDescriptor *&out,
                                           const char *name,
                                           std::string *error) {
  out = nil;
  if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
    return fail(error, std::string("GFX MPSRT: MPS TopK ") + name +
                           " tensor is not matrix storage");
  }
  if (desc.matrix_rows == 0 || desc.matrix_columns == 0 ||
      desc.matrix_row_bytes == 0) {
    return fail(error, std::string("GFX MPSRT: MPS TopK ") + name +
                           " matrix descriptor is incomplete");
  }
  if (desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::I32) &&
      desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::U32)) {
    return fail(error, std::string("GFX MPSRT: MPS TopK ") + name +
                           " dtype must be i32/u32");
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
    return fail(error, std::string("GFX MPSRT: failed to create MPS TopK ") +
                           name + " descriptor");
  }
  return true;
}

GfxMpsrtTensorAbiDesc make_mps_topk_u32_index_desc(
    const GfxMpsrtTensorAbiDesc &output) {
  GfxMpsrtTensorAbiDesc desc = output;
  desc.dtype = static_cast<uint32_t>(GfxMpsrtDType::U32);
  desc.byte_offset = 0;
  desc.matrix_row_bytes = output.matrix_columns * sizeof(uint32_t);
  desc.byte_length = static_cast<uint64_t>(matrix_count_or_one(output)) *
                     output.matrix_rows * desc.matrix_row_bytes;
  return desc;
}

struct MpsrtTopKI64PackParams {
  uint32_t rows = 0;
  uint32_t k = 0;
  uint32_t matrix_count = 1;
  uint32_t src_matrix_stride_u32 = 0;
  uint32_t dst_row_stride_i32 = 0;
  uint32_t dst_matrix_stride_i32 = 0;
};

bool encode_mps_topk_i64_pack_bridge(GpuCommandBufferHandle command_buffer,
                                     MpsrtContext &context,
                                     id<MTLBuffer> src,
                                     id<MTLBuffer> dst,
                                     NSUInteger dst_offset,
                                     const GfxMpsrtTensorAbiDesc &src_desc,
                                     const GfxMpsrtTensorAbiDesc &dst_desc,
                                     const KernelExecutionHooks *hooks,
                                     std::string *error) {
  const auto &kernel = mpsrt_topk_pack_u32_to_i64_kernel_source();

  bool cache_hit = false;
  id<MTLComputePipelineState> pipeline =
      MpsrtMslKernelLoader::load_pipeline(context, kernel, 64, cache_hit,
                                          error);
  if (!pipeline) {
    return false;
  }

  bool encoder_created = false;
  id<MTLComputeCommandEncoder> encoder =
      static_cast<id<MTLComputeCommandEncoder>>(
          metal_get_or_create_compute_encoder(command_buffer,
                                              &encoder_created));
  if (!encoder) {
    return fail(error,
                "GFX MPSRT: failed to create TopK i64 pack compute encoder");
  }
  metal_set_compute_pipeline_if_needed(
      command_buffer, reinterpret_cast<GpuCommandEncoderHandle>(encoder),
      (__bridge void *)pipeline);

  MpsrtTopKI64PackParams params{};
  params.rows = src_desc.matrix_rows;
  params.k = src_desc.matrix_columns;
  params.matrix_count = matrix_count_or_one(src_desc);
  params.src_matrix_stride_u32 =
      static_cast<uint32_t>(matrix_bytes_for_desc(src_desc) / sizeof(uint32_t));
  params.dst_row_stride_i32 =
      static_cast<uint32_t>(dst_desc.matrix_row_bytes / sizeof(uint32_t));
  params.dst_matrix_stride_i32 =
      static_cast<uint32_t>(matrix_bytes_for_desc(dst_desc) / sizeof(uint32_t));

  [encoder setBuffer:src offset:0 atIndex:0];
  [encoder setBuffer:dst offset:dst_offset atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];

  const NSUInteger total =
      static_cast<NSUInteger>(params.rows) * params.k * params.matrix_count;
  [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

  if (hooks && hooks->on_counter) {
    hooks->on_counter("mpsrt_mps_topk_i64_index_bridge_encode_count", 1);
    hooks->on_counter("mpsrt_mps_topk_i64_pack_bridge_encode_count", 1);
    hooks->on_counter(cache_hit
                          ? "mpsrt_mps_topk_i64_pack_pipeline_cache_hit_count"
                          : "mpsrt_mps_topk_i64_pack_pipeline_cache_miss_count",
                      1);
    if (encoder_created) {
      hooks->on_counter("mpsrt_mps_topk_i64_pack_encoder_create_count", 1);
    }
  }
  return true;
}

bool mpsrt_topk_value_type(uint32_t dtype, MpsrtTopKValueType *out) {
  if (dtype == static_cast<uint32_t>(GfxMpsrtDType::F16)) {
    *out = MpsrtTopKValueType::F16;
    return true;
  }
  if (dtype == static_cast<uint32_t>(GfxMpsrtDType::F32)) {
    *out = MpsrtTopKValueType::F32;
    return true;
  }
  return false;
}

struct MpsrtTopKStableIndexParams {
  uint32_t rows = 0;
  uint32_t k = 0;
  uint32_t source_columns = 0;
  uint32_t matrix_count = 1;
  uint32_t input_matrix_stride = 0;
  uint32_t input_row_stride = 0;
  uint32_t values_matrix_stride = 0;
  uint32_t values_row_stride = 0;
  uint32_t fallback_matrix_stride = 0;
  uint32_t dst_row_stride_i32 = 0;
  uint32_t dst_matrix_stride_i32 = 0;
};

bool encode_mps_topk_stable_i64_index_resolve(
    GpuCommandBufferHandle command_buffer, MpsrtContext &context,
    id<MTLBuffer> input, id<MTLBuffer> values, id<MTLBuffer> fallback_indices,
    id<MTLBuffer> dst, NSUInteger dst_offset,
    const GfxMpsrtTensorAbiDesc &input_desc,
    const GfxMpsrtTensorAbiDesc &values_desc,
    const GfxMpsrtTensorAbiDesc &fallback_desc,
    const GfxMpsrtTensorAbiDesc &dst_desc, const KernelExecutionHooks *hooks,
    std::string *error) {
  MpsrtTopKValueType value_type{};
  if (!mpsrt_topk_value_type(input_desc.dtype, &value_type) ||
      dst_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::I64)) {
    return fail(error,
                "GFX MPSRT: stable TopK i64 index resolve dtype is unsupported");
  }
  const auto &kernel = mpsrt_topk_stable_i64_indices_kernel_source(value_type);

  bool cache_hit = false;
  id<MTLComputePipelineState> pipeline =
      MpsrtMslKernelLoader::load_pipeline(context, kernel, 64, cache_hit,
                                          error);
  if (!pipeline) {
    return false;
  }

  bool encoder_created = false;
  id<MTLComputeCommandEncoder> encoder =
      static_cast<id<MTLComputeCommandEncoder>>(
          metal_get_or_create_compute_encoder(command_buffer,
                                              &encoder_created));
  if (!encoder) {
    return fail(error,
                "GFX MPSRT: failed to create TopK stable index encoder");
  }
  metal_set_compute_pipeline_if_needed(
      command_buffer, reinterpret_cast<GpuCommandEncoderHandle>(encoder),
      (__bridge void *)pipeline);

  const uint32_t value_size =
      static_cast<uint32_t>(gfx_mpsrt_element_size_bytes(
          static_cast<GfxMpsrtDType>(input_desc.dtype)));
  MpsrtTopKStableIndexParams params{};
  params.rows = values_desc.matrix_rows;
  params.k = values_desc.matrix_columns;
  params.source_columns = input_desc.matrix_columns;
  params.matrix_count = matrix_count_or_one(values_desc);
  params.input_matrix_stride =
      static_cast<uint32_t>(matrix_bytes_for_desc(input_desc) / value_size);
  params.input_row_stride =
      static_cast<uint32_t>(input_desc.matrix_row_bytes / value_size);
  params.values_matrix_stride =
      static_cast<uint32_t>(matrix_bytes_for_desc(values_desc) / value_size);
  params.values_row_stride =
      static_cast<uint32_t>(values_desc.matrix_row_bytes / value_size);
  params.fallback_matrix_stride =
      static_cast<uint32_t>(matrix_bytes_for_desc(fallback_desc) / sizeof(uint32_t));
  params.dst_row_stride_i32 =
      static_cast<uint32_t>(dst_desc.matrix_row_bytes / sizeof(uint32_t));
  params.dst_matrix_stride_i32 =
      static_cast<uint32_t>(matrix_bytes_for_desc(dst_desc) / sizeof(uint32_t));

  [encoder setBuffer:input offset:0 atIndex:0];
  [encoder setBuffer:values offset:0 atIndex:1];
  [encoder setBuffer:fallback_indices offset:0 atIndex:2];
  [encoder setBuffer:dst offset:dst_offset atIndex:3];
  [encoder setBytes:&params length:sizeof(params) atIndex:4];

  const NSUInteger total = static_cast<NSUInteger>(params.rows) * params.k *
                           params.matrix_count;
  [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

  if (hooks && hooks->on_counter) {
    hooks->on_counter("mpsrt_mps_topk_i64_index_bridge_encode_count", 1);
    hooks->on_counter("mpsrt_mps_topk_stable_i64_index_resolve_encode_count",
                      1);
    hooks->on_counter(
        cache_hit
            ? "mpsrt_mps_topk_stable_i64_index_resolve_pipeline_cache_hit_count"
            : "mpsrt_mps_topk_stable_i64_index_resolve_pipeline_cache_miss_count",
        1);
    if (encoder_created) {
      hooks->on_counter(
          "mpsrt_mps_topk_stable_i64_index_resolve_encoder_create_count",
          1);
    }
  }
  return true;
}

bool validate_mps_gemm_batch_contract(const GfxMpsrtTensorAbiDesc &lhs,
                                      const GfxMpsrtTensorAbiDesc &rhs,
                                      const GfxMpsrtTensorAbiDesc &output,
                                      std::string *error) {
  const uint32_t lhs_count = matrix_count_or_one(lhs);
  const uint32_t rhs_count = matrix_count_or_one(rhs);
  const uint32_t output_count = matrix_count_or_one(output);
  if (output_count == 0) {
    return fail(error, "GFX MPSRT: MPS GEMM output matrix count is zero");
  }
  if ((lhs_count != output_count && lhs_count != 1) ||
      (rhs_count != output_count && rhs_count != 1)) {
    return fail(error, "GFX MPSRT: MPS GEMM batch matrix counts must be either "
                       "1 or output matrix count");
  }
  return true;
}

bool lookup_bound_buffer(const MpsrtTensorBindings &bindings,
                         GfxMpsrtValue value, const char *name,
                         MpsrtBoundBuffer &out, std::string *error) {
  const auto *bound = bindings.lookup(value);
  if (!bound || !bound->buffer) {
    return fail(error,
                std::string("GFX MPSRT: missing tensor binding for MPS GEMM ") +
                    name);
  }
  out = *bound;
  return true;
}

bool lookup_bound_image(const MpsrtTensorBindings &bindings,
                        GfxMpsrtValue value, const char *name,
                        MpsrtBoundBuffer &out, std::string *error) {
  const auto *bound = bindings.lookup(value);
  if (!bound || !bound->texture) {
    return fail(
        error,
        std::string("GFX MPSRT: missing tensor binding for MPS Conv2D ") +
            name + " image");
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

uint32_t conv_kernel_height(const GfxMpsrtTensorAbiDesc &weights) {
  return weights.rank == 5 ? weights.dims[3] : weights.dims[2];
}

uint32_t conv_kernel_width(const GfxMpsrtTensorAbiDesc &weights) {
  return weights.rank == 5 ? weights.dims[4] : weights.dims[3];
}

NSInteger mps_conv_offset(uint32_t kernel, uint32_t dilation,
                          uint32_t pad_before) {
  const uint32_t kernel_extent = kernel == 0 ? 1 : kernel;
  const uint32_t dilation_extent = dilation == 0 ? 1 : dilation;
  const uint32_t effective_kernel =
      kernel_extent + (kernel_extent - 1) * (dilation_extent - 1);
  return static_cast<NSInteger>(effective_kernel / 2) -
         static_cast<NSInteger>(pad_before);
}

bool make_mps_image_wrapper(const GfxMpsrtTensorAbiDesc &desc,
                            const MpsrtBoundBuffer &bound, const char *name,
                            MPSImage *&out, std::string *error) {
  out = nil;
  if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Image)) {
    return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name +
                           " tensor is not image storage");
  }
  if (!bound.texture) {
    return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name +
                           " texture binding is null");
  }
  if (bound.offset != 0) {
    return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name +
                           " image binding must have zero byte offset");
  }
  if (desc.image_width == 0 || desc.image_height == 0 ||
      desc.image_feature_channels == 0 || desc.image_batch == 0) {
    return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name +
                           " image descriptor is incomplete");
  }

  id<MTLTexture> texture = static_cast<id<MTLTexture>>(bound.texture);
  if ([texture width] != desc.image_width ||
      [texture height] != desc.image_height) {
    return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name +
                           " texture shape mismatch");
  }
  const MTLPixelFormat expected_pixel_format =
      mps_image_pixel_format_from_gfx(desc.dtype);
  if (expected_pixel_format == MTLPixelFormatInvalid ||
      [texture pixelFormat] != expected_pixel_format) {
    return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name +
                           " texture pixel format mismatch");
  }

  const uint32_t slices = image_slice_count(desc.image_feature_channels);
  const NSUInteger expected_array_length =
      static_cast<NSUInteger>(desc.image_batch * slices);
  if ([texture textureType] == MTLTextureType2D) {
    if (expected_array_length != 1) {
      return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name +
                             " 2D texture cannot hold image array");
    }
  } else if ([texture textureType] != MTLTextureType2DArray ||
             [texture arrayLength] != expected_array_length) {
    return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name +
                           " texture array layout mismatch");
  }

  // MPSImage validates featureChannels against the physical texture channel
  // packing. The ABI descriptor keeps logical NCHW channels, while the Metal
  // texture is RGBA-slice backed; pass the padded physical channel count only
  // to the wrapper and keep stage descriptors/bridges logical.
  const uint32_t mps_feature_channels =
      image_slice_count(desc.image_feature_channels) * 4u;
  out = [[MPSImage alloc] initWithTexture:texture
                          featureChannels:mps_feature_channels];
  if (!out) {
    return fail(error, std::string("GFX MPSRT: failed to create MPS Conv2D ") +
                           name + " image wrapper");
  }
  if ([out numberOfImages] != desc.image_batch) {
    [out release];
    out = nil;
    return fail(error, std::string("GFX MPSRT: MPS Conv2D ") + name +
                           " image batch mismatch");
  }
  return true;
}

} // namespace

void MpsrtTensorBindings::clear() { m_bindings.clear(); }

void MpsrtTensorBindings::bind(GfxMpsrtValue value, MpsrtBoundBuffer buffer) {
  for (auto &binding : m_bindings) {
    if (binding.value == value) {
      binding.buffer = buffer;
      return;
    }
  }
  m_bindings.push_back({value, buffer});
}

const MpsrtBoundBuffer *MpsrtTensorBindings::lookup(GfxMpsrtValue value) const {
  for (const auto &binding : m_bindings) {
    if (binding.value == value) {
      return &binding.buffer;
    }
  }
  return nullptr;
}

std::vector<MpsrtBoundBuffer>
make_mpsrt_bound_buffers(const std::vector<void *> &buffers,
                         const std::vector<size_t> &offsets) {
  std::vector<MpsrtBoundBuffer> bound;
  bound.reserve(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    bound.push_back({buffers[i], i < offsets.size() ? offsets[i] : 0});
  }
  return bound;
}

MpsrtBoundBuffer make_mpsrt_bound_image(void *texture) {
  return {nullptr, 0, texture};
}

bool mpsrt_model_has_msl_dispatch(const MpsrtModel &model) {
  return std::any_of(model.stages.begin(), model.stages.end(),
                     [](const auto &stage) {
                       return stage.kind == GfxMpsrtStageKind::MSLDispatch;
                     });
}

const MpsrtRuntimeStage *
find_single_msl_dispatch_stage_with_roles(const MpsrtModel &model) {
  const MpsrtRuntimeStage *dispatch_stage = nullptr;
  for (const auto &stage : model.stages) {
    if (stage.kind != GfxMpsrtStageKind::MSLDispatch) {
      continue;
    }
    if (dispatch_stage) {
      return nullptr;
    }
    dispatch_stage = &stage;
  }
  if (!dispatch_stage || dispatch_stage->kernel_argument_roles.empty()) {
    return nullptr;
  }
  return dispatch_stage;
}

bool materialize_mpsrt_external_buffers_from_runtime_buffers(
    const MpsrtModel &model,
    const std::vector<MpsrtBoundBuffer> &runtime_buffers,
    std::vector<MpsrtBoundBuffer> &external_buffers, std::string *error) {
  external_buffers.clear();
  const size_t external_abi_count =
      runtime_mpsrt::mpsrt_model_external_buffer_abi_count(model);
  if (runtime_buffers.size() == external_abi_count) {
    return false;
  }

  const auto *dispatch_stage = find_single_msl_dispatch_stage_with_roles(model);
  if (!dispatch_stage) {
    return false;
  }
  const auto &roles = dispatch_stage->kernel_argument_roles;
  if (runtime_buffers.size() != roles.size()) {
    return fail(error, "GFX MPSRT: typed MSL runtime binding count does not "
                       "match manifest kernel roles");
  }

  std::vector<GfxMpsrtExternalBufferRole> resolved_roles;
  resolved_roles.reserve(roles.size());
  external_buffers.reserve(external_abi_count);
  for (size_t arg_index = 0; arg_index < roles.size(); ++arg_index) {
    const auto role = roles[arg_index];
    if (is_gfx_kernel_scalar_role(role)) {
      continue;
    }
    const auto mpsrt_role =
        gfx_mpsrt_external_buffer_role_from_kernel_role(role);
    if (!gfx_mpsrt_is_valid_external_buffer_role(mpsrt_role)) {
      return fail(error, "GFX MPSRT: typed MSL manifest contains non-external "
                         "kernel role");
    }
    resolved_roles.push_back(mpsrt_role);
    external_buffers.push_back(runtime_buffers[arg_index]);
  }
  if (resolved_roles != model.external_buffer_roles ||
      external_buffers.size() != external_abi_count) {
    return fail(error,
                "GFX MPSRT: typed MSL external buffer roles do not match "
                "runtime model ABI");
  }
  return true;
}

bool mpsrt_external_abi_matches_exact_binding_plan(
    const MpsrtModel &model, const KernelBindingPlan &binding_plan) {
  const auto external_abi_count =
      runtime_mpsrt::mpsrt_model_external_buffer_abi_count(model);
  if (external_abi_count == 0) {
    return false;
  }
  if (binding_plan.arg_count() != external_abi_count) {
    const auto *dispatch_stage =
        find_single_msl_dispatch_stage_with_roles(model);
    if (!dispatch_stage || dispatch_stage->kernel_argument_roles.size() !=
                               binding_plan.arg_count()) {
      return false;
    }
    const auto external_roles =
        gfx_mpsrt_external_buffer_roles_from_kernel_roles(
            dispatch_stage->kernel_argument_roles);
    if (external_roles != model.external_buffer_roles ||
        external_roles.size() != external_abi_count) {
      return false;
    }
  }
  uint32_t external_output_count = 0;
  for (const auto role : model.external_buffer_roles) {
    if (gfx_mpsrt_is_external_output_buffer_role(role)) {
      ++external_output_count;
    }
  }
  return binding_plan.output_arg_count() == 0 ||
         binding_plan.output_arg_count() == external_output_count;
}

static bool build_mpsrt_tensor_bindings(
    const MpsrtModel &model,
    const std::vector<MpsrtBoundBuffer> &external_buffers,
    MpsrtTensorBindings &bindings, MpsrtBindingBuildResult *result,
    std::string *error, const MpsrtPreparedModel *prepared_model) {
  if (result) {
    *result = {};
  }
  bindings.clear();

  if (model.resources.empty()) {
    return fail(error, "GFX MPSRT: runtime resource table is required for "
                       "external buffer bindings");
  }
  std::vector<MpsrtTensorBindingPlanEntry> plan;
  if (!runtime_mpsrt::mpsrt_model_tensor_binding_plan(model, plan, error)) {
    return false;
  }
  const size_t expected_external_buffer_count =
      runtime_mpsrt::mpsrt_model_external_buffer_abi_count(model);
  if (external_buffers.size() != expected_external_buffer_count) {
    return fail(error, "GFX MPSRT: external binding count does not match model "
                       "external buffer ABI");
  }
  return bind_tensor_binding_plan(model, external_buffers, prepared_model,
                                  bindings, result, error);
}

static bool build_mpsrt_bindings_from_runtime_buffers(
    const MpsrtModel &model, const KernelBindingPlan &binding_plan,
    const std::vector<void *> &buffer_ptrs, const std::vector<size_t> &offsets,
    const MpsrtPreparedModel *prepared_model, MpsrtTensorBindings &bindings,
    MpsrtBindingBuildResult &binding_result,
    std::vector<MpsrtImageBridgeCopy> &image_bridge_copies,
    std::vector<MpsrtBoundBuffer> *direct_msl_runtime_buffers,
    std::string *error) {
  std::vector<MpsrtBoundBuffer> runtime_buffers =
      make_mpsrt_bound_buffers(buffer_ptrs, offsets);
  std::vector<MpsrtBoundBuffer> external_buffers;
  const bool filtered_runtime_buffers =
      materialize_mpsrt_external_buffers_from_runtime_buffers(
          model, runtime_buffers, external_buffers, error);
  if (!filtered_runtime_buffers) {
    external_buffers = runtime_buffers;
  }
  const size_t external_binding_count =
      runtime_mpsrt::mpsrt_model_external_buffer_abi_count(model);
  if (external_buffers.size() == external_binding_count) {
    if (!materialize_mpsrt_image_bridge_bindings_from_plan(
            model, prepared_model, external_buffers, image_bridge_copies,
            error)) {
      return false;
    }
    if (!build_mpsrt_tensor_bindings(model, external_buffers, bindings,
                                     &binding_result, error, prepared_model)) {
      return false;
    }
    if (filtered_runtime_buffers && direct_msl_runtime_buffers &&
        image_bridge_copies.empty() && model.stages.size() == 1 &&
        model.stages.front().kind == GfxMpsrtStageKind::MSLDispatch) {
      *direct_msl_runtime_buffers = std::move(runtime_buffers);
    }
    return true;
  }

  if (mpsrt_model_has_msl_dispatch(model)) {
    return fail(error, "GFX MPSRT: typed MSL dispatch requires exact external "
                       "buffer ABI bindings");
  }

  const size_t input_count = model.input_values.size();
  const size_t output_count = model.output_values.size();
  if (external_buffers.size() < input_count) {
    return fail(error, "GFX MPSRT: runtime buffers do not cover model inputs");
  }

  std::vector<MpsrtBoundBuffer> input_buffers(
      external_buffers.begin(), external_buffers.begin() + input_count);
  std::vector<MpsrtBoundBuffer> output_buffers;
  if (output_count != 0) {
    if (binding_plan.output_arg_count() != 0) {
      if (binding_plan.output_arg_count() != output_count) {
        return fail(error,
                    "GFX MPSRT: output arg count does not match model outputs");
      }
      if (external_buffers.size() < output_count) {
        return fail(error,
                    "GFX MPSRT: runtime buffers do not cover model outputs");
      }
      output_buffers.assign(external_buffers.end() - output_count,
                            external_buffers.end());
    } else if (external_buffers.size() >= input_count + output_count) {
      output_buffers.assign(external_buffers.begin() + input_count,
                            external_buffers.begin() + input_count +
                                output_count);
    } else if (external_buffers.size() == input_count &&
               input_count == output_count) {
      output_buffers = input_buffers;
    } else {
      return fail(
          error,
          "GFX MPSRT: cannot infer model output bindings from runtime buffers");
    }
  }

  if (!materialize_mpsrt_image_bridge_bindings(
          model, prepared_model, model.input_values,
          GfxMpsrtStorageBridgeDirection::BufferToImage, input_buffers,
          image_bridge_copies, error) ||
      !materialize_mpsrt_image_bridge_bindings(
          model, prepared_model, model.output_values,
          GfxMpsrtStorageBridgeDirection::ImageToBuffer, output_buffers,
          image_bridge_copies, error)) {
    return false;
  }

  std::vector<MpsrtBoundBuffer> vendor_only_external_buffers;
  vendor_only_external_buffers.reserve(input_buffers.size() +
                                       output_buffers.size());
  vendor_only_external_buffers.insert(vendor_only_external_buffers.end(),
                                      input_buffers.begin(),
                                      input_buffers.end());
  vendor_only_external_buffers.insert(vendor_only_external_buffers.end(),
                                      output_buffers.begin(),
                                      output_buffers.end());
  return build_mpsrt_tensor_bindings(model, vendor_only_external_buffers,
                                     bindings, &binding_result, error,
                                     prepared_model);
}

static void
record_mpsrt_request_binding_counters(const KernelExecutionHooks *hooks,
                                      const MpsrtBindingBuildResult &result,
                                      size_t image_bridge_copy_count) {
  if (!hooks || !hooks->on_counter) {
    return;
  }
  hooks->on_counter("mpsrt_binding_external_input_count",
                    static_cast<uint64_t>(result.external_inputs_bound));
  hooks->on_counter("mpsrt_binding_external_output_count",
                    static_cast<uint64_t>(result.external_outputs_bound));
  hooks->on_counter("mpsrt_binding_external_resource_count",
                    static_cast<uint64_t>(result.external_resources_bound));
  hooks->on_counter("mpsrt_binding_model_resource_count",
                    static_cast<uint64_t>(result.model_resources_bound));
  hooks->on_counter("mpsrt_binding_prepared_transient_buffer_count",
                    static_cast<uint64_t>(result.transient_buffers_allocated));
  hooks->on_counter("mpsrt_binding_prepared_transient_image_count",
                    static_cast<uint64_t>(result.transient_images_allocated));
  hooks->on_counter("mpsrt_image_bridge_copy_count",
                    static_cast<uint64_t>(image_bridge_copy_count));
}

bool MpsrtRequest::build_binding_set_from_runtime_buffers(
    const MpsrtModel &model, const KernelBindingPlan &binding_plan,
    const std::vector<void *> &buffer_ptrs, const std::vector<size_t> &offsets,
    const MpsrtPreparedModel *prepared_model,
    MpsrtRequestBindingSet &binding_set, const KernelExecutionHooks *hooks,
    std::string *error) const {
  binding_set.bindings.clear();
  binding_set.image_bridge_copies.clear();
  binding_set.direct_msl_runtime_buffers.clear();
  MpsrtBindingBuildResult result;
  if (!build_mpsrt_bindings_from_runtime_buffers(
          model, binding_plan, buffer_ptrs, offsets, prepared_model,
          binding_set.bindings, result, binding_set.image_bridge_copies,
          &binding_set.direct_msl_runtime_buffers, error)) {
    return false;
  }
  record_mpsrt_request_binding_counters(hooks, result,
                                        binding_set.image_bridge_copies.size());
  return true;
}

bool MpsrtRequest::build_binding_set_from_external_buffers(
    const MpsrtModel &model,
    const std::vector<MpsrtBoundBuffer> &external_buffers,
    const MpsrtPreparedModel *prepared_model,
    MpsrtRequestBindingSet &binding_set, const KernelExecutionHooks *hooks,
    std::string *error) const {
  binding_set.bindings.clear();
  binding_set.image_bridge_copies.clear();
  binding_set.direct_msl_runtime_buffers.clear();
  MpsrtBindingBuildResult result;
  std::vector<MpsrtBoundBuffer> bridge_resolved_external_buffers =
      external_buffers;
  if (!materialize_mpsrt_image_bridge_bindings_from_plan(
          model, prepared_model, bridge_resolved_external_buffers,
          binding_set.image_bridge_copies, error)) {
    return false;
  }
  if (!build_mpsrt_tensor_bindings(model, bridge_resolved_external_buffers,
                                   binding_set.bindings, &result, error,
                                   prepared_model)) {
    return false;
  }
  record_mpsrt_request_binding_counters(hooks, result,
                                        binding_set.image_bridge_copies.size());
  return true;
}

MpsrtPreparedMslDispatch
make_prepared_msl_dispatch_from_pipeline(const MpsrtRuntimeStage &stage,
                                         size_t stage_index,
                                         id<MTLComputePipelineState> pipeline) {
  OPENVINO_ASSERT(
      stage.kind == GfxMpsrtStageKind::MSLDispatch,
      "GFX MPSRT: cannot prepare non-MSL stage from Metal pipeline");
  OPENVINO_ASSERT(pipeline, "GFX MPSRT: Metal pipeline is null");

  MpsrtPreparedMslDispatch prepared;
  prepared.stage_index = stage_index;
  prepared.stage_record_key = stage.stage_record_key;
  prepared.dispatch_entry_point = stage.dispatch_entry_point;
  prepared.dispatch_kernel_family_id = stage.dispatch_kernel_family_id;
  prepared.dispatch_threads_per_threadgroup =
      stage.dispatch_threads_per_threadgroup;
  prepared.thread_execution_width =
      static_cast<uint32_t>([pipeline threadExecutionWidth]);
  prepared.max_total_threads_per_threadgroup =
      static_cast<uint32_t>([pipeline maxTotalThreadsPerThreadgroup]);
  prepared.pipeline_cache_hit = true;
  prepared.pipeline = pipeline;
  return prepared;
}

bool MpsrtRequest::encode_msl_dispatch(
    GpuCommandBufferHandle command_buffer,
    const MpsrtPreparedMslDispatch &prepared, const KernelDispatch &dispatch,
    const std::vector<MpsrtBoundBuffer> &buffers,
    const KernelExecutionHooks *hooks, MpsrtMslEncodeResult *result) const {
  if (result) {
    *result = {};
  }
  OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
  OPENVINO_ASSERT(prepared.pipeline,
                  "GFX MPSRT: prepared MSL pipeline is null");

  const auto setup_start = hooks && (hooks->on_segment || hooks->on_counter)
                               ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  bool encoder_created = false;
  id<MTLComputeCommandEncoder> enc = static_cast<id<MTLComputeCommandEncoder>>(
      metal_get_or_create_compute_encoder(command_buffer, &encoder_created));
  OPENVINO_ASSERT(enc, "GFX MPSRT: failed to create compute encoder");

  const bool pipeline_bound = metal_set_compute_pipeline_if_needed(
      command_buffer, reinterpret_cast<GpuCommandEncoderHandle>(enc),
      prepared.pipeline);

  const bool has_textures =
      std::any_of(buffers.begin(), buffers.end(),
                  [](const auto &buffer) { return buffer.texture != nullptr; });
  size_t bound_buffers = 0;
  size_t bound_textures = 0;
  if (!has_textures) {
    std::vector<void *> raw_buffers;
    std::vector<size_t> offsets;
    raw_buffers.reserve(buffers.size());
    offsets.reserve(buffers.size());
    for (const auto &buffer : buffers) {
      OPENVINO_ASSERT(buffer.buffer, "GFX MPSRT: bound MSL buffer is null");
      raw_buffers.push_back(buffer.buffer);
      offsets.push_back(buffer.offset);
    }
    bound_buffers = metal_bind_compute_buffers_if_needed(
        command_buffer, reinterpret_cast<GpuCommandEncoderHandle>(enc),
        raw_buffers, offsets);
  } else {
    for (const auto &binding : buffers) {
      if (binding.texture) {
        [enc setTexture:static_cast<id<MTLTexture>>(binding.texture)
                atIndex:bound_textures++];
        continue;
      }
      OPENVINO_ASSERT(binding.buffer, "GFX MPSRT: bound MSL resource is null");
      [enc setBuffer:static_cast<id<MTLBuffer>>(binding.buffer)
              offset:static_cast<NSUInteger>(binding.offset)
             atIndex:bound_buffers++];
    }
  }

  if (result) {
    result->encoder_created = encoder_created;
    result->pipeline_bound = pipeline_bound;
    result->bound_buffers = bound_buffers;
    result->bound_textures = bound_textures;
  }
  if (hooks && hooks->on_counter) {
    hooks->on_counter("mpsrt_msl_request_encode_count", 1);
    if (encoder_created) {
      hooks->on_counter("mpsrt_msl_encoder_create_count", 1);
    }
    if (pipeline_bound) {
      hooks->on_counter("mpsrt_msl_pipeline_bind_count", 1);
    }
    hooks->on_counter("mpsrt_msl_bound_buffer_count",
                      static_cast<uint64_t>(bound_buffers));
    if (bound_textures != 0) {
      hooks->on_counter("mpsrt_msl_bound_texture_count",
                        static_cast<uint64_t>(bound_textures));
    }
  }
  if (hooks && hooks->on_segment) {
    const auto setup_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - setup_start);
    hooks->on_segment("mpsrt_encode", prepared.dispatch_entry_point,
                      setup_cpu_us, 0,
                      static_cast<uint32_t>(bound_buffers + bound_textures), 0,
                      0, 0, 0, -1, 0,
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
  MTLSize tg =
      MTLSizeMake(dispatch.threads_per_group[0], dispatch.threads_per_group[1],
                  dispatch.threads_per_group[2]);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];

  if (hooks && hooks->on_end) {
    hooks->on_end(enc);
  }
  return true;
}

bool MpsrtRequest::build_msl_stage_buffers(
    const MpsrtRuntimeStage &stage, const MpsrtTensorBindings &bindings,
    std::vector<MpsrtBoundBuffer> &buffers,
    const std::vector<MpsrtBoundBuffer> *direct_runtime_buffers,
    std::string *error) const {
  buffers.clear();
  if (stage.kind != GfxMpsrtStageKind::MSLDispatch) {
    return fail(error, "GFX MPSRT: cannot bind buffers for non-MSL stage");
  }
  if (direct_runtime_buffers && !direct_runtime_buffers->empty()) {
    if (stage.kernel_argument_roles.empty() ||
        direct_runtime_buffers->size() != stage.kernel_argument_roles.size()) {
      return fail(error, "GFX MPSRT: direct MSL runtime buffers do not match "
                         "manifest kernel roles");
    }
    buffers = *direct_runtime_buffers;
    return true;
  }
  const auto &buffer_order = stage.kernel_buffer_order;
  if (buffer_order.empty()) {
    return fail(error,
                "GFX MPSRT: MSL stage kernel buffer order is not materialized");
  }
  if (stage.msl_dispatch_desc.input_count +
          stage.msl_dispatch_desc.output_count !=
      buffer_order.size()) {
    return fail(error,
                "GFX MPSRT: MSL stage kernel buffer order metadata mismatch");
  }

  buffers.reserve(buffer_order.size());
  for (const auto value : buffer_order) {
    const auto *bound = bindings.lookup(value);
    if (!bound || (!bound->buffer && !bound->texture)) {
      return fail(error,
                  "GFX MPSRT: missing tensor binding for kernel buffer value " +
                      std::to_string(value));
    }
    buffers.push_back(*bound);
  }
  return true;
}

bool MpsrtRequest::encode_mps_gemm(
    GpuCommandBufferHandle command_buffer, const MpsrtModel &model,
    const MpsrtRuntimeStage &stage, const MpsrtPreparedMpsGemm &prepared,
    const MpsrtTensorBindings &bindings, const KernelExecutionHooks *hooks,
    MpsrtMpsGemmEncodeResult *result, std::string *error) const {
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
  if (stage.inputs.size() != 2 || stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return fail(error,
                "GFX MPSRT: MPS GEMM requires two inputs and one output");
  }

  const auto *lhs_tensor = find_tensor(model, stage.inputs[0]);
  const auto *rhs_tensor = find_tensor(model, stage.inputs[1]);
  if (!lhs_tensor || !rhs_tensor) {
    return fail(error,
                "GFX MPSRT: MPS GEMM input tensor descriptor is missing");
  }

  MpsrtBoundBuffer lhs_buffer;
  MpsrtBoundBuffer rhs_buffer;
  MpsrtBoundBuffer output_buffer;
  if (!lookup_bound_buffer(bindings, stage.inputs[0], "lhs", lhs_buffer,
                           error) ||
      !lookup_bound_buffer(bindings, stage.inputs[1], "rhs", rhs_buffer,
                           error) ||
      !lookup_bound_buffer(bindings, stage.outputs[0], "output", output_buffer,
                           error)) {
    return false;
  }

  MPSMatrixDescriptor *lhs_desc = nil;
  MPSMatrixDescriptor *rhs_desc = nil;
  MPSMatrixDescriptor *output_desc = nil;
  if (!make_mps_matrix_descriptor(lhs_tensor->desc, lhs_desc, "lhs", error) ||
      !make_mps_matrix_descriptor(rhs_tensor->desc, rhs_desc, "rhs", error) ||
      !make_mps_matrix_descriptor(stage.output_descs.front(), output_desc,
                                  "output", error)) {
    return false;
  }
  if (!validate_mps_gemm_batch_contract(lhs_tensor->desc, rhs_tensor->desc,
                                        stage.output_descs.front(), error)) {
    return false;
  }

  if (prepared.uses_mps_graph_gemm) {
    if (!prepared.graph_lhs_tensor || !prepared.graph_rhs_tensor ||
        !prepared.graph_output_tensor) {
      return fail(error, "GFX MPSRT: prepared MPSGraph GEMM state is incomplete");
    }
    const MPSDataType data_type = mps_data_type_from_gfx(prepared.data_type);
    if (data_type == MPSDataTypeInvalid) {
      return fail(error, "GFX MPSRT: MPSGraph GEMM dtype is unsupported");
    }

    const NSUInteger lhs_offset =
        static_cast<NSUInteger>(lhs_buffer.offset + lhs_tensor->desc.byte_offset);
    const NSUInteger rhs_offset =
        static_cast<NSUInteger>(rhs_buffer.offset + rhs_tensor->desc.byte_offset);
    const NSUInteger output_offset =
        static_cast<NSUInteger>(output_buffer.offset +
                                stage.output_descs.front().byte_offset);
    if (lhs_offset != 0 || rhs_offset != 0 || output_offset != 0) {
      return fail(error,
                  "GFX MPSRT: MPSGraph GEMM requires zero-offset dense buffers");
    }

    metal_end_compute_encoder(command_buffer);
    id<MTLCommandBuffer> command =
        static_cast<id<MTLCommandBuffer>>(command_buffer);
    MPSCommandBuffer *mps_command =
        [MPSCommandBuffer commandBufferWithCommandBuffer:command];
    MPSGraph *graph = static_cast<MPSGraph *>(prepared.kernel);
    NSArray<NSNumber *> *lhs_shape = mps_shape_from_tensor_desc(lhs_tensor->desc);
    NSArray<NSNumber *> *rhs_shape = mps_shape_from_tensor_desc(rhs_tensor->desc);
    NSArray<NSNumber *> *output_shape =
        mps_shape_from_tensor_desc(stage.output_descs.front());
    MPSGraphTensorData *lhs_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:static_cast<id<MTLBuffer>>(lhs_buffer.buffer)
                    shape:lhs_shape
                 dataType:data_type];
    MPSGraphTensorData *rhs_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:static_cast<id<MTLBuffer>>(rhs_buffer.buffer)
                    shape:rhs_shape
                 dataType:data_type];
    MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:static_cast<id<MTLBuffer>>(output_buffer.buffer)
                    shape:output_shape
                 dataType:data_type];
    if (!mps_command || !graph || !lhs_data || !rhs_data || !output_data) {
      [lhs_data release];
      [rhs_data release];
      [output_data release];
      return fail(error, "GFX MPSRT: failed to bind MPSGraph GEMM tensors");
    }

    const auto encode_start = hooks && hooks->on_segment
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
    bool encoded_executable = false;
    if (prepared.graph_executable) {
      MPSGraphExecutable *executable =
          static_cast<MPSGraphExecutable *>(prepared.graph_executable);
      NSMutableArray<MPSGraphTensorData *> *inputs = [NSMutableArray array];
      NSArray<MPSGraphTensor *> *feed_tensors = executable.feedTensors;
      if (feed_tensors.count == 0) {
        [inputs addObject:lhs_data];
        [inputs addObject:rhs_data];
      } else {
        for (MPSGraphTensor *feed_tensor in feed_tensors) {
          if (feed_tensor ==
              static_cast<MPSGraphTensor *>(prepared.graph_lhs_tensor)) {
            [inputs addObject:lhs_data];
          } else if (feed_tensor ==
                     static_cast<MPSGraphTensor *>(
                         prepared.graph_rhs_tensor)) {
            [inputs addObject:rhs_data];
          } else {
            [lhs_data release];
            [rhs_data release];
            [output_data release];
            return fail(error,
                        "GFX MPSRT: MPSGraph GEMM executable feed tensor is unknown");
          }
        }
      }
      [executable encodeToCommandBuffer:mps_command
                            inputsArray:inputs
                           resultsArray:@[ output_data ]
                    executionDescriptor:nil];
      encoded_executable = true;
    } else {
      MPSGraphTensorDataDictionary *feeds =
          [NSDictionary dictionaryWithObjectsAndKeys:
                            lhs_data,
                            static_cast<MPSGraphTensor *>(
                                prepared.graph_lhs_tensor),
                            rhs_data,
                            static_cast<MPSGraphTensor *>(
                                prepared.graph_rhs_tensor),
                            nil];
      MPSGraphTensorDataDictionary *results =
          [NSDictionary dictionaryWithObjectsAndKeys:
                            output_data,
                            static_cast<MPSGraphTensor *>(
                                prepared.graph_output_tensor),
                            nil];
      [graph encodeToCommandBuffer:mps_command
                             feeds:feeds
                  targetOperations:nil
                 resultsDictionary:results
               executionDescriptor:nil];
    }

    [lhs_data release];
    [rhs_data release];
    [output_data release];

    if (result) {
      result->bound_buffers = 3;
      result->kernel_encodes = 1;
    }
    if (hooks && hooks->on_counter) {
      hooks->on_counter("mpsrt_mps_gemm_request_encode_count", 1);
      hooks->on_counter("mpsrt_mps_gemm_kernel_encode_count", 1);
      hooks->on_counter("mpsrt_mps_gemm_bound_buffer_count", 3);
      hooks->on_counter("mpsrt_mps_graph_gemm_request_encode_count", 1);
      hooks->on_counter("mpsrt_mps_graph_gemm_kernel_encode_count", 1);
      if (encoded_executable) {
        hooks->on_counter("mpsrt_mps_graph_gemm_executable_encode_count", 1);
      }
    }
    if (hooks && hooks->on_segment) {
      const auto setup_cpu_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - encode_start);
      hooks->on_segment("mpsrt_encode", stage.stage_record_key, setup_cpu_us,
                        0, 3, 0, 0, 0, 0, -1, 0,
                        reinterpret_cast<uint64_t>(command_buffer));
    }
    return true;
  }

  metal_end_compute_encoder(command_buffer);
  id<MTLCommandBuffer> command =
      static_cast<id<MTLCommandBuffer>>(command_buffer);
  const auto encode_start = hooks && hooks->on_segment
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
  const uint32_t output_count = matrix_count_or_one(stage.output_descs.front());
  const bool needs_batch_loop =
      output_count > 1 &&
      (matrix_count_or_one(lhs_tensor->desc) != output_count ||
       matrix_count_or_one(rhs_tensor->desc) != output_count);
  size_t kernel_encodes = 0;
  if (!needs_batch_loop) {
    MPSMatrix *lhs_matrix = [[MPSMatrix alloc]
        initWithBuffer:static_cast<id<MTLBuffer>>(lhs_buffer.buffer)
                offset:static_cast<NSUInteger>(lhs_buffer.offset +
                                               lhs_tensor->desc.byte_offset)
            descriptor:lhs_desc];
    MPSMatrix *rhs_matrix = [[MPSMatrix alloc]
        initWithBuffer:static_cast<id<MTLBuffer>>(rhs_buffer.buffer)
                offset:static_cast<NSUInteger>(rhs_buffer.offset +
                                               rhs_tensor->desc.byte_offset)
            descriptor:rhs_desc];
    MPSMatrix *output_matrix = [[MPSMatrix alloc]
        initWithBuffer:static_cast<id<MTLBuffer>>(output_buffer.buffer)
                offset:static_cast<NSUInteger>(
                           output_buffer.offset +
                           stage.output_descs.front().byte_offset)
            descriptor:output_desc];
    if (!lhs_matrix || !rhs_matrix || !output_matrix) {
      [lhs_matrix release];
      [rhs_matrix release];
      [output_matrix release];
      return fail(error,
                  "GFX MPSRT: failed to create MPS GEMM matrix wrappers");
    }

    [(MPSMatrixMultiplication *)prepared.kernel
        encodeToCommandBuffer:command
                   leftMatrix:lhs_matrix
                  rightMatrix:rhs_matrix
                 resultMatrix:output_matrix];
    [lhs_matrix release];
    [rhs_matrix release];
    [output_matrix release];
    kernel_encodes = 1;
  } else {
    MPSMatrixDescriptor *single_lhs_desc = nil;
    MPSMatrixDescriptor *single_rhs_desc = nil;
    MPSMatrixDescriptor *single_output_desc = nil;
    if (!make_mps_matrix_descriptor(lhs_tensor->desc, single_lhs_desc, "lhs",
                                    error, 1) ||
        !make_mps_matrix_descriptor(rhs_tensor->desc, single_rhs_desc, "rhs",
                                    error, 1) ||
        !make_mps_matrix_descriptor(stage.output_descs.front(),
                                    single_output_desc, "output", error, 1)) {
      return false;
    }
    for (uint32_t batch = 0; batch < output_count; ++batch) {
      MPSMatrix *lhs_matrix = [[MPSMatrix alloc]
          initWithBuffer:static_cast<id<MTLBuffer>>(lhs_buffer.buffer)
                  offset:static_cast<NSUInteger>(
                             lhs_buffer.offset +
                             matrix_batch_offset(lhs_tensor->desc, batch))
              descriptor:single_lhs_desc];
      MPSMatrix *rhs_matrix = [[MPSMatrix alloc]
          initWithBuffer:static_cast<id<MTLBuffer>>(rhs_buffer.buffer)
                  offset:static_cast<NSUInteger>(
                             rhs_buffer.offset +
                             matrix_batch_offset(rhs_tensor->desc, batch))
              descriptor:single_rhs_desc];
      MPSMatrix *output_matrix = [[MPSMatrix alloc]
          initWithBuffer:static_cast<id<MTLBuffer>>(output_buffer.buffer)
                  offset:static_cast<NSUInteger>(
                             output_buffer.offset +
                             matrix_batch_offset(stage.output_descs.front(),
                                                 batch))
              descriptor:single_output_desc];
      if (!lhs_matrix || !rhs_matrix || !output_matrix) {
        [lhs_matrix release];
        [rhs_matrix release];
        [output_matrix release];
        return fail(
            error,
            "GFX MPSRT: failed to create MPS GEMM broadcast matrix wrappers");
      }

      [(MPSMatrixMultiplication *)prepared.kernel
          encodeToCommandBuffer:command
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
    const auto setup_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
    hooks->on_segment("mpsrt_encode", stage.stage_record_key, setup_cpu_us, 0,
                      3, 0, 0, 0, 0, -1, 0,
                      reinterpret_cast<uint64_t>(command_buffer));
  }
  return true;
}

bool MpsrtRequest::encode_mps_conv2d(
    GpuCommandBufferHandle command_buffer, const MpsrtModel &model,
    const MpsrtRuntimeStage &stage, const MpsrtPreparedMpsConv2D &prepared,
    const MpsrtTensorBindings &bindings, const KernelExecutionHooks *hooks,
    MpsrtMpsConv2DEncodeResult *result, std::string *error) const {
  if (result) {
    *result = {};
  }
  OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
  if (!is_mps_conv2d_stage(stage.kind)) {
    return fail(error,
                "GFX MPSRT: cannot encode non-Conv2D stage with MPS Conv2D");
  }
  if (!prepared.weights_buffer) {
    return fail(error, "GFX MPSRT: prepared MPS Conv2D weights buffer is null");
  }
  if (!prepared.kernel) {
    return fail(error, "GFX MPSRT: prepared MPS Conv2D kernel is null");
  }
  if ((stage.inputs.size() != 2 && stage.inputs.size() != 3) ||
      stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return fail(error,
                "GFX MPSRT: MPS Conv2D requires input, weights, optional bias and one output");
  }

  const auto *input_tensor = find_tensor(model, stage.inputs[0]);
  const auto *weights_tensor = find_tensor(model, stage.inputs[1]);
  if (!input_tensor || !weights_tensor) {
    return fail(
        error,
        "GFX MPSRT: MPS Conv2D input or weights tensor descriptor is missing");
  }

  MpsrtBoundBuffer input_binding;
  MpsrtBoundBuffer output_binding;
  if (!lookup_bound_image(bindings, stage.inputs[0], "input", input_binding,
                          error) ||
      !lookup_bound_image(bindings, stage.outputs[0], "output", output_binding,
                          error)) {
    return false;
  }

  MPSImage *input_image = nil;
  MPSImage *output_image = nil;
  if (!make_mps_image_wrapper(input_tensor->desc, input_binding, "input",
                              input_image, error) ||
      !make_mps_image_wrapper(stage.output_descs.front(), output_binding,
                              "output", output_image, error)) {
    [input_image release];
    [output_image release];
    return false;
  }

  metal_end_compute_encoder(command_buffer);
  id<MTLCommandBuffer> command =
      static_cast<id<MTLCommandBuffer>>(command_buffer);
  const auto encode_start = hooks && hooks->on_segment
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};

  MPSCNNConvolution *kernel = static_cast<MPSCNNConvolution *>(prepared.kernel);
  kernel.edgeMode = MPSImageEdgeModeZero;
  kernel.offset = (MPSOffset){
      .x = mps_conv_offset(conv_kernel_width(weights_tensor->desc),
                           stage.conv2d_desc.dilations[1] == 0
                               ? 1
                               : stage.conv2d_desc.dilations[1],
                           stage.conv2d_desc.pads[1]),
      .y = mps_conv_offset(conv_kernel_height(weights_tensor->desc),
                           stage.conv2d_desc.dilations[0] == 0
                               ? 1
                               : stage.conv2d_desc.dilations[0],
                           stage.conv2d_desc.pads[0]),
      .z = 0,
  };
  kernel.clipRect =
      MTLRegionMake3D(0, 0, 0, stage.output_descs.front().image_width,
                      stage.output_descs.front().image_height,
                      stage.output_descs.front().image_batch);

  [kernel encodeToCommandBuffer:command
                    sourceImage:input_image
               destinationImage:output_image];
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
    const auto setup_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
    hooks->on_segment("mpsrt_encode", stage.stage_record_key, setup_cpu_us, 0,
                      2, 0, 0, 0, 0, -1, 0,
                      reinterpret_cast<uint64_t>(command_buffer));
  }
  return true;
}

bool MpsrtRequest::encode_mps_pool2d(
    GpuCommandBufferHandle command_buffer, const MpsrtModel &model,
    const MpsrtRuntimeStage &stage, const MpsrtPreparedMpsPool2D &prepared,
    const MpsrtTensorBindings &bindings, const KernelExecutionHooks *hooks,
    MpsrtMpsPool2DEncodeResult *result, std::string *error) const {
  if (result) {
    *result = {};
  }
  OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
  if (stage.kind != GfxMpsrtStageKind::MPSPool2D) {
    return fail(error,
                "GFX MPSRT: cannot encode non-Pool2D stage with MPS Pool2D");
  }
  if (!prepared.kernel) {
    return fail(error, "GFX MPSRT: prepared MPS Pool2D kernel is null");
  }
  if (stage.inputs.size() != 1 || stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return fail(error,
                "GFX MPSRT: MPS Pool2D requires one input and one output");
  }

  const auto *input_tensor = find_tensor(model, stage.inputs[0]);
  if (!input_tensor) {
    return fail(error,
                "GFX MPSRT: MPS Pool2D input tensor descriptor is missing");
  }

  MpsrtBoundBuffer input_binding;
  MpsrtBoundBuffer output_binding;
  if (!lookup_bound_image(bindings, stage.inputs[0], "input", input_binding,
                          error) ||
      !lookup_bound_image(bindings, stage.outputs[0], "output", output_binding,
                          error)) {
    return false;
  }

  MPSImage *input_image = nil;
  MPSImage *output_image = nil;
  if (!make_mps_image_wrapper(input_tensor->desc, input_binding, "input",
                              input_image, error) ||
      !make_mps_image_wrapper(stage.output_descs.front(), output_binding,
                              "output", output_image, error)) {
    [input_image release];
    [output_image release];
    return false;
  }

  metal_end_compute_encoder(command_buffer);
  id<MTLCommandBuffer> command =
      static_cast<id<MTLCommandBuffer>>(command_buffer);
  const auto encode_start = hooks && hooks->on_segment
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};

  MPSCNNPooling *kernel = static_cast<MPSCNNPooling *>(prepared.kernel);
  kernel.edgeMode = MPSImageEdgeModeZero;
  kernel.offset = (MPSOffset){
      .x = mps_conv_offset(stage.pool2d_desc.kernel[1], 1,
                           stage.pool2d_desc.pads[1]),
      .y = mps_conv_offset(stage.pool2d_desc.kernel[0], 1,
                           stage.pool2d_desc.pads[0]),
      .z = 0,
  };
  kernel.clipRect =
      MTLRegionMake3D(0, 0, 0, stage.output_descs.front().image_width,
                      stage.output_descs.front().image_height,
                      stage.output_descs.front().image_batch);

  [kernel encodeToCommandBuffer:command
                    sourceImage:input_image
               destinationImage:output_image];
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
    const auto setup_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
    hooks->on_segment("mpsrt_encode", stage.stage_record_key, setup_cpu_us, 0,
                      2, 0, 0, 0, 0, -1, 0,
                      reinterpret_cast<uint64_t>(command_buffer));
  }
  return true;
}

bool MpsrtRequest::encode_mps_resize2d(
    GpuCommandBufferHandle command_buffer, const MpsrtModel &model,
    const MpsrtRuntimeStage &stage, const MpsrtPreparedMpsResize2D &prepared,
    const MpsrtTensorBindings &bindings, const KernelExecutionHooks *hooks,
    MpsrtMpsResize2DEncodeResult *result, std::string *error) const {
  if (result) {
    *result = {};
  }
  OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
  if (stage.kind != GfxMpsrtStageKind::MPSResize2D) {
    return fail(
        error, "GFX MPSRT: cannot encode non-Resize2D stage with MPS Resize2D");
  }
  if (!prepared.kernel) {
    return fail(error, "GFX MPSRT: prepared MPS Resize2D kernel is null");
  }
  if (stage.resize2d_desc.nearest != 0) {
    return fail(error,
                "GFX MPSRT: MPS Resize2D encode supports bilinear mode only");
  }
  if (stage.inputs.size() != 1 || stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return fail(error,
                "GFX MPSRT: MPS Resize2D requires one input and one output");
  }

  const auto *input_tensor = find_tensor(model, stage.inputs[0]);
  if (!input_tensor) {
    return fail(error,
                "GFX MPSRT: MPS Resize2D input tensor descriptor is missing");
  }

  MpsrtBoundBuffer input_binding;
  MpsrtBoundBuffer output_binding;
  if (!lookup_bound_image(bindings, stage.inputs[0], "input", input_binding,
                          error) ||
      !lookup_bound_image(bindings, stage.outputs[0], "output", output_binding,
                          error)) {
    return false;
  }

  MPSImage *input_image = nil;
  MPSImage *output_image = nil;
  if (!make_mps_image_wrapper(input_tensor->desc, input_binding, "input",
                              input_image, error) ||
      !make_mps_image_wrapper(stage.output_descs.front(), output_binding,
                              "output", output_image, error)) {
    [input_image release];
    [output_image release];
    return false;
  }

  metal_end_compute_encoder(command_buffer);
  id<MTLCommandBuffer> command =
      static_cast<id<MTLCommandBuffer>>(command_buffer);
  const auto encode_start = hooks && hooks->on_segment
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};

  MPSImageBilinearScale *kernel =
      static_cast<MPSImageBilinearScale *>(prepared.kernel);
  kernel.edgeMode = MPSImageEdgeModeClamp;
  kernel.clipRect =
      MTLRegionMake3D(0, 0, 0, stage.output_descs.front().image_width,
                      stage.output_descs.front().image_height,
                      stage.output_descs.front().image_batch);
  kernel.scaleTransform = nullptr;
  [kernel encodeToCommandBuffer:command
                    sourceImage:input_image
               destinationImage:output_image];
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
    const auto setup_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
    hooks->on_segment("mpsrt_encode", stage.stage_record_key, setup_cpu_us, 0,
                      2, 0, 0, 0, 0, -1, 0,
                      reinterpret_cast<uint64_t>(command_buffer));
  }
  return true;
}

bool MpsrtRequest::encode_mps_softmax(
    GpuCommandBufferHandle command_buffer, const MpsrtModel &model,
    const MpsrtRuntimeStage &stage, const MpsrtPreparedMpsSoftmax &prepared,
    const MpsrtTensorBindings &bindings, const KernelExecutionHooks *hooks,
    MpsrtMpsSoftmaxEncodeResult *result, std::string *error) const {
  if (result) {
    *result = {};
  }
  OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
  if (stage.kind != GfxMpsrtStageKind::MPSSoftmax) {
    return fail(error,
                "GFX MPSRT: cannot encode non-Softmax stage with MPS Softmax");
  }
  if (!prepared.kernel) {
    return fail(error, "GFX MPSRT: prepared MPS Softmax kernel is null");
  }
  if (stage.softmax_desc.log_softmax != 0) {
    return fail(error,
                "GFX MPSRT: MPS Softmax encode does not implement LogSoftmax");
  }
  if (stage.inputs.size() != 1 || stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return fail(error,
                "GFX MPSRT: MPS Softmax requires one input and one output");
  }

  const auto *input_tensor = find_tensor(model, stage.inputs[0]);
  if (!input_tensor) {
    return fail(error,
                "GFX MPSRT: MPS Softmax input tensor descriptor is missing");
  }

  MpsrtBoundBuffer input_buffer;
  MpsrtBoundBuffer output_buffer;
  if (!lookup_bound_buffer(bindings, stage.inputs[0], "input", input_buffer,
                           error) ||
      !lookup_bound_buffer(bindings, stage.outputs[0], "output", output_buffer,
                           error)) {
    return false;
  }

  MPSMatrixDescriptor *input_desc = nil;
  MPSMatrixDescriptor *output_desc = nil;
  if (!make_mps_matrix_descriptor(input_tensor->desc, input_desc, "input",
                                  error) ||
      !make_mps_matrix_descriptor(stage.output_descs.front(), output_desc,
                                  "output", error)) {
    return false;
  }

  MPSMatrix *input_matrix = [[MPSMatrix alloc]
      initWithBuffer:static_cast<id<MTLBuffer>>(input_buffer.buffer)
              offset:static_cast<NSUInteger>(input_buffer.offset +
                                             input_tensor->desc.byte_offset)
          descriptor:input_desc];
  MPSMatrix *output_matrix = [[MPSMatrix alloc]
      initWithBuffer:static_cast<id<MTLBuffer>>(output_buffer.buffer)
              offset:static_cast<NSUInteger>(
                         output_buffer.offset +
                         stage.output_descs.front().byte_offset)
          descriptor:output_desc];
  if (!input_matrix || !output_matrix) {
    [input_matrix release];
    [output_matrix release];
    return fail(error,
                "GFX MPSRT: failed to create MPS Softmax matrix wrappers");
  }

  metal_end_compute_encoder(command_buffer);
  id<MTLCommandBuffer> command =
      static_cast<id<MTLCommandBuffer>>(command_buffer);
  const auto encode_start = hooks && hooks->on_segment
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};

  [(MPSMatrixSoftMax *)prepared.kernel encodeToCommandBuffer:command
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
    const auto setup_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
    hooks->on_segment("mpsrt_encode", stage.stage_record_key, setup_cpu_us, 0,
                      2, 0, 0, 0, 0, -1, 0,
                      reinterpret_cast<uint64_t>(command_buffer));
  }
  return true;
}

bool MpsrtRequest::encode_mps_topk(
    GpuCommandBufferHandle command_buffer, MpsrtContext &context,
    const MpsrtModel &model,
    const MpsrtRuntimeStage &stage, const MpsrtPreparedMpsTopK &prepared,
    const MpsrtTensorBindings &bindings, const KernelExecutionHooks *hooks,
    MpsrtMpsTopKEncodeResult *result, std::string *error) const {
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
  if (stage.inputs.size() != 1 || stage.outputs.size() != 2 ||
      stage.output_descs.size() != 2) {
    return fail(error,
                "GFX MPSRT: MPS TopK requires one input and two outputs");
  }

  const auto *input_tensor = find_tensor(model, stage.inputs[0]);
  if (!input_tensor) {
    return fail(error,
                "GFX MPSRT: MPS TopK input tensor descriptor is missing");
  }

  MpsrtBoundBuffer input_buffer;
  MpsrtBoundBuffer values_buffer;
  MpsrtBoundBuffer indices_buffer;
  if (!lookup_bound_buffer(bindings, stage.inputs[0], "input", input_buffer,
                           error) ||
      !lookup_bound_buffer(bindings, stage.outputs[0], "values output",
                           values_buffer, error) ||
      !lookup_bound_buffer(bindings, stage.outputs[1], "indices output",
                           indices_buffer, error)) {
    return false;
  }

  if (prepared.uses_mps_graph_topk) {
    if (!prepared.graph_input_tensor || !prepared.graph_values_tensor ||
        !prepared.graph_indices_tensor) {
      return fail(error, "GFX MPSRT: prepared MPSGraph TopK state is incomplete");
    }
    const MPSDataType data_type = mps_data_type_from_gfx(prepared.data_type);
    if (data_type == MPSDataTypeInvalid) {
      return fail(error, "GFX MPSRT: MPSGraph TopK dtype is unsupported");
    }
    const NSUInteger input_offset =
        static_cast<NSUInteger>(input_buffer.offset +
                                input_tensor->desc.byte_offset);
    const NSUInteger values_offset =
        static_cast<NSUInteger>(values_buffer.offset +
                                stage.output_descs[0].byte_offset);
    if (input_offset != 0 || values_offset != 0) {
      return fail(error,
                  "GFX MPSRT: MPSGraph TopK requires zero-offset input/value buffers");
    }

    const bool pack_i64_indices =
        stage.output_descs[1].dtype ==
        static_cast<uint32_t>(GfxMpsrtDType::I64);
    const GfxMpsrtTensorAbiDesc graph_indices_tensor_desc =
        pack_i64_indices ? make_mps_topk_u32_index_desc(stage.output_descs[1])
                         : stage.output_descs[1];
    id<MTLBuffer> graph_indices_buffer =
        static_cast<id<MTLBuffer>>(indices_buffer.buffer);
    if (pack_i64_indices) {
      id<MTLDevice> device = context.device();
      graph_indices_buffer =
          [device newBufferWithLength:static_cast<NSUInteger>(
                                          graph_indices_tensor_desc.byte_length)
                              options:MTLResourceStorageModePrivate];
      if (!graph_indices_buffer) {
        return fail(error,
                    "GFX MPSRT: failed to allocate temporary TopK i32 indices");
      }
    } else {
      const NSUInteger indices_offset =
          static_cast<NSUInteger>(indices_buffer.offset +
                                  stage.output_descs[1].byte_offset);
      if (indices_offset != 0) {
        return fail(error,
                    "GFX MPSRT: MPSGraph TopK requires zero-offset direct index buffer");
      }
    }

    metal_end_compute_encoder(command_buffer);
    id<MTLCommandBuffer> command =
        static_cast<id<MTLCommandBuffer>>(command_buffer);
    MPSCommandBuffer *mps_command =
        [MPSCommandBuffer commandBufferWithCommandBuffer:command];
    MPSGraph *graph = static_cast<MPSGraph *>(prepared.kernel);
    NSArray<NSNumber *> *input_shape =
        mps_shape_from_tensor_desc(input_tensor->desc);
    NSArray<NSNumber *> *values_shape =
        mps_shape_from_tensor_desc(stage.output_descs[0]);
    NSArray<NSNumber *> *indices_shape =
        mps_shape_from_tensor_desc(graph_indices_tensor_desc);
    MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:static_cast<id<MTLBuffer>>(input_buffer.buffer)
                    shape:input_shape
                 dataType:data_type];
    MPSGraphTensorData *values_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:static_cast<id<MTLBuffer>>(values_buffer.buffer)
                    shape:values_shape
                 dataType:data_type];
    MPSGraphTensorData *indices_data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:graph_indices_buffer
                    shape:indices_shape
                 dataType:MPSDataTypeInt32];
    if (!mps_command || !graph || !input_data || !values_data ||
        !indices_data) {
      [input_data release];
      [values_data release];
      [indices_data release];
      if (pack_i64_indices) {
        [graph_indices_buffer release];
      }
      return fail(error, "GFX MPSRT: failed to bind MPSGraph TopK tensors");
    }

    const auto encode_start = hooks && hooks->on_segment
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
    bool encoded_executable = false;
    if (prepared.graph_executable) {
      MPSGraphExecutable *executable =
          static_cast<MPSGraphExecutable *>(prepared.graph_executable);
      NSMutableArray<MPSGraphTensorData *> *inputs = [NSMutableArray array];
      NSArray<MPSGraphTensor *> *feed_tensors = executable.feedTensors;
      if (feed_tensors.count == 0) {
        [inputs addObject:input_data];
      } else {
        for (MPSGraphTensor *feed_tensor in feed_tensors) {
          if (feed_tensor ==
              static_cast<MPSGraphTensor *>(prepared.graph_input_tensor)) {
            [inputs addObject:input_data];
          } else {
            [input_data release];
            [values_data release];
            [indices_data release];
            if (pack_i64_indices) {
              [graph_indices_buffer release];
            }
            return fail(error,
                        "GFX MPSRT: MPSGraph TopK executable feed tensor is unknown");
          }
        }
      }
      [executable encodeToCommandBuffer:mps_command
                            inputsArray:inputs
                           resultsArray:@[ values_data, indices_data ]
                    executionDescriptor:nil];
      encoded_executable = true;
    } else {
      MPSGraphTensorDataDictionary *feeds =
          [NSDictionary dictionaryWithObjectsAndKeys:
                            input_data,
                            static_cast<MPSGraphTensor *>(
                                prepared.graph_input_tensor),
                            nil];
      MPSGraphTensorDataDictionary *results =
          [NSDictionary dictionaryWithObjectsAndKeys:
                            values_data,
                            static_cast<MPSGraphTensor *>(
                                prepared.graph_values_tensor),
                            indices_data,
                            static_cast<MPSGraphTensor *>(
                                prepared.graph_indices_tensor),
                            nil];
      [graph encodeToCommandBuffer:mps_command
                             feeds:feeds
                  targetOperations:nil
                 resultsDictionary:results
               executionDescriptor:nil];
    }

    [input_data release];
    [values_data release];
    [indices_data release];

    if (pack_i64_indices) {
      const bool resolved = encode_mps_topk_stable_i64_index_resolve(
          command_buffer, context,
          static_cast<id<MTLBuffer>>(input_buffer.buffer),
          static_cast<id<MTLBuffer>>(values_buffer.buffer), graph_indices_buffer,
          static_cast<id<MTLBuffer>>(indices_buffer.buffer),
          static_cast<NSUInteger>(indices_buffer.offset +
                                  stage.output_descs[1].byte_offset),
          input_tensor->desc, stage.output_descs[0], graph_indices_tensor_desc,
          stage.output_descs[1], hooks, error);
      [graph_indices_buffer release];
      if (!resolved) {
        return false;
      }
    }

    if (result) {
      result->bound_buffers = 3;
      result->kernel_encodes = pack_i64_indices ? 2 : 1;
    }
    if (hooks && hooks->on_counter) {
      hooks->on_counter("mpsrt_mps_topk_request_encode_count", 1);
      hooks->on_counter("mpsrt_mps_topk_kernel_encode_count", 1);
      hooks->on_counter("mpsrt_mps_topk_bound_buffer_count", 3);
      hooks->on_counter("mpsrt_mps_graph_topk_request_encode_count", 1);
      hooks->on_counter("mpsrt_mps_graph_topk_kernel_encode_count", 1);
      if (encoded_executable) {
        hooks->on_counter("mpsrt_mps_graph_topk_executable_encode_count", 1);
      }
    }
    if (hooks && hooks->on_segment) {
      const auto setup_cpu_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - encode_start);
      hooks->on_segment("mpsrt_encode", stage.stage_record_key, setup_cpu_us,
                        0, 3, 0, 0, 0, 0, -1, 0,
                        reinterpret_cast<uint64_t>(command_buffer));
    }
    return true;
  }

  MPSMatrixDescriptor *input_desc = nil;
  MPSMatrixDescriptor *values_desc = nil;
  MPSMatrixDescriptor *indices_desc = nil;
  const bool pack_i64_indices =
      stage.output_descs[1].dtype == static_cast<uint32_t>(GfxMpsrtDType::I64);
  const GfxMpsrtTensorAbiDesc mps_indices_tensor_desc =
      pack_i64_indices ? make_mps_topk_u32_index_desc(stage.output_descs[1])
                       : stage.output_descs[1];
  if (!make_mps_matrix_descriptor(input_tensor->desc, input_desc, "input",
                                  error) ||
      !make_mps_matrix_descriptor(stage.output_descs[0], values_desc,
                                  "values output", error) ||
      !make_mps_topk_index_matrix_descriptor(
          mps_indices_tensor_desc, indices_desc, "indices output", error)) {
    return false;
  }

  id<MTLBuffer> mps_indices_buffer =
      static_cast<id<MTLBuffer>>(indices_buffer.buffer);
  NSUInteger mps_indices_offset =
      static_cast<NSUInteger>(indices_buffer.offset +
                              stage.output_descs[1].byte_offset);
  if (pack_i64_indices) {
    id<MTLDevice> device = context.device();
    mps_indices_buffer =
        [device newBufferWithLength:static_cast<NSUInteger>(
                                        mps_indices_tensor_desc.byte_length)
                            options:MTLResourceStorageModePrivate];
    mps_indices_offset = 0;
    if (!mps_indices_buffer) {
      return fail(error,
                  "GFX MPSRT: failed to allocate temporary TopK u32 indices");
    }
  }

  MPSMatrix *input_matrix = [[MPSMatrix alloc]
      initWithBuffer:static_cast<id<MTLBuffer>>(input_buffer.buffer)
              offset:static_cast<NSUInteger>(input_buffer.offset +
                                             input_tensor->desc.byte_offset)
          descriptor:input_desc];
  MPSMatrix *values_matrix = [[MPSMatrix alloc]
      initWithBuffer:static_cast<id<MTLBuffer>>(values_buffer.buffer)
              offset:static_cast<NSUInteger>(values_buffer.offset +
                                             stage.output_descs[0].byte_offset)
          descriptor:values_desc];
  MPSMatrix *indices_matrix = [[MPSMatrix alloc]
      initWithBuffer:mps_indices_buffer
              offset:mps_indices_offset
          descriptor:indices_desc];
  if (!input_matrix || !values_matrix || !indices_matrix) {
    [input_matrix release];
    [values_matrix release];
    [indices_matrix release];
    if (pack_i64_indices) {
      [mps_indices_buffer release];
    }
    return fail(error, "GFX MPSRT: failed to create MPS TopK matrix wrappers");
  }

  metal_end_compute_encoder(command_buffer);
  id<MTLCommandBuffer> command =
      static_cast<id<MTLCommandBuffer>>(command_buffer);
  const auto encode_start = hooks && hooks->on_segment
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};

  MPSMatrixFindTopK *kernel = static_cast<MPSMatrixFindTopK *>(prepared.kernel);
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

  if (pack_i64_indices) {
    const bool packed = encode_mps_topk_i64_pack_bridge(
        command_buffer, context, mps_indices_buffer,
        static_cast<id<MTLBuffer>>(indices_buffer.buffer),
        static_cast<NSUInteger>(indices_buffer.offset +
                                stage.output_descs[1].byte_offset),
        mps_indices_tensor_desc, stage.output_descs[1], hooks, error);
    [mps_indices_buffer release];
    if (!packed) {
      return false;
    }
  }

  if (result) {
    result->bound_buffers = 3;
    result->kernel_encodes = pack_i64_indices ? 2 : 1;
  }
  if (hooks && hooks->on_counter) {
    hooks->on_counter("mpsrt_mps_topk_request_encode_count", 1);
    hooks->on_counter("mpsrt_mps_topk_kernel_encode_count", 1);
    hooks->on_counter("mpsrt_mps_topk_bound_buffer_count", 3);
  }
  if (hooks && hooks->on_segment) {
    const auto setup_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
    hooks->on_segment("mpsrt_encode", stage.stage_record_key, setup_cpu_us, 0,
                      3, 0, 0, 0, 0, -1, 0,
                      reinterpret_cast<uint64_t>(command_buffer));
  }
  return true;
}

bool MpsrtRequest::encode_mps_sdpa(
    GpuCommandBufferHandle command_buffer, const MpsrtModel &model,
    const MpsrtRuntimeStage &stage, const MpsrtPreparedMpsSdpa &prepared,
    const MpsrtTensorBindings &bindings, const KernelExecutionHooks *hooks,
    MpsrtMpsSdpaEncodeResult *result, std::string *error) const {
  if (result) {
    *result = {};
  }
  OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
  if (stage.kind != GfxMpsrtStageKind::MPSSdpa) {
    return fail(error, "GFX MPSRT: cannot encode non-SDPA stage with MPS SDPA");
  }
  if (!prepared.kernel) {
    return fail(error, "GFX MPSRT: prepared MPS SDPA graph is null");
  }
  if (stage.inputs.size() != 3 || stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return fail(error,
                "GFX MPSRT: MPS SDPA requires Q, K, V and one output");
  }
  if (!prepared.graph_query_tensor || !prepared.graph_key_tensor ||
      !prepared.graph_value_tensor || !prepared.graph_output_tensor) {
    return fail(error, "GFX MPSRT: prepared MPSGraph SDPA state is incomplete");
  }

  const auto *query_tensor = find_tensor(model, stage.inputs[0]);
  const auto *key_tensor = find_tensor(model, stage.inputs[1]);
  const auto *value_tensor = find_tensor(model, stage.inputs[2]);
  if (!query_tensor || !key_tensor || !value_tensor) {
    return fail(error,
                "GFX MPSRT: MPS SDPA input tensor descriptor is missing");
  }

  MpsrtBoundBuffer query_buffer;
  MpsrtBoundBuffer key_buffer;
  MpsrtBoundBuffer value_buffer;
  MpsrtBoundBuffer output_buffer;
  if (!lookup_bound_buffer(bindings, stage.inputs[0], "query", query_buffer,
                           error) ||
      !lookup_bound_buffer(bindings, stage.inputs[1], "key", key_buffer,
                           error) ||
      !lookup_bound_buffer(bindings, stage.inputs[2], "value", value_buffer,
                           error) ||
      !lookup_bound_buffer(bindings, stage.outputs[0], "output", output_buffer,
                           error)) {
    return false;
  }

  const MPSDataType data_type = mps_data_type_from_gfx(prepared.data_type);
  if (data_type == MPSDataTypeInvalid) {
    return fail(error, "GFX MPSRT: MPSGraph SDPA dtype is unsupported");
  }

  const NSUInteger query_offset =
      static_cast<NSUInteger>(query_buffer.offset +
                              query_tensor->desc.byte_offset);
  const NSUInteger key_offset =
      static_cast<NSUInteger>(key_buffer.offset + key_tensor->desc.byte_offset);
  const NSUInteger value_offset =
      static_cast<NSUInteger>(value_buffer.offset +
                              value_tensor->desc.byte_offset);
  const NSUInteger output_offset =
      static_cast<NSUInteger>(output_buffer.offset +
                              stage.output_descs.front().byte_offset);
  if (query_offset != 0 || key_offset != 0 || value_offset != 0 ||
      output_offset != 0) {
    return fail(error,
                "GFX MPSRT: MPSGraph SDPA requires zero-offset dense buffers");
  }

  metal_end_compute_encoder(command_buffer);
  id<MTLCommandBuffer> command =
      static_cast<id<MTLCommandBuffer>>(command_buffer);
  MPSCommandBuffer *mps_command =
      [MPSCommandBuffer commandBufferWithCommandBuffer:command];
  MPSGraph *graph = static_cast<MPSGraph *>(prepared.kernel);
  NSArray<NSNumber *> *query_shape =
      mps_shape_from_tensor_desc(query_tensor->desc);
  NSArray<NSNumber *> *key_shape =
      mps_shape_from_tensor_desc(key_tensor->desc);
  NSArray<NSNumber *> *value_shape =
      mps_shape_from_tensor_desc(value_tensor->desc);
  NSArray<NSNumber *> *output_shape =
      mps_shape_from_tensor_desc(stage.output_descs.front());
  MPSGraphTensorData *query_data = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:static_cast<id<MTLBuffer>>(query_buffer.buffer)
                  shape:query_shape
               dataType:data_type];
  MPSGraphTensorData *key_data = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:static_cast<id<MTLBuffer>>(key_buffer.buffer)
                  shape:key_shape
               dataType:data_type];
  MPSGraphTensorData *value_data = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:static_cast<id<MTLBuffer>>(value_buffer.buffer)
                  shape:value_shape
               dataType:data_type];
  MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:static_cast<id<MTLBuffer>>(output_buffer.buffer)
                  shape:output_shape
               dataType:data_type];
  if (!mps_command || !graph || !query_data || !key_data || !value_data ||
      !output_data) {
    [query_data release];
    [key_data release];
    [value_data release];
    [output_data release];
    return fail(error, "GFX MPSRT: failed to bind MPSGraph SDPA tensors");
  }

  const auto encode_start = hooks && hooks->on_segment
                                ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
  bool encoded_executable = false;
  if (prepared.graph_executable) {
    MPSGraphExecutable *executable =
        static_cast<MPSGraphExecutable *>(prepared.graph_executable);
    NSMutableArray<MPSGraphTensorData *> *inputs = [NSMutableArray array];
    NSArray<MPSGraphTensor *> *feed_tensors = executable.feedTensors;
    if (feed_tensors.count == 0) {
      [inputs addObject:query_data];
      [inputs addObject:key_data];
      [inputs addObject:value_data];
    } else {
      for (MPSGraphTensor *feed_tensor in feed_tensors) {
        if (feed_tensor ==
            static_cast<MPSGraphTensor *>(prepared.graph_query_tensor)) {
          [inputs addObject:query_data];
        } else if (feed_tensor ==
                   static_cast<MPSGraphTensor *>(prepared.graph_key_tensor)) {
          [inputs addObject:key_data];
        } else if (feed_tensor ==
                   static_cast<MPSGraphTensor *>(prepared.graph_value_tensor)) {
          [inputs addObject:value_data];
        } else {
          [query_data release];
          [key_data release];
          [value_data release];
          [output_data release];
          return fail(error,
                      "GFX MPSRT: MPSGraph SDPA executable feed tensor is unknown");
        }
      }
    }
    [executable encodeToCommandBuffer:mps_command
                          inputsArray:inputs
                         resultsArray:@[ output_data ]
                  executionDescriptor:nil];
    encoded_executable = true;
  } else {
    MPSGraphTensorDataDictionary *feeds =
        [NSDictionary dictionaryWithObjectsAndKeys:
                          query_data,
                          static_cast<MPSGraphTensor *>(
                              prepared.graph_query_tensor),
                          key_data,
                          static_cast<MPSGraphTensor *>(
                              prepared.graph_key_tensor),
                          value_data,
                          static_cast<MPSGraphTensor *>(
                              prepared.graph_value_tensor),
                          nil];
    MPSGraphTensorDataDictionary *results =
        [NSDictionary dictionaryWithObjectsAndKeys:
                          output_data,
                          static_cast<MPSGraphTensor *>(
                              prepared.graph_output_tensor),
                          nil];
    [graph encodeToCommandBuffer:mps_command
                           feeds:feeds
                targetOperations:nil
               resultsDictionary:results
             executionDescriptor:nil];
  }

  [query_data release];
  [key_data release];
  [value_data release];
  [output_data release];

  if (result) {
    result->bound_buffers = 4;
    result->kernel_encodes = 1;
  }
  if (hooks && hooks->on_counter) {
    hooks->on_counter("mpsrt_mps_sdpa_request_encode_count", 1);
    hooks->on_counter("mpsrt_mps_sdpa_kernel_encode_count", 1);
    hooks->on_counter("mpsrt_mps_sdpa_bound_buffer_count", 4);
    hooks->on_counter("mpsrt_mps_graph_sdpa_request_encode_count", 1);
    hooks->on_counter("mpsrt_mps_graph_sdpa_kernel_encode_count", 1);
    if (encoded_executable) {
      hooks->on_counter("mpsrt_mps_graph_sdpa_executable_encode_count", 1);
    }
  }
  if (hooks && hooks->on_segment) {
    const auto setup_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_start);
    hooks->on_segment("mpsrt_encode", stage.stage_record_key, setup_cpu_us, 0,
                      4, 0, 0, 0, 0, -1, 0,
                      reinterpret_cast<uint64_t>(command_buffer));
  }
  return true;
}

bool MpsrtRequest::encode_prepared_model(
    GpuCommandBufferHandle command_buffer, const MpsrtModel &model,
    const MpsrtPreparedModel &prepared_model,
    const std::vector<KernelDispatch> &stage_dispatches,
    const MpsrtTensorBindings &bindings, const KernelExecutionHooks *hooks,
    MpsrtModelEncodeResult *result, std::string *error,
    MpsrtContext *context,
    const std::vector<MpsrtBoundBuffer> *direct_msl_runtime_buffers) const {
  if (result) {
    *result = {};
  }
  OPENVINO_ASSERT(command_buffer, "GFX MPSRT: command buffer is null");
  if (stage_dispatches.size() < model.stages.size()) {
    return fail(
        error,
        "GFX MPSRT: missing dispatch descriptors for prepared model stages");
  }

  if (hooks && hooks->on_counter) {
    hooks->on_counter("mpsrt_model_request_encode_count", 1);
  }

  std::vector<MpsrtBoundBuffer> stage_buffers;
  for (size_t stage_index = 0; stage_index < model.stages.size();
       ++stage_index) {
    const auto &stage = model.stages[stage_index];
    if (stage.kind == GfxMpsrtStageKind::MPSGemm) {
      const auto *prepared =
          find_prepared_mps_gemm(prepared_model, stage_index);
      if (!prepared) {
        return fail(error, "GFX MPSRT: missing prepared MPS GEMM for stage " +
                               std::to_string(stage_index));
      }
      MpsrtMpsGemmEncodeResult stage_result;
      if (!encode_mps_gemm(command_buffer, model, stage, *prepared, bindings,
                           hooks, &stage_result, error)) {
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
      const auto *prepared =
          find_prepared_mps_conv2d(prepared_model, stage_index);
      if (!prepared) {
        return fail(error, "GFX MPSRT: missing prepared MPS Conv2D for stage " +
                               std::to_string(stage_index));
      }
      MpsrtMpsConv2DEncodeResult stage_result;
      if (!encode_mps_conv2d(command_buffer, model, stage, *prepared, bindings,
                             hooks, &stage_result, error)) {
        return false;
      }
      if (result) {
        ++result->encoded_mps_conv2d_stages;
        result->bound_buffers += stage_result.bound_resources;
      }
      if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_model_request_mps_conv2d_stage_encode_count",
                          1);
      }
      continue;
    }

    if (stage.kind == GfxMpsrtStageKind::MPSPool2D) {
      const auto *prepared =
          find_prepared_mps_pool2d(prepared_model, stage_index);
      if (!prepared) {
        return fail(error, "GFX MPSRT: missing prepared MPS Pool2D for stage " +
                               std::to_string(stage_index));
      }
      MpsrtMpsPool2DEncodeResult stage_result;
      if (!encode_mps_pool2d(command_buffer, model, stage, *prepared, bindings,
                             hooks, &stage_result, error)) {
        return false;
      }
      if (result) {
        ++result->encoded_mps_pool2d_stages;
        result->bound_buffers += stage_result.bound_resources;
      }
      if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_model_request_mps_pool2d_stage_encode_count",
                          1);
      }
      continue;
    }

    if (stage.kind == GfxMpsrtStageKind::MPSResize2D) {
      const auto *prepared =
          find_prepared_mps_resize2d(prepared_model, stage_index);
      if (!prepared) {
        return fail(error,
                    "GFX MPSRT: missing prepared MPS Resize2D for stage " +
                        std::to_string(stage_index));
      }
      MpsrtMpsResize2DEncodeResult stage_result;
      if (!encode_mps_resize2d(command_buffer, model, stage, *prepared,
                               bindings, hooks, &stage_result, error)) {
        return false;
      }
      if (result) {
        ++result->encoded_mps_resize2d_stages;
        result->bound_buffers += stage_result.bound_resources;
      }
      if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_model_request_mps_resize2d_stage_encode_count",
                          1);
      }
      continue;
    }

    if (stage.kind == GfxMpsrtStageKind::MPSSoftmax) {
      const auto *prepared =
          find_prepared_mps_softmax(prepared_model, stage_index);
      if (!prepared) {
        return fail(error,
                    "GFX MPSRT: missing prepared MPS Softmax for stage " +
                        std::to_string(stage_index));
      }
      MpsrtMpsSoftmaxEncodeResult stage_result;
      if (!encode_mps_softmax(command_buffer, model, stage, *prepared, bindings,
                              hooks, &stage_result, error)) {
        return false;
      }
      if (result) {
        ++result->encoded_mps_softmax_stages;
        result->bound_buffers += stage_result.bound_buffers;
      }
      if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_model_request_mps_softmax_stage_encode_count",
                          1);
      }
      continue;
    }

    if (stage.kind == GfxMpsrtStageKind::MPSTopK) {
      const auto *prepared =
          find_prepared_mps_topk(prepared_model, stage_index);
      if (!prepared) {
        return fail(error, "GFX MPSRT: missing prepared MPS TopK for stage " +
                               std::to_string(stage_index));
      }
      if (!context) {
        return fail(error, "GFX MPSRT: MPS TopK i64 pack requires context");
      }
      MpsrtMpsTopKEncodeResult stage_result;
      if (!encode_mps_topk(command_buffer, *context, model, stage, *prepared,
                           bindings, hooks, &stage_result, error)) {
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

    if (stage.kind == GfxMpsrtStageKind::MPSSdpa) {
      const auto *prepared =
          find_prepared_mps_sdpa(prepared_model, stage_index);
      if (!prepared) {
        return fail(error, "GFX MPSRT: missing prepared MPS SDPA for stage " +
                               std::to_string(stage_index));
      }
      MpsrtMpsSdpaEncodeResult stage_result;
      if (!encode_mps_sdpa(command_buffer, model, stage, *prepared, bindings,
                           hooks, &stage_result, error)) {
        return false;
      }
      if (result) {
        ++result->encoded_mps_sdpa_stages;
        result->bound_buffers += stage_result.bound_buffers;
      }
      if (hooks && hooks->on_counter) {
        hooks->on_counter("mpsrt_model_request_mps_sdpa_stage_encode_count", 1);
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

    const auto *prepared =
        find_prepared_msl_dispatch(prepared_model, stage_index);
    if (!prepared) {
      return fail(error, "GFX MPSRT: missing prepared MSL dispatch for stage " +
                             std::to_string(stage_index));
    }
    if (!build_msl_stage_buffers(stage, bindings, stage_buffers,
                                 direct_msl_runtime_buffers, error)) {
      return false;
    }

    MpsrtMslEncodeResult stage_result;
    const KernelDispatch msl_dispatch =
        make_msl_stage_dispatch(stage, *prepared, stage_dispatches[stage_index]);
    if (!encode_msl_dispatch(command_buffer, *prepared,
                             msl_dispatch, stage_buffers, hooks,
                             &stage_result)) {
      return fail(error, "GFX MPSRT: failed to encode MSL stage " +
                             std::to_string(stage_index));
    }
    if (result) {
      ++result->encoded_msl_dispatches;
      result->bound_buffers +=
          stage_result.bound_buffers + stage_result.bound_textures;
    }
    if (hooks && hooks->on_counter) {
      hooks->on_counter("mpsrt_model_request_msl_stage_encode_count", 1);
    }
  }
  return true;
}

bool MpsrtRequest::encode_prepared_model_with_image_bridges(
    GpuCommandBufferHandle command_buffer, MpsrtContext &context,
    const MpsrtModel &model, const MpsrtPreparedModel &prepared_model,
    const std::vector<KernelDispatch> &stage_dispatches,
    const MpsrtTensorBindings &bindings,
    const std::vector<MpsrtImageBridgeCopy> &image_bridge_copies,
    const KernelExecutionHooks *hooks, MpsrtModelEncodeResult *result,
    std::string *error,
    const std::vector<MpsrtBoundBuffer> *direct_msl_runtime_buffers) const {
  if (!encode_mpsrt_image_bridge_copies(
          command_buffer, context, image_bridge_copies,
          GfxMpsrtStorageBridgeDirection::BufferToImage, hooks, error)) {
    return false;
  }
  if (!encode_prepared_model(command_buffer, model, prepared_model,
                             stage_dispatches, bindings, hooks, result, error,
                             &context, direct_msl_runtime_buffers)) {
    return false;
  }
  return encode_mpsrt_image_bridge_copies(
      command_buffer, context, image_bridge_copies,
      GfxMpsrtStorageBridgeDirection::ImageToBuffer, hooks, error);
}

bool MpsrtRequest::encode_prepared_model_with_binding_set(
    GpuCommandBufferHandle command_buffer, MpsrtContext &context,
    const MpsrtModel &model, const MpsrtPreparedModel &prepared_model,
    const std::vector<KernelDispatch> &stage_dispatches,
    const MpsrtRequestBindingSet &binding_set,
    const KernelExecutionHooks *hooks, MpsrtModelEncodeResult *result,
    std::string *error) const {
  return encode_prepared_model_with_image_bridges(
      command_buffer, context, model, prepared_model, stage_dispatches,
      binding_set.bindings, binding_set.image_bridge_copies, hooks, result,
      error, &binding_set.direct_msl_runtime_buffers);
}

} // namespace mpsrt
} // namespace metal
} // namespace gfx_plugin
} // namespace ov
