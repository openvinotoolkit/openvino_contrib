// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "backends/metal/runtime/mpsrt/mpsrt_context.hpp"
#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "runtime/gpu_backend_base.hpp"

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {

struct MpsrtBoundBuffer {
  void *buffer = nullptr;
  size_t offset = 0;
  void *texture = nullptr;
};

struct MpsrtImageBridgeCopy {
  GfxMpsrtStorageBridgeDirection direction =
      GfxMpsrtStorageBridgeDirection::BufferToImage;
  GfxMpsrtValue value = 0;
  GfxMpsrtTensorAbiDesc desc{};
  MpsrtBoundBuffer buffer_binding{};
  MpsrtBoundBuffer image_binding{};
};

struct MpsrtMslEncodeResult {
  bool encoder_created = false;
  bool pipeline_bound = false;
  size_t bound_buffers = 0;
  size_t bound_textures = 0;
};

struct MpsrtMpsGemmEncodeResult {
  size_t bound_buffers = 0;
  size_t kernel_encodes = 0;
};

struct MpsrtMpsConv2DEncodeResult {
  size_t bound_resources = 0;
  size_t kernel_encodes = 0;
};

struct MpsrtMpsPool2DEncodeResult {
  size_t bound_resources = 0;
  size_t kernel_encodes = 0;
};

struct MpsrtMpsResize2DEncodeResult {
  size_t bound_resources = 0;
  size_t kernel_encodes = 0;
};

struct MpsrtMpsSoftmaxEncodeResult {
  size_t bound_buffers = 0;
  size_t kernel_encodes = 0;
};

struct MpsrtMpsTopKEncodeResult {
  size_t bound_buffers = 0;
  size_t kernel_encodes = 0;
};

struct MpsrtMpsSdpaEncodeResult {
  size_t bound_buffers = 0;
  size_t kernel_encodes = 0;
};

struct MpsrtModelEncodeResult {
  size_t encoded_msl_dispatches = 0;
  size_t encoded_mps_gemm_stages = 0;
  size_t encoded_mps_conv2d_stages = 0;
  size_t encoded_mps_pool2d_stages = 0;
  size_t encoded_mps_resize2d_stages = 0;
  size_t encoded_mps_softmax_stages = 0;
  size_t encoded_mps_topk_stages = 0;
  size_t encoded_mps_sdpa_stages = 0;
  size_t skipped_non_msl_stages = 0;
  size_t bound_buffers = 0;
};

struct MpsrtBoundTensor {
  GfxMpsrtValue value = 0;
  MpsrtBoundBuffer buffer{};
};

class MpsrtTensorBindings final {
public:
  void clear();
  void bind(GfxMpsrtValue value, MpsrtBoundBuffer buffer);
  const MpsrtBoundBuffer *lookup(GfxMpsrtValue value) const;
  size_t size() const { return m_bindings.size(); }

private:
  std::vector<MpsrtBoundTensor> m_bindings;
};

struct MpsrtRequestBindingSet {
  MpsrtTensorBindings bindings;
  std::vector<MpsrtImageBridgeCopy> image_bridge_copies;
  std::vector<MpsrtBoundBuffer> direct_msl_runtime_buffers;
};

std::vector<MpsrtBoundBuffer>
make_mpsrt_bound_buffers(const std::vector<void *> &buffers,
                         const std::vector<size_t> &offsets);
MpsrtBoundBuffer make_mpsrt_bound_image(void *texture);

bool mpsrt_model_has_msl_dispatch(
    const ::ov::gfx_plugin::mpsrt::MpsrtModel &model);

bool mpsrt_external_abi_matches_exact_binding_plan(
    const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
    const KernelBindingPlan &binding_plan);

MpsrtPreparedMslDispatch make_prepared_msl_dispatch_from_pipeline(
    const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage &stage, size_t stage_index,
    id<MTLComputePipelineState> pipeline);

class MpsrtRequest final {
public:
  bool encode_msl_dispatch(GpuCommandBufferHandle command_buffer,
                           const MpsrtPreparedMslDispatch &prepared,
                           const KernelDispatch &dispatch,
                           const std::vector<MpsrtBoundBuffer> &buffers,
                           const KernelExecutionHooks *hooks = nullptr,
                           MpsrtMslEncodeResult *result = nullptr) const;

  bool build_msl_stage_buffers(
      const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage &stage,
      const MpsrtTensorBindings &bindings,
      std::vector<MpsrtBoundBuffer> &buffers,
      const std::vector<MpsrtBoundBuffer> *direct_runtime_buffers = nullptr,
      std::string *error = nullptr) const;

  bool build_binding_set_from_external_buffers(
      const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
      const std::vector<MpsrtBoundBuffer> &external_buffers,
      const MpsrtPreparedModel *prepared_model,
      MpsrtRequestBindingSet &binding_set,
      const KernelExecutionHooks *hooks = nullptr,
      std::string *error = nullptr) const;

  bool build_binding_set_from_runtime_buffers(
      const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
      const KernelBindingPlan &binding_plan,
      const std::vector<void *> &buffer_ptrs,
      const std::vector<size_t> &offsets,
      const MpsrtPreparedModel *prepared_model,
      MpsrtRequestBindingSet &binding_set,
      const KernelExecutionHooks *hooks = nullptr,
      std::string *error = nullptr) const;

  bool encode_mps_gemm(GpuCommandBufferHandle command_buffer,
                       const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
                       const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage &stage,
                       const MpsrtPreparedMpsGemm &prepared,
                       const MpsrtTensorBindings &bindings,
                       const KernelExecutionHooks *hooks = nullptr,
                       MpsrtMpsGemmEncodeResult *result = nullptr,
                       std::string *error = nullptr) const;

  bool
  encode_mps_conv2d(GpuCommandBufferHandle command_buffer,
                    const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
                    const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage &stage,
                    const MpsrtPreparedMpsConv2D &prepared,
                    const MpsrtTensorBindings &bindings,
                    const KernelExecutionHooks *hooks = nullptr,
                    MpsrtMpsConv2DEncodeResult *result = nullptr,
                    std::string *error = nullptr) const;

  bool
  encode_mps_pool2d(GpuCommandBufferHandle command_buffer,
                    const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
                    const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage &stage,
                    const MpsrtPreparedMpsPool2D &prepared,
                    const MpsrtTensorBindings &bindings,
                    const KernelExecutionHooks *hooks = nullptr,
                    MpsrtMpsPool2DEncodeResult *result = nullptr,
                    std::string *error = nullptr) const;

  bool
  encode_mps_resize2d(GpuCommandBufferHandle command_buffer,
                      const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
                      const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage &stage,
                      const MpsrtPreparedMpsResize2D &prepared,
                      const MpsrtTensorBindings &bindings,
                      const KernelExecutionHooks *hooks = nullptr,
                      MpsrtMpsResize2DEncodeResult *result = nullptr,
                      std::string *error = nullptr) const;

  bool
  encode_mps_softmax(GpuCommandBufferHandle command_buffer,
                     const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
                     const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage &stage,
                     const MpsrtPreparedMpsSoftmax &prepared,
                     const MpsrtTensorBindings &bindings,
                     const KernelExecutionHooks *hooks = nullptr,
                     MpsrtMpsSoftmaxEncodeResult *result = nullptr,
                     std::string *error = nullptr) const;

  bool encode_mps_topk(GpuCommandBufferHandle command_buffer,
                       MpsrtContext &context,
                       const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
                       const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage &stage,
                       const MpsrtPreparedMpsTopK &prepared,
                       const MpsrtTensorBindings &bindings,
                       const KernelExecutionHooks *hooks = nullptr,
                       MpsrtMpsTopKEncodeResult *result = nullptr,
                       std::string *error = nullptr) const;

  bool encode_mps_sdpa(GpuCommandBufferHandle command_buffer,
                       const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
                       const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage &stage,
                       const MpsrtPreparedMpsSdpa &prepared,
                       const MpsrtTensorBindings &bindings,
                       const KernelExecutionHooks *hooks = nullptr,
                       MpsrtMpsSdpaEncodeResult *result = nullptr,
                       std::string *error = nullptr) const;

  bool encode_prepared_model(
      GpuCommandBufferHandle command_buffer,
      const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
      const MpsrtPreparedModel &prepared_model,
      const std::vector<KernelDispatch> &stage_dispatches,
      const MpsrtTensorBindings &bindings,
      const KernelExecutionHooks *hooks = nullptr,
      MpsrtModelEncodeResult *result = nullptr, std::string *error = nullptr,
      MpsrtContext *context = nullptr,
      const std::vector<MpsrtBoundBuffer> *direct_msl_runtime_buffers =
          nullptr) const;

  bool encode_prepared_model_with_image_bridges(
      GpuCommandBufferHandle command_buffer, MpsrtContext &context,
      const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
      const MpsrtPreparedModel &prepared_model,
      const std::vector<KernelDispatch> &stage_dispatches,
      const MpsrtTensorBindings &bindings,
      const std::vector<MpsrtImageBridgeCopy> &image_bridge_copies,
      const KernelExecutionHooks *hooks = nullptr,
      MpsrtModelEncodeResult *result = nullptr, std::string *error = nullptr,
      const std::vector<MpsrtBoundBuffer> *direct_msl_runtime_buffers =
          nullptr) const;

  bool encode_prepared_model_with_binding_set(
      GpuCommandBufferHandle command_buffer, MpsrtContext &context,
      const ::ov::gfx_plugin::mpsrt::MpsrtModel &model,
      const MpsrtPreparedModel &prepared_model,
      const std::vector<KernelDispatch> &stage_dispatches,
      const MpsrtRequestBindingSet &binding_set,
      const KernelExecutionHooks *hooks = nullptr,
      MpsrtModelEncodeResult *result = nullptr,
      std::string *error = nullptr) const;
};

} // namespace mpsrt
} // namespace metal
} // namespace gfx_plugin
} // namespace ov
