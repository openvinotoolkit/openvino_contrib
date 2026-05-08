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
    void* buffer = nullptr;
    size_t offset = 0;
    void* texture = nullptr;
};

struct MpsrtMslEncodeResult {
    bool encoder_created = false;
    bool pipeline_bound = false;
    size_t bound_buffers = 0;
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

struct MpsrtModelEncodeResult {
    size_t encoded_msl_dispatches = 0;
    size_t encoded_mps_gemm_stages = 0;
    size_t encoded_mps_conv2d_stages = 0;
    size_t encoded_mps_pool2d_stages = 0;
    size_t encoded_mps_resize2d_stages = 0;
    size_t encoded_mps_softmax_stages = 0;
    size_t encoded_mps_topk_stages = 0;
    size_t skipped_non_msl_stages = 0;
    size_t bound_buffers = 0;
};

struct MpsrtBoundTensor {
    GfxMpsrtValue value = 0;
    MpsrtBoundBuffer buffer{};
};

struct MpsrtBindingBuildResult {
    size_t external_inputs_bound = 0;
    size_t external_outputs_bound = 0;
    size_t external_resources_bound = 0;
    size_t model_resources_bound = 0;
    size_t transient_buffers_allocated = 0;
    size_t transient_images_allocated = 0;
};

class MpsrtTensorBindings final {
public:
    void clear();
    void bind(GfxMpsrtValue value, MpsrtBoundBuffer buffer);
    const MpsrtBoundBuffer* lookup(GfxMpsrtValue value) const;
    size_t size() const {
        return m_bindings.size();
    }

private:
    std::vector<MpsrtBoundTensor> m_bindings;
};

std::vector<MpsrtBoundBuffer> make_mpsrt_bound_buffers(const std::vector<void*>& buffers,
                                                       const std::vector<size_t>& offsets);
MpsrtBoundBuffer make_mpsrt_bound_image(void* texture);

bool build_mpsrt_tensor_bindings(const ::ov::gfx_plugin::mpsrt::MpsrtModel& model,
                                 const std::vector<MpsrtBoundBuffer>& input_buffers,
                                 const std::vector<MpsrtBoundBuffer>& output_buffers, MpsrtTensorBindings& bindings,
                                 MpsrtBindingBuildResult* result = nullptr, std::string* error = nullptr,
                                 const MpsrtPreparedModel* prepared_model = nullptr);

bool build_mpsrt_external_tensor_bindings(const ::ov::gfx_plugin::mpsrt::MpsrtModel& model,
                                          const std::vector<MpsrtBoundBuffer>& external_buffers,
                                          MpsrtTensorBindings& bindings, MpsrtBindingBuildResult* result = nullptr,
                                          std::string* error = nullptr,
                                          const MpsrtPreparedModel* prepared_model = nullptr);

MpsrtPreparedMslDispatch
make_prepared_msl_dispatch_from_pipeline(const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage& stage, size_t stage_index,
                                         id<MTLComputePipelineState> pipeline);

class MpsrtRequest final {
public:
    bool encode_msl_dispatch(GpuCommandBufferHandle command_buffer, const MpsrtPreparedMslDispatch& prepared,
                             const KernelDispatch& dispatch, const std::vector<MpsrtBoundBuffer>& buffers,
                             const KernelExecutionHooks* hooks = nullptr, MpsrtMslEncodeResult* result = nullptr) const;

    bool build_msl_stage_buffers(const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage& stage,
                                 const MpsrtTensorBindings& bindings, std::vector<MpsrtBoundBuffer>& buffers,
                                 std::string* error = nullptr) const;

    bool encode_mps_gemm(GpuCommandBufferHandle command_buffer, const ::ov::gfx_plugin::mpsrt::MpsrtModel& model,
                         const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage& stage, const MpsrtPreparedMpsGemm& prepared,
                         const MpsrtTensorBindings& bindings, const KernelExecutionHooks* hooks = nullptr,
                         MpsrtMpsGemmEncodeResult* result = nullptr, std::string* error = nullptr) const;

    bool encode_mps_conv2d(GpuCommandBufferHandle command_buffer, const ::ov::gfx_plugin::mpsrt::MpsrtModel& model,
                           const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage& stage,
                           const MpsrtPreparedMpsConv2D& prepared, const MpsrtTensorBindings& bindings,
                           const KernelExecutionHooks* hooks = nullptr, MpsrtMpsConv2DEncodeResult* result = nullptr,
                           std::string* error = nullptr) const;

    bool encode_mps_pool2d(GpuCommandBufferHandle command_buffer, const ::ov::gfx_plugin::mpsrt::MpsrtModel& model,
                           const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage& stage,
                           const MpsrtPreparedMpsPool2D& prepared, const MpsrtTensorBindings& bindings,
                           const KernelExecutionHooks* hooks = nullptr, MpsrtMpsPool2DEncodeResult* result = nullptr,
                           std::string* error = nullptr) const;

    bool encode_mps_resize2d(GpuCommandBufferHandle command_buffer, const ::ov::gfx_plugin::mpsrt::MpsrtModel& model,
                             const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage& stage,
                             const MpsrtPreparedMpsResize2D& prepared, const MpsrtTensorBindings& bindings,
                             const KernelExecutionHooks* hooks = nullptr,
                             MpsrtMpsResize2DEncodeResult* result = nullptr, std::string* error = nullptr) const;

    bool encode_mps_softmax(GpuCommandBufferHandle command_buffer, const ::ov::gfx_plugin::mpsrt::MpsrtModel& model,
                            const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage& stage,
                            const MpsrtPreparedMpsSoftmax& prepared, const MpsrtTensorBindings& bindings,
                            const KernelExecutionHooks* hooks = nullptr, MpsrtMpsSoftmaxEncodeResult* result = nullptr,
                            std::string* error = nullptr) const;

    bool encode_mps_topk(GpuCommandBufferHandle command_buffer, const ::ov::gfx_plugin::mpsrt::MpsrtModel& model,
                         const ::ov::gfx_plugin::mpsrt::MpsrtRuntimeStage& stage, const MpsrtPreparedMpsTopK& prepared,
                         const MpsrtTensorBindings& bindings, const KernelExecutionHooks* hooks = nullptr,
                         MpsrtMpsTopKEncodeResult* result = nullptr, std::string* error = nullptr) const;

    bool encode_prepared_model(GpuCommandBufferHandle command_buffer, const ::ov::gfx_plugin::mpsrt::MpsrtModel& model,
                               const MpsrtPreparedModel& prepared_model,
                               const std::vector<KernelDispatch>& stage_dispatches, const MpsrtTensorBindings& bindings,
                               const KernelExecutionHooks* hooks = nullptr, MpsrtModelEncodeResult* result = nullptr,
                               std::string* error = nullptr) const;
};

} // namespace mpsrt
} // namespace metal
} // namespace gfx_plugin
} // namespace ov
