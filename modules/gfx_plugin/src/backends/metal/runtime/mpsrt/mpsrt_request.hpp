// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <functional>
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
};

struct MpsrtMslEncodeResult {
    bool encoder_created = false;
    bool pipeline_bound = false;
    size_t bound_buffers = 0;
};

struct MpsrtModelEncodeResult {
    size_t encoded_msl_dispatches = 0;
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
    size_t transient_buffers_allocated = 0;
    size_t const_tensors_skipped = 0;
};

using MpsrtTransientAllocator = std::function<MpsrtBoundBuffer(const MpsrtRuntimeTensor& tensor)>;

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

bool build_mpsrt_tensor_bindings(const MpsrtModel& model,
                                 const std::vector<MpsrtBoundBuffer>& input_buffers,
                                 const std::vector<MpsrtBoundBuffer>& output_buffers,
                                 const MpsrtTransientAllocator& transient_allocator,
                                 MpsrtTensorBindings& bindings,
                                 MpsrtBindingBuildResult* result = nullptr,
                                 std::string* error = nullptr);

bool build_mpsrt_external_tensor_bindings(const MpsrtModel& model,
                                          const std::vector<MpsrtBoundBuffer>& external_buffers,
                                          const MpsrtTransientAllocator& transient_allocator,
                                          MpsrtTensorBindings& bindings,
                                          MpsrtBindingBuildResult* result = nullptr,
                                          std::string* error = nullptr);

MpsrtPreparedMslDispatch make_prepared_msl_dispatch_from_pipeline(const MpsrtRuntimeStage& stage,
                                                                  size_t stage_index,
                                                                  id<MTLComputePipelineState> pipeline);

class MpsrtRequest final {
public:
    bool encode_msl_dispatch(GpuCommandBufferHandle command_buffer,
                             const MpsrtPreparedMslDispatch& prepared,
                             const KernelDispatch& dispatch,
                             const std::vector<MpsrtBoundBuffer>& buffers,
                             const KernelExecutionHooks* hooks = nullptr,
                             MpsrtMslEncodeResult* result = nullptr) const;

    bool build_msl_stage_buffers(const MpsrtRuntimeStage& stage,
                                 const MpsrtTensorBindings& bindings,
                                 std::vector<MpsrtBoundBuffer>& buffers,
                                 std::string* error = nullptr) const;

    bool encode_prepared_model(GpuCommandBufferHandle command_buffer,
                               const MpsrtModel& model,
                               const MpsrtPreparedModel& prepared_model,
                               const std::vector<KernelDispatch>& stage_dispatches,
                               const MpsrtTensorBindings& bindings,
                               const KernelExecutionHooks* hooks = nullptr,
                               MpsrtModelEncodeResult* result = nullptr,
                               std::string* error = nullptr) const;
};

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
