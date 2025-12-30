// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"

#include "openvino/gfx_plugin/profiling.hpp"
#include "runtime/gpu_buffer.hpp"

#include "backends/metal/runtime/profiling/profiler.hpp"

namespace ov {
namespace gfx_plugin {

class GfxRemoteTensor;
struct MetalInferState;
struct MetalInferStateDeleter {
    void operator()(MetalInferState* ptr) const;
};

class VulkanProfiler;
struct VulkanProfilerDeleter {
    void operator()(VulkanProfiler* ptr) const;
};

struct InferRequestState {
    std::vector<ov::Tensor> bound_inputs;
    std::vector<std::shared_ptr<GfxRemoteTensor>> bound_remote_inputs;
    std::vector<ov::Tensor> bound_output_hosts;
    std::vector<std::shared_ptr<GfxRemoteTensor>> bound_remote_outputs;

    std::vector<std::pair<std::string, ov::Tensor>> debug_tensors;
    std::vector<GpuBuffer> debug_buffers;

    std::vector<std::vector<BufferHandle>> stage_output_handles;

    std::vector<BufferHandle> vk_input_handles;
    std::vector<BufferHandle> vk_input_staging_handles;
    std::vector<std::vector<BufferHandle>> vk_stage_output_handles;
    std::vector<BufferHandle> vk_output_staging_handles;

    std::vector<ov::ProfilingInfo> last_profiling;
    GfxProfilerConfig profiler_cfg{};

    std::unique_ptr<MetalInferState, MetalInferStateDeleter> metal;
    std::unique_ptr<MetalProfiler> metal_profiler;
    std::unique_ptr<VulkanProfiler, VulkanProfilerDeleter> vulkan_profiler;
};

}  // namespace gfx_plugin
}  // namespace ov
