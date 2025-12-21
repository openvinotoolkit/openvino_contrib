// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/runtime/isync_infer_request.hpp"
#include "runtime/metal_memory.hpp"
#include "runtime/profiling/metal_profiler_config.hpp"

namespace ov {
namespace gfx_plugin {

class CompiledModel;
class GfxRemoteTensor;
class MetalProfiler;

class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model);
    ~InferRequest() override;

    void infer() override;
    void set_input_tensor(const ov::Tensor& tensor);
    void set_input_tensor(size_t idx, const ov::Tensor& tensor);
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    ov::Tensor get_output_tensor(size_t idx) const;
    ov::Tensor get_output_tensor() const { return get_output_tensor(0); }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override { return {}; }
    const std::vector<std::pair<std::string, ov::Tensor>>& get_debug_tensors() const { return m_debug_tensors; }

private:
    const std::shared_ptr<const CompiledModel> get_compiled_model_typed() const;

    std::vector<ov::Tensor> m_bound_inputs;
    std::vector<std::shared_ptr<GfxRemoteTensor>> m_bound_remote_inputs;
    std::vector<ov::Tensor> m_bound_output_hosts;
    std::vector<std::shared_ptr<GfxRemoteTensor>> m_bound_remote_outputs;
    std::vector<std::pair<std::string, ov::Tensor>> m_debug_tensors;
    std::vector<MetalBuffer> m_debug_buffers;
    mutable MetalTensorMap m_tensor_map;
    std::unique_ptr<MetalHeapPool> m_heaps;
    std::unique_ptr<MetalFreeList> m_freelist;
    std::unique_ptr<MetalStagingPool> m_staging;
    std::unique_ptr<MetalAllocator> m_allocator;
    MetalAllocatorCore* m_alloc_core = nullptr;
    MetalDeviceCaps m_caps{};
    std::vector<std::vector<BufferHandle>> m_stage_output_handles;
    mutable std::vector<ov::ProfilingInfo> m_last_profiling;
    std::unique_ptr<MetalProfiler> m_profiler;
    MetalProfilerConfig m_profiler_cfg{};
};

}  // namespace gfx_plugin
}  // namespace ov
