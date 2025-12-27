// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "remote_stub.hpp"
#include "backends/metal/runtime/memory.hpp"
#include "runtime/profiling/gfx_profiler_config.hpp"

namespace ov {
namespace gfx_plugin {

class CompiledModel;
class GfxRemoteTensor;
class MetalProfiler;
class VulkanProfiler;

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
    void infer_vulkan_impl(const std::shared_ptr<const CompiledModel>& cm);
    void release_vulkan_cache();
    ov::Tensor resolve_host_input_tensor(size_t idx) {
        if (idx < m_bound_inputs.size() && m_bound_inputs[idx]) {
            return m_bound_inputs[idx];
        }
        auto impl = ov::ISyncInferRequest::get_tensor(get_inputs().at(idx));
        ov::Tensor src;
        if (!impl._ptr) {
            ov::Shape sh = get_inputs().at(idx).get_partial_shape().is_static()
                               ? get_inputs().at(idx).get_shape()
                               : ov::Shape{1};
            src = ov::Tensor{get_inputs().at(idx).get_element_type(), sh};
            ov::ISyncInferRequest::set_tensor(get_inputs().at(idx), ov::get_tensor_impl(src));
        } else {
            src = ov::make_tensor(impl);
        }
        if (!src || !src.data()) {
            ov::Shape sh = get_inputs().at(idx).get_partial_shape().is_static()
                               ? get_inputs().at(idx).get_shape()
                               : ov::Shape{1};
            src = ov::Tensor{get_inputs().at(idx).get_element_type(), sh};
            ov::ISyncInferRequest::set_tensor(get_inputs().at(idx), ov::get_tensor_impl(src));
        }
        if (idx >= m_bound_inputs.size()) {
            m_bound_inputs.resize(get_inputs().size());
        }
        m_bound_inputs[idx] = src;
        return src;
    }
    GpuTensor resolve_remote_input_tensor(size_t idx,
                                          GpuBackend expected_backend,
                                          const char* error_prefix) const {
        OPENVINO_ASSERT(idx < m_bound_remote_inputs.size() && m_bound_remote_inputs[idx],
                        error_prefix, ": remote input is not bound");
        const auto& remote = m_bound_remote_inputs[idx];
        OPENVINO_ASSERT(remote->backend() == expected_backend,
                        error_prefix, ": remote input backend mismatch");
        GpuTensor tensor = remote->gpu_tensor();
        if (tensor.shape.empty()) {
            tensor.shape = remote->get_shape();
        }
        if (tensor.expected_type == ov::element::dynamic) {
            tensor.expected_type = remote->get_element_type();
        }
        return tensor;
    }
    const ov::Tensor* get_host_output_override(size_t idx,
                                               const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const char* error_prefix) const {
        if (idx >= m_bound_output_hosts.size() || !m_bound_output_hosts[idx]) {
            return nullptr;
        }
        const auto& host = m_bound_output_hosts[idx];
        OPENVINO_ASSERT(host.get_element_type() == type,
                        error_prefix, ": output tensor type mismatch");
        OPENVINO_ASSERT(host.get_shape() == shape,
                        error_prefix, ": output tensor shape mismatch");
        OPENVINO_ASSERT(host.data(), error_prefix, ": output tensor has null data");
        return &host;
    }

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
    std::vector<BufferHandle> m_vulkan_input_handles;
    std::vector<BufferHandle> m_vulkan_input_staging_handles;
    std::vector<std::vector<BufferHandle>> m_vulkan_stage_output_handles;
    std::vector<BufferHandle> m_vulkan_output_staging_handles;
    mutable std::vector<ov::ProfilingInfo> m_last_profiling;
    std::unique_ptr<MetalProfiler> m_profiler;
    std::unique_ptr<VulkanProfiler> m_vulkan_profiler;
    GfxProfilerConfig m_profiler_cfg{};
};

}  // namespace gfx_plugin
}  // namespace ov
