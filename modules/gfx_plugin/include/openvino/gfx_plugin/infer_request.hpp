// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "openvino/gfx_plugin/profiling.hpp"

namespace ov {
namespace gfx_plugin {

class CompiledModel;
class GfxRemoteTensor;
struct GpuTensor;
struct InferStage;
struct OutputViewInfo;
struct InferRequestState;

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
    const std::vector<std::pair<std::string, ov::Tensor>>& get_debug_tensors() const;

private:
    const std::shared_ptr<const CompiledModel> get_compiled_model_typed() const;
    void infer_metal_impl(const std::shared_ptr<const CompiledModel>& cm);
    void infer_vulkan_impl(const std::shared_ptr<const CompiledModel>& cm);
    ov::Tensor resolve_host_input_tensor(size_t idx);
    GpuTensor resolve_remote_input_tensor(size_t idx,
                                          GpuBackend expected_backend,
                                          const char* error_prefix) const;
    const ov::Tensor* get_host_output_override(size_t idx,
                                               const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const char* error_prefix) const;
    void ensure_input_handles(size_t count, bool with_staging, const char* error_prefix);
    void ensure_output_staging_handles(size_t count, const char* error_prefix);
    void bind_inputs_for_infer(
        GpuBackend expected_backend,
        const std::function<void(size_t, const GpuTensor&)>& remote_handler,
        const std::function<void(size_t, const ov::Tensor&)>& host_handler,
        const char* error_prefix);
    void bind_outputs_for_infer(
        const std::shared_ptr<const CompiledModel>& cm,
        std::vector<InferStage>& pipeline,
        const std::unordered_map<const ov::Node*, size_t>& node_map,
        const std::unordered_map<const ov::Node*, size_t>& param_map,
        const std::function<GpuTensor*(size_t)>& output_input_lookup,
        const std::function<void(size_t, const std::shared_ptr<GfxRemoteTensor>&)>& remote_setter,
        const std::function<void(size_t, GpuTensor&, const OutputViewInfo&, const ov::Tensor*)>& device_setter,
        bool allow_missing,
        const char* error_prefix);

    std::unique_ptr<InferRequestState> m_state;
};

}  // namespace gfx_plugin
}  // namespace ov
