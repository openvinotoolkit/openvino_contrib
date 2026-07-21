// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "openvino/gfx_plugin/profiling.hpp"

namespace ov {
namespace gfx_plugin {

namespace compiler {
class BackendTarget;
}  // namespace compiler

class CompiledModel;
class GfxRemoteTensor;
struct GpuTensor;
struct InferStage;
struct OutputViewInfo;
struct BufferHandle;
struct GfxProfiler;
struct OutputBindingResult;
struct InferRequestState;
struct InferRequestBackendAccess;

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
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    const std::vector<std::pair<std::string, ov::Tensor>>& get_debug_tensors() const;

private:
    friend struct InferRequestBackendAccess;

    const std::shared_ptr<const CompiledModel> get_compiled_model_typed() const;
    ov::Tensor resolve_host_input_tensor(size_t idx);
    GpuTensor resolve_remote_input_tensor(size_t idx,
                                          const compiler::BackendTarget& expected_target,
                                          const char* error_prefix) const;
    const ov::Tensor* get_host_output_override(size_t idx,
                                               const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const char* error_prefix) const;
    void ensure_input_handles(size_t count, bool with_staging, const char* error_prefix);
    void ensure_output_staging_handles(size_t count, const char* error_prefix);
    void bind_inputs_for_infer(
        const compiler::BackendTarget& expected_target,
        const std::function<void(size_t, const GpuTensor&)>& remote_handler,
        const std::function<void(size_t, const ov::Tensor&)>& host_handler,
        const char* error_prefix);
    void bind_inputs_before_infer(
        const compiler::BackendTarget& expected_target,
        std::vector<GpuTensor>& input_tensors,
        const std::function<GpuTensor(size_t, const ov::Tensor&, BufferHandle*)>& host_binder,
        const std::function<void(size_t, const GpuTensor&)>& device_result_handler,
        GfxProfiler* profiler,
        bool profiling,
        bool with_staging,
        const char* error_prefix);
    void bind_outputs_for_infer(
        const std::shared_ptr<const CompiledModel>& cm,
        std::vector<InferStage>& pipeline,
        const std::function<GpuTensor*(size_t)>& output_input_lookup,
        const std::function<void(size_t, const std::shared_ptr<GfxRemoteTensor>&)>& remote_setter,
        const std::function<void(size_t, GpuTensor&, const OutputViewInfo&, const ov::Tensor*)>& device_setter,
        bool allow_missing,
        const char* error_prefix);
    void bind_outputs_after_infer(
        const std::shared_ptr<const CompiledModel>& cm,
        std::vector<InferStage>& pipeline,
        const std::function<GpuTensor*(size_t)>& output_input_lookup,
        const std::function<OutputBindingResult(size_t,
                                                GpuTensor&,
                                                const OutputViewInfo&,
                                                const ov::Tensor*,
                                                ov::Tensor*,
                                                BufferHandle*)>& device_binder,
        const std::function<void(size_t, const OutputBindingResult&)>& device_result_handler,
        GfxProfiler* profiler,
        bool profiling,
        const char* error_prefix);

    std::unique_ptr<InferRequestState> m_state;
};

}  // namespace gfx_plugin
}  // namespace ov
