// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.hpp"

#include "compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "backends/metal/profiling/profiler.hpp"
#include "backends/vulkan/profiling/profiler.hpp"
#include "remote_stub.hpp"

namespace ov {
namespace gfx_plugin {

InferRequest::InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [input](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(input.get_element_type(),
                                     input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [output](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(output.get_element_type(),
                                     output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }

    m_bound_inputs.resize(get_inputs().size());
    m_bound_remote_inputs.resize(get_inputs().size());
    m_bound_output_hosts.resize(get_outputs().size());
    m_bound_remote_outputs.resize(get_outputs().size());
}

InferRequest::~InferRequest() {
    release_vulkan_cache();
}

void InferRequest::set_input_tensor(const ov::Tensor& tensor) {
    set_input_tensor(0, tensor);
}

void InferRequest::set_input_tensor(size_t idx, const ov::Tensor& tensor) {
    if (idx >= m_bound_inputs.size())
        m_bound_inputs.resize(get_inputs().size());

    if (idx >= m_bound_remote_inputs.size())
        m_bound_remote_inputs.resize(get_inputs().size());

    auto impl = ov::get_tensor_impl(tensor);
    if (std::dynamic_pointer_cast<ov::IRemoteTensor>(impl._ptr)) {
        auto remote = std::dynamic_pointer_cast<GfxRemoteTensor>(impl._ptr);
        OPENVINO_ASSERT(remote, "GFX: remote tensor type mismatch");
        OPENVINO_ASSERT(remote->backend() == GpuBackend::Vulkan,
                        "GFX: remote tensor backend mismatch (expected Vulkan)");
        ov::ISyncInferRequest::set_tensor(get_inputs().at(idx), impl);
        m_bound_inputs[idx] = {};
        m_bound_remote_inputs[idx] = remote;
        return;
    }

    m_bound_inputs[idx] = tensor;
    m_bound_remote_inputs[idx] = {};
    ov::ISyncInferRequest::set_tensor(get_inputs().at(idx), impl);
}

void InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    auto remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
    if (remote) {
        auto gfx_remote = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
        OPENVINO_ASSERT(gfx_remote, "GFX: remote tensor type mismatch");
        OPENVINO_ASSERT(gfx_remote->backend() == GpuBackend::Vulkan,
                        "GFX: remote tensor backend mismatch (expected Vulkan)");
        ov::ISyncInferRequest::set_tensor(port, tensor);
    } else {
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }

    auto& inputs = get_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] == port) {
            if (i >= m_bound_inputs.size())
                m_bound_inputs.resize(inputs.size());
            if (i >= m_bound_remote_inputs.size())
                m_bound_remote_inputs.resize(inputs.size());
            if (remote) {
                m_bound_inputs[i] = {};
                m_bound_remote_inputs[i] = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
            } else {
                m_bound_inputs[i] = ov::make_tensor(tensor);
                m_bound_remote_inputs[i] = {};
            }
            return;
        }
    }

    auto& outputs = get_outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i] == port) {
            if (i >= m_bound_output_hosts.size())
                m_bound_output_hosts.resize(outputs.size());
            if (i >= m_bound_remote_outputs.size())
                m_bound_remote_outputs.resize(outputs.size());
            if (remote) {
                m_bound_remote_outputs[i] = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
            } else {
                m_bound_output_hosts[i] = ov::make_tensor(tensor);
                m_bound_remote_outputs[i] = {};
            }
            return;
        }
    }
}

ov::SoPtr<ov::ITensor> InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    auto found = find_port(port);
    if (found.found() && found.is_output()) {
        size_t idx = found.idx;
        if (idx < m_bound_remote_outputs.size() && m_bound_remote_outputs[idx]) {
            return ov::SoPtr<ov::ITensor>{m_bound_remote_outputs[idx], nullptr};
        }
    }
    return ov::ISyncInferRequest::get_tensor(port);
}

ov::Tensor InferRequest::get_output_tensor(size_t idx) const {
    auto so = get_tensor(get_outputs().at(idx));
    return ov::make_tensor(so);
}

void InferRequest::infer() {
    auto cm = get_compiled_model_typed();
    OPENVINO_ASSERT(cm, "CompiledModel is null");
    infer_vulkan_impl(cm);
}

std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    return {};
}

const std::shared_ptr<const CompiledModel> InferRequest::get_compiled_model_typed() const {
    return std::static_pointer_cast<const CompiledModel>(ov::ISyncInferRequest::get_compiled_model());
}

}  // namespace gfx_plugin
}  // namespace ov
