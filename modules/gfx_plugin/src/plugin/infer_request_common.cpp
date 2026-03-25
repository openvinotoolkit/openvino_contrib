// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "plugin/infer_request_state.hpp"
#include "plugin/backend_state.hpp"
#include "plugin/infer_io_utils.hpp"
#include "plugin/infer_pipeline.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gfx_remote_context.hpp"

namespace ov {
namespace gfx_plugin {

InferRequest::InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    m_state = std::make_unique<InferRequestState>();
    // Allocate host tensors for inputs/outputs to satisfy AsyncInferRequest checks.
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

    auto& state = *m_state;
    state.bound_inputs.resize(get_inputs().size());
    state.bound_remote_inputs.resize(get_inputs().size());
    state.bound_output_hosts.resize(get_outputs().size());
    state.bound_remote_outputs.resize(get_outputs().size());

    if (auto cm = get_compiled_model_typed()) {
        if (auto* backend = cm->backend_state()) {
            backend->init_infer_state(state);
        }
    }
}

InferRequest::~InferRequest() {
}

void InferRequest::set_input_tensor(const ov::Tensor& tensor) {
    set_input_tensor(0, tensor);
}

void InferRequest::set_input_tensor(size_t idx, const ov::Tensor& tensor) {
    auto& state = *m_state;
    if (idx >= state.bound_inputs.size())
        state.bound_inputs.resize(get_inputs().size());

    if (idx >= state.bound_remote_inputs.size())
        state.bound_remote_inputs.resize(get_inputs().size());

    auto impl = ov::get_tensor_impl(tensor);
    if (std::dynamic_pointer_cast<ov::IRemoteTensor>(impl._ptr)) {
        auto remote = std::dynamic_pointer_cast<GfxRemoteTensor>(impl._ptr);
        OPENVINO_ASSERT(remote, "GFX: remote tensor type mismatch");
        auto cm = get_compiled_model_typed();
        OPENVINO_ASSERT(cm, "CompiledModel is null");
        const auto backend = cm->backend();
        const char* backend_name = backend_to_string(backend);
        OPENVINO_ASSERT(remote->backend() == backend,
                        "GFX: remote tensor backend mismatch (expected ", backend_name, ")");
        ov::ISyncInferRequest::set_tensor(get_inputs().at(idx), impl);
        state.bound_inputs[idx] = {};
        state.bound_remote_inputs[idx] = remote;
        return;
    }

    // Cache a host view (no CPU copy).
    state.bound_inputs[idx] = tensor;
    state.bound_remote_inputs[idx] = {};
    ov::ISyncInferRequest::set_tensor(get_inputs().at(idx), impl);
}

void InferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    auto& state = *m_state;
    auto remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
    if (remote) {
        auto gfx_remote = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
        OPENVINO_ASSERT(gfx_remote, "GFX: remote tensor type mismatch");
        auto cm = get_compiled_model_typed();
        OPENVINO_ASSERT(cm, "CompiledModel is null");
        const auto backend = cm->backend();
        const char* backend_name = backend_to_string(backend);
        OPENVINO_ASSERT(gfx_remote->backend() == backend,
                        "GFX: remote tensor backend mismatch (expected ", backend_name, ")");
        ov::ISyncInferRequest::set_tensor(port, tensor);
    } else {
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }

    auto& inputs = get_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] == port) {
            if (i >= state.bound_inputs.size())
                state.bound_inputs.resize(inputs.size());
            if (i >= state.bound_remote_inputs.size())
                state.bound_remote_inputs.resize(inputs.size());
            if (remote) {
                state.bound_inputs[i] = {};
                state.bound_remote_inputs[i] = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
            } else {
                ov::Tensor stored_view = ov::make_tensor(ov::ISyncInferRequest::get_tensor(port));
                state.bound_inputs[i] = stored_view;
                state.bound_remote_inputs[i] = {};
            }
            break;
        }
    }

    auto& outputs = get_outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i] == port) {
            if (i >= state.bound_remote_outputs.size())
                state.bound_remote_outputs.resize(outputs.size());
            if (remote) {
                state.bound_remote_outputs[i] = std::dynamic_pointer_cast<GfxRemoteTensor>(remote);
            } else {
                state.bound_remote_outputs[i] = {};
                if (i >= state.bound_output_hosts.size())
                    state.bound_output_hosts.resize(outputs.size());
                state.bound_output_hosts[i] = ov::make_tensor(tensor);
            }
            break;
        }
    }
}

ov::SoPtr<ov::ITensor> InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    auto found = find_port(port);
    if (found.found() && found.is_output()) {
        size_t idx = found.idx;
        const auto& state = *m_state;
        if (idx < state.bound_remote_outputs.size() && state.bound_remote_outputs[idx]) {
            return ov::SoPtr<ov::ITensor>{state.bound_remote_outputs[idx], nullptr};
        }
        if (auto cm = get_compiled_model_typed()) {
            if (auto* backend = cm->backend_state()) {
                auto override_tensor = backend->get_tensor_override(state, idx, get_outputs());
                if (override_tensor._ptr) {
                    return override_tensor;
                }
            }
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
    switch (cm->backend()) {
        case GpuBackend::Metal:
            infer_metal_impl(cm);
            break;
        case GpuBackend::Vulkan:
            infer_vulkan_impl(cm);
            break;
        default:
            OPENVINO_THROW("GFX: unsupported backend for infer");
    }
}

std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    return m_state ? m_state->last_profiling : std::vector<ov::ProfilingInfo>{};
}

const std::shared_ptr<const CompiledModel> InferRequest::get_compiled_model_typed() const {
    return std::static_pointer_cast<const CompiledModel>(ov::ISyncInferRequest::get_compiled_model());
}

ov::Tensor InferRequest::resolve_host_input_tensor(size_t idx) {
    auto& state = *m_state;
    if (idx < state.bound_inputs.size() && state.bound_inputs[idx]) {
        if (state.bound_inputs[idx].data()) {
            return state.bound_inputs[idx];
        }
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
    if (idx >= state.bound_inputs.size()) {
        state.bound_inputs.resize(get_inputs().size());
    }
    state.bound_inputs[idx] = src;
    return src;
}

GpuTensor InferRequest::resolve_remote_input_tensor(size_t idx,
                                                    GpuBackend expected_backend,
                                                    const char* error_prefix) const {
    const auto& state = *m_state;
    OPENVINO_ASSERT(idx < state.bound_remote_inputs.size() && state.bound_remote_inputs[idx],
                    error_prefix, ": remote input is not bound");
    const auto& remote = state.bound_remote_inputs[idx];
    normalize_remote_tensor(*remote, expected_backend, error_prefix);
    return remote->gpu_tensor();
}

const ov::Tensor* InferRequest::get_host_output_override(size_t idx,
                                                         const ov::element::Type& type,
                                                         const ov::Shape& shape,
                                                         const char* error_prefix) const {
    const auto& state = *m_state;
    if (idx >= state.bound_output_hosts.size() || !state.bound_output_hosts[idx]) {
        return nullptr;
    }
    const auto& host = state.bound_output_hosts[idx];
    OPENVINO_ASSERT(host.get_element_type() == type,
                    error_prefix, ": output tensor type mismatch");
    OPENVINO_ASSERT(host.get_shape() == shape,
                    error_prefix, ": output tensor shape mismatch");
    OPENVINO_ASSERT(host.data(), error_prefix, ": output tensor has null data");
    return &host;
}

const std::vector<std::pair<std::string, ov::Tensor>>& InferRequest::get_debug_tensors() const {
    static const std::vector<std::pair<std::string, ov::Tensor>> kEmpty;
    if (!m_state) {
        return kEmpty;
    }
    return m_state->debug_tensors;
}

void InferRequest::ensure_output_staging_handles(size_t count, const char* error_prefix) {
    OPENVINO_ASSERT(m_state && m_state->backend, error_prefix, ": infer backend state is not initialized");
    auto& handles = m_state->backend->output_staging_handles;
    if (handles.size() < count) {
        handles.resize(count);
    }
}

void InferRequest::ensure_input_handles(size_t count, bool with_staging, const char* error_prefix) {
    OPENVINO_ASSERT(m_state && m_state->backend, error_prefix, ": infer backend state is not initialized");
    auto& handles = m_state->backend->input_handles;
    if (handles.size() < count) {
        handles.resize(count);
    }
    if (!with_staging) {
        return;
    }
    auto& staging = m_state->backend->input_staging_handles;
    if (staging.size() < count) {
        staging.resize(count);
    }
}

void InferRequest::bind_inputs_for_infer(
    GpuBackend expected_backend,
    const std::function<void(size_t, const GpuTensor&)>& remote_handler,
    const std::function<void(size_t, const ov::Tensor&)>& host_handler,
    const char* error_prefix) {
    auto& state = *m_state;
    for_each_input_tensor(
        get_inputs().size(),
        state.bound_remote_inputs,
        [&](size_t idx) {
            return resolve_remote_input_tensor(idx, expected_backend, error_prefix);
        },
        [&](size_t idx) {
            return resolve_host_input_tensor(idx);
        },
        remote_handler,
        host_handler);
}

void InferRequest::bind_outputs_for_infer(
    const std::shared_ptr<const CompiledModel>& cm,
    std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    const std::function<GpuTensor*(size_t)>& output_input_lookup,
    const std::function<void(size_t, const std::shared_ptr<GfxRemoteTensor>&)>& remote_setter,
    const std::function<void(size_t, GpuTensor&, const OutputViewInfo&, const ov::Tensor*)>& device_setter,
    bool allow_missing,
    const char* error_prefix) {
    OPENVINO_ASSERT(cm, error_prefix, ": compiled model is null");
    auto& state = *m_state;
    OPENVINO_ASSERT(state.backend, error_prefix, ": infer backend state is not initialized");
    prepare_reusable_output_plan(state.backend->reusable_output_plan,
                                 get_outputs(),
                                 cm->get_runtime_model(),
                                 pipeline,
                                 node_map,
                                 param_map,
                                 error_prefix);
    prepare_reusable_host_output_plan(state.backend->reusable_host_output_plan,
                                      state.backend->reusable_output_plan,
                                      state.bound_output_hosts);
    bind_outputs_common(
        get_outputs(),
        cm->get_runtime_model(),
        node_map,
        param_map,
        pipeline,
        output_input_lookup,
        state.bound_remote_outputs,
        [&](size_t idx, const ov::element::Type& type, const ov::Shape& shape, const char* err) {
            return get_host_output_override(idx, type, shape, err);
        },
        remote_setter,
        device_setter,
        &state.backend->reusable_output_plan,
        allow_missing,
        error_prefix);
}

}  // namespace gfx_plugin
}  // namespace ov
