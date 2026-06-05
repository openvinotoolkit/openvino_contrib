// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include <chrono>

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "plugin/infer_request_state.hpp"
#include "plugin/infer_request_variable_state.hpp"
#include "plugin/infer_io_utils.hpp"
#include "runtime/backend_runtime.hpp"
#include "runtime/backend_runtime_provider.hpp"
#include "common/gfx_backend_utils.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "runtime/infer_pipeline.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

bool host_output_matches_public_port(const ov::Tensor& tensor,
                                     const ov::Output<const ov::Node>& port) {
    if (!tensor || !tensor.data()) {
        return false;
    }
    if (tensor.get_element_type() != port.get_element_type()) {
        return false;
    }
    if (port.get_partial_shape().is_static()) {
        return tensor.get_shape() == port.get_shape();
    }
    return tensor.get_byte_size() > 0;
}

}  // namespace

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
            initialize_variable_states(state, cm->get_runtime_model(), backend->resources());
            backend->init_infer_state(state.runtime);
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
        normalize_remote_tensor(*remote, cm->target(), "GFX");
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
        normalize_remote_tensor(*gfx_remote, cm->target(), "GFX");
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
        auto& state = *m_state;
        if (idx < state.bound_remote_outputs.size() && state.bound_remote_outputs[idx]) {
            return ov::SoPtr<ov::ITensor>{state.bound_remote_outputs[idx], nullptr};
        }
        if (idx < state.bound_output_hosts.size() && state.bound_output_hosts[idx]) {
            return ov::get_tensor_impl(state.bound_output_hosts[idx]);
        }
        auto current = ov::ISyncInferRequest::get_tensor(port);
        if (current._ptr && !std::dynamic_pointer_cast<ov::IRemoteTensor>(current._ptr)) {
            auto host_output = ov::make_tensor(current);
            if (host_output_matches_public_port(host_output, get_outputs().at(idx))) {
                if (idx >= state.bound_output_hosts.size()) {
                    state.bound_output_hosts.resize(get_outputs().size());
                }
                state.bound_output_hosts[idx] = host_output;
                return ov::get_tensor_impl(state.bound_output_hosts[idx]);
            }
            return current;
        }
        if (auto cm = get_compiled_model_typed()) {
            if (auto* backend = cm->backend_state()) {
                auto override_tensor = backend->get_tensor_override(state.runtime, idx, get_outputs());
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
    execute_backend_infer(cm->target(), *this, cm);
}

std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    return m_state ? m_state->last_profiling : std::vector<ov::ProfilingInfo>{};
}

std::vector<ov::SoPtr<ov::IVariableState>> InferRequest::query_state() const {
    return m_state ? query_variable_states(*m_state)
                   : std::vector<ov::SoPtr<ov::IVariableState>>{};
}

const std::shared_ptr<const CompiledModel> InferRequest::get_compiled_model_typed() const {
    return std::static_pointer_cast<const CompiledModel>(ov::ISyncInferRequest::get_compiled_model());
}

ov::Tensor InferRequest::resolve_host_input_tensor(size_t idx) {
    auto& state = *m_state;
    auto impl = ov::ISyncInferRequest::get_tensor(get_inputs().at(idx));
    ov::Tensor src;
    if (impl._ptr) {
        src = ov::make_tensor(impl);
    }
    if ((!src || !src.data()) &&
        idx < state.bound_inputs.size() &&
        state.bound_inputs[idx] &&
        state.bound_inputs[idx].data()) {
        src = state.bound_inputs[idx];
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
                                                    const compiler::BackendTarget& expected_target,
                                                    const char* error_prefix) const {
    const auto& state = *m_state;
    OPENVINO_ASSERT(idx < state.bound_remote_inputs.size() && state.bound_remote_inputs[idx],
                    error_prefix, ": remote input is not bound");
    const auto& remote = state.bound_remote_inputs[idx];
    normalize_remote_tensor(*remote, expected_target, error_prefix);
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
    OPENVINO_ASSERT(m_state && m_state->runtime.backend, error_prefix, ": infer backend state is not initialized");
    auto& handles = m_state->runtime.backend->output_staging_handles;
    if (handles.size() < count) {
        handles.resize(count);
    }
}

void InferRequest::ensure_input_handles(size_t count, bool with_staging, const char* error_prefix) {
    OPENVINO_ASSERT(m_state && m_state->runtime.backend, error_prefix, ": infer backend state is not initialized");
    auto& handles = m_state->runtime.backend->input_handles;
    if (handles.size() < count) {
        handles.resize(count);
    }
    if (!with_staging) {
        return;
    }
    auto& staging = m_state->runtime.backend->input_staging_handles;
    if (staging.size() < count) {
        staging.resize(count);
    }
}

void InferRequest::bind_inputs_for_infer(
    const compiler::BackendTarget& expected_target,
    const std::function<void(size_t, const GpuTensor&)>& remote_handler,
    const std::function<void(size_t, const ov::Tensor&)>& host_handler,
    const char* error_prefix) {
    auto& state = *m_state;
    for_each_input_tensor(
        get_inputs().size(),
        state.bound_remote_inputs,
        [&](size_t idx) {
            return resolve_remote_input_tensor(idx, expected_target, error_prefix);
        },
        [&](size_t idx) {
            return resolve_host_input_tensor(idx);
        },
        remote_handler,
        host_handler);
}

void InferRequest::bind_inputs_before_infer(
    const compiler::BackendTarget& expected_target,
    std::vector<GpuTensor>& input_tensors,
    const std::function<GpuTensor(size_t, const ov::Tensor&, BufferHandle*)>& host_binder,
    const std::function<void(size_t, const GpuTensor&)>& device_result_handler,
    GfxProfiler* profiler,
    bool profiling,
    bool with_staging,
    const char* error_prefix) {
    OPENVINO_ASSERT(host_binder, error_prefix, ": input host binder is not configured");
    const size_t input_count = get_inputs().size();
    if (input_tensors.size() != input_count) {
        input_tensors.assign(input_count, {});
    }

    ensure_input_handles(input_count, with_staging, error_prefix);
    auto& input_handles = m_state->runtime.backend->input_handles;
    const auto bind_inputs_start =
        profiling ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
    bind_inputs_for_infer(
        expected_target,
        [&](size_t idx, const GpuTensor& dev) {
            input_tensors[idx] = dev;
            if (device_result_handler) {
                device_result_handler(idx, input_tensors[idx]);
            }
        },
        [&](size_t idx, const ov::Tensor& host) {
            input_tensors[idx] =
                host_binder(idx, host, &input_handles[idx]);
            if (device_result_handler) {
                device_result_handler(idx, input_tensors[idx]);
            }
        },
        error_prefix);
    if (profiling && profiler) {
        profiler->record_segment(
            "upload",
            "bind_inputs_for_infer",
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - bind_inputs_start));
    }
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
    OPENVINO_ASSERT(state.runtime.backend, error_prefix, ": infer backend state is not initialized");
    prepare_reusable_output_plan(state.runtime.backend->reusable_output_plan,
                                 get_outputs(),
                                 cm->get_runtime_model(),
                                 pipeline,
                                 node_map,
                                 param_map,
                                 error_prefix);
    prepare_reusable_host_output_plan(state.runtime.backend->reusable_host_output_plan,
                                      state.runtime.backend->reusable_output_plan,
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
        &state.runtime.backend->reusable_output_plan,
        allow_missing,
        error_prefix);
}

void InferRequest::bind_outputs_after_infer(
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
    const char* error_prefix) {
    OPENVINO_ASSERT(cm, error_prefix, ": compiled model is null");
    OPENVINO_ASSERT(m_state && m_state->runtime.backend,
                    error_prefix,
                    ": infer backend state is not initialized");
    OPENVINO_ASSERT(device_binder, error_prefix, ": output device binder is not configured");

    ensure_output_staging_handles(get_outputs().size(), error_prefix);
    auto& output_handles = m_state->runtime.backend->output_staging_handles;
    auto& output_tensors = m_state->runtime.backend->output_tensors;
    output_tensors.assign(get_outputs().size(), {});

    const auto bind_outputs_start =
        profiling ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
    bind_outputs_for_infer(
        cm,
        pipeline,
        cm->node_to_stage(),
        cm->parameter_index(),
        output_input_lookup,
        [&](size_t idx, const std::shared_ptr<GfxRemoteTensor>& remote) {
            ov::ISyncInferRequest::set_tensor(
                get_outputs()[idx],
                ov::SoPtr<ov::ITensor>{remote, nullptr});
        },
        [&](size_t idx,
            GpuTensor& dev,
            const OutputViewInfo& info,
            const ov::Tensor* host_override) {
            ov::Tensor* reusable_host = nullptr;
            auto& host_plan =
                m_state->runtime.backend->reusable_host_output_plan;
            if (idx < host_plan.outputs.size()) {
                auto& prepared = host_plan.outputs[idx];
                if (prepared.host) {
                    reusable_host = &prepared.host;
                }
            }
            auto bound = device_binder(idx,
                                       dev,
                                       info,
                                       host_override,
                                       reusable_host,
                                       &output_handles[idx]);
            if (idx < output_tensors.size()) {
                output_tensors[idx] = bound.device_tensor;
            }
            if (device_result_handler) {
                device_result_handler(idx, bound);
            }
            ov::ISyncInferRequest::set_tensor(get_outputs()[idx],
                                              ov::get_tensor_impl(bound.host_tensor));
        },
        /*allow_missing=*/false,
        error_prefix);
    if (profiling && profiler) {
        profiler->record_segment(
            "download",
            "bind_outputs_for_infer",
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - bind_outputs_start));
    }
}

}  // namespace gfx_plugin
}  // namespace ov
