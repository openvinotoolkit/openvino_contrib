// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_pipeline.hpp"

namespace ov {
namespace gfx_plugin {

bool is_view_op(const InferStage& stage) {
    if (!stage.stage) {
        return false;
    }
    const auto& type = stage.stage->type();
    return (type == "Reshape" || type == "Squeeze" || type == "Unsqueeze");
}

ov::Shape ensure_stage_output_shape(InferStage& stage, size_t out_idx) {
    if (out_idx >= stage.outputs.size()) {
        return {};
    }
    auto& out = stage.outputs[out_idx];
    if (out->shape.empty() && stage.node &&
        stage.node->get_output_partial_shape(out_idx).is_static()) {
        out->shape = stage.node->get_output_shape(out_idx);
    }
    return out->shape;
}

ov::element::Type resolve_stage_output_type(const InferStage& stage,
                                            const GpuTensor& out,
                                            size_t out_idx,
                                            const char* error_prefix) {
    if (out.expected_type != ov::element::dynamic) {
        return out.expected_type;
    }
    if (stage.node) {
        return stage.node->get_output_element_type(out_idx);
    }
    OPENVINO_THROW(error_prefix, ": stage output type is not known");
}

void normalize_remote_tensor(GfxRemoteTensor& remote,
                             GpuBackend expected_backend,
                             const char* error_prefix) {
    OPENVINO_ASSERT(remote.backend() == expected_backend,
                    error_prefix,
                    ": remote tensor backend mismatch");
    auto& tensor = remote.gpu_tensor();
    OPENVINO_ASSERT(tensor.buf.buffer, error_prefix, ": remote tensor buffer is null");
    if (tensor.shape.empty()) {
        tensor.shape = remote.get_shape();
    }
    if (tensor.expected_type == ov::element::dynamic) {
        tensor.expected_type = remote.get_element_type();
    }
    const auto elem_type =
        tensor.expected_type != ov::element::dynamic ? tensor.expected_type : tensor.buf.type;
    if (elem_type != ov::element::dynamic && !tensor.shape.empty() && tensor.buf.size > 0) {
        const size_t required = ov::shape_size(tensor.shape) * elem_type.size();
        OPENVINO_ASSERT(required <= tensor.buf.size,
                        error_prefix,
                        ": remote tensor buffer is smaller than required");
    }
}

void normalize_remote_outputs(std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                              GpuBackend expected_backend,
                              const char* error_prefix) {
    for (auto& remote : remote_outputs) {
        if (!remote) {
            continue;
        }
        normalize_remote_tensor(*remote, expected_backend, error_prefix);
    }
}

std::vector<InferStage> build_infer_pipeline(const std::vector<PipelineStageDesc>& descs,
                                             GpuBufferManager* buffer_manager,
                                             void* profiler,
                                             bool profiling_enabled) {
    std::vector<InferStage> pipeline;
    pipeline.reserve(descs.size());
    for (size_t stage_id = 0; stage_id < descs.size(); ++stage_id) {
        const auto& desc = descs[stage_id];
        InferStage stage;
        stage.node = desc.node;
        stage.stage = desc.stage->clone();
        OPENVINO_ASSERT(stage.stage, "GFX: failed to clone stage for ", desc.node->get_friendly_name());
        stage.inputs = desc.inputs;
        stage.outputs.reserve(desc.outputs.size());
        stage.output_is_model_output.reserve(desc.outputs.size());
        for (const auto& out_desc : desc.outputs) {
            auto out_tensor = std::make_unique<GpuTensor>();
            out_tensor->shape = out_desc.shape;
            out_tensor->expected_type = out_desc.type;
            stage.outputs.emplace_back(std::move(out_tensor));
            stage.output_is_model_output.push_back(out_desc.is_model_output);
        }
        if (stage.outputs.size() == 1) {
            stage.stage->set_output(stage.outputs[0].get());
        } else {
            stage.stage->set_outputs(stage.outputs);
        }
        stage.stage->init(buffer_manager);
        stage.stage->enable_profiling(profiling_enabled);
        if (profiling_enabled && profiler) {
            const std::string node_name =
                stage.node ? stage.node->get_friendly_name() : stage.stage->name();
            const std::string node_type =
                stage.node ? stage.node->get_type_name() : stage.stage->type();
            stage.stage->set_profiler(profiler,
                                      static_cast<uint32_t>(stage_id),
                                      node_name,
                                      node_type);
        }
        pipeline.emplace_back(std::move(stage));
    }
    return pipeline;
}

void bind_remote_outputs(const std::vector<ov::Output<const ov::Node>>& outputs,
                         const std::unordered_map<const ov::Node*, size_t>& node_map,
                         const std::unordered_map<const ov::Node*, size_t>& param_map,
                         const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                         const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_inputs,
                         std::vector<InferStage>& pipeline,
                         const char* error_prefix) {
    if (remote_outputs.empty()) {
        return;
    }
    for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
        if (out_idx >= remote_outputs.size() || !remote_outputs[out_idx]) {
            continue;
        }
        auto res_node = outputs[out_idx].get_node();
        auto src_node = res_node->input_value(0).get_node_shared_ptr();
        if (auto it = node_map.find(src_node.get()); it != node_map.end()) {
            size_t src_port = res_node->input_value(0).get_index();
            auto& outs = pipeline[it->second].outputs;
            OPENVINO_ASSERT(src_port < outs.size(), error_prefix, ": remote output port out of range");
            auto& dst = outs[src_port];
            const auto& src_tensor = remote_outputs[out_idx]->gpu_tensor();
            dst->buf = src_tensor.buf;
            dst->shape = src_tensor.shape;
            dst->expected_type = src_tensor.expected_type;
            continue;
        }
        if (auto pit = param_map.find(src_node.get()); pit != param_map.end()) {
            const size_t input_idx = pit->second;
            if (input_idx < remote_inputs.size() && remote_inputs[input_idx]) {
                auto in_buf = remote_inputs[input_idx]->gpu_tensor().buf.buffer;
                auto out_buf = remote_outputs[out_idx]->gpu_tensor().buf.buffer;
                OPENVINO_ASSERT(in_buf == out_buf,
                                error_prefix, ": remote output must alias remote input for passthrough outputs");
                continue;
            }
            OPENVINO_THROW(error_prefix, ": remote output cannot be bound to non-remote input passthrough");
        }
        OPENVINO_THROW(error_prefix, ": failed to bind remote output ", out_idx, " (pipeline incomplete)");
    }
}

std::vector<InferStage> build_bound_pipeline(
    const std::vector<PipelineStageDesc>& descs,
    GpuBufferManager* buffer_manager,
    void* profiler,
    bool profiling_enabled,
    const std::vector<ov::Output<const ov::Node>>& outputs,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
    const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_inputs,
    GpuBackend expected_backend,
    const char* error_prefix) {
    auto pipeline = build_infer_pipeline(descs, buffer_manager, profiler, profiling_enabled);
    normalize_remote_outputs(remote_outputs, expected_backend, error_prefix);
    bind_remote_outputs(outputs,
                        node_map,
                        param_map,
                        remote_outputs,
                        remote_inputs,
                        pipeline,
                        error_prefix);
    return pipeline;
}

ov::Shape resolve_output_shape(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                               const OutputSource& source,
                               const GpuTensor& tensor,
                               size_t out_idx,
                               bool allow_fallback_one) {
    if (!tensor.shape.empty()) {
        return tensor.shape;
    }
    if (source.node && source.node->get_output_partial_shape(source.port).is_static()) {
        return source.node->get_output_shape(source.port);
    }
    if (out_idx < public_outputs.size() && public_outputs[out_idx].get_partial_shape().is_static()) {
        return public_outputs[out_idx].get_shape();
    }
    if (allow_fallback_one) {
        return ov::Shape{1};
    }
    return {};
}

ov::element::Type resolve_output_element_type(const OutputSource& source,
                                              const GpuTensor& tensor,
                                              const char* error_prefix) {
    if (tensor.expected_type != ov::element::dynamic) {
        return tensor.expected_type;
    }
    if (tensor.buf.type != ov::element::dynamic) {
        return tensor.buf.type;
    }
    if (source.node) {
        return source.node->get_output_element_type(source.port);
    }
    OPENVINO_THROW(error_prefix, ": output element type is not known");
}

OutputSource resolve_output_source(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                                   const std::shared_ptr<const ov::Model>& runtime_model,
                                   size_t out_idx) {
    OutputSource src;
    if (out_idx >= public_outputs.size()) {
        return src;
    }
    const auto runtime_results = runtime_model ? runtime_model->get_results() : ov::ResultVector{};
    const bool use_runtime_results = runtime_results.size() == public_outputs.size();
    const ov::Node* res_node = nullptr;
    if (use_runtime_results) {
        res_node = runtime_results[out_idx].get();
    } else {
        res_node = public_outputs[out_idx].get_node();
    }
    if (!res_node) {
        return src;
    }
    const auto input = res_node->input_value(0);
    src.node = input.get_node_shared_ptr();
    src.port = input.get_index();
    return src;
}

OutputViewInfo resolve_output_view(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                                   const std::shared_ptr<const ov::Model>& runtime_model,
                                   GpuTensor& tensor,
                                   size_t out_idx,
                                   bool allow_fallback_one,
                                   const char* error_prefix) {
    OutputViewInfo info;
    info.source = resolve_output_source(public_outputs, runtime_model, out_idx);
    info.shape = resolve_output_shape(public_outputs, info.source, tensor, out_idx, allow_fallback_one);
    if (tensor.shape.empty() && !info.shape.empty()) {
        tensor.shape = info.shape;
    }
    info.type = resolve_output_element_type(info.source, tensor, error_prefix);
    if (!tensor.shape.empty()) {
        info.shape = tensor.shape;
    }
    return info;
}

}  // namespace gfx_plugin
}  // namespace ov
