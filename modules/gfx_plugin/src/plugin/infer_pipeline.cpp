// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_pipeline.hpp"

#include "openvino/op/constant.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

bool append_materialized_constant_output(std::vector<InferStage>& pipeline,
                                         GpuBufferManager* buffer_manager,
                                         const OutputSource& source,
                                         const char* error_prefix) {
    auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(source.node);
    if (!constant) {
        return false;
    }
    if (!buffer_manager) {
        OPENVINO_THROW(error_prefix, ": const buffer manager is required for constant output");
    }
    if (find_pipeline_output(pipeline, source.node.get(), source.port, nullptr)) {
        return true;
    }

    const auto et = constant->get_output_element_type(source.port);
    const auto shape = constant->get_output_shape(source.port);
    const size_t bytes = constant->get_byte_size();
    std::string key = "gfx/output_const/";
    key += constant->get_friendly_name();
    key += "/";
    key += std::to_string(source.port);
    key += "/";
    key += std::to_string(bytes);

    GpuBuffer buf = buffer_manager->wrap_const(key, constant->get_data_ptr(), bytes, et);
    OPENVINO_ASSERT(buf.valid(),
                    error_prefix,
                    ": failed to materialize constant output ",
                    constant->get_friendly_name());

    InferStage stage;
    stage.node = source.node;
    auto tensor = std::make_unique<GpuTensor>();
    tensor->buf = buf;
    tensor->shape = shape;
    tensor->expected_type = et;
    tensor->prefer_private = false;
    stage.outputs.emplace_back(std::move(tensor));
    stage.output_is_model_output.push_back(true);
    pipeline.emplace_back(std::move(stage));
    return true;
}

void materialize_constant_outputs(std::vector<InferStage>& pipeline,
                                  GpuBufferManager* buffer_manager,
                                  const std::shared_ptr<const ov::Model>& runtime_model,
                                  const std::vector<ov::Output<const ov::Node>>& outputs,
                                  const std::unordered_map<const ov::Node*, size_t>& node_map,
                                  const std::unordered_map<const ov::Node*, size_t>& param_map,
                                  const char* error_prefix) {
    for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
        const auto source = resolve_output_source(outputs, runtime_model, out_idx);
        if (!source.node) {
            continue;
        }
        if (node_map.count(source.node.get()) || param_map.count(source.node.get())) {
            continue;
        }
        append_materialized_constant_output(pipeline, buffer_manager, source, error_prefix);
    }
}

size_t find_pipeline_stage_index(const std::vector<InferStage>& pipeline,
                                 const ov::Node* node,
                                 size_t port) {
    if (!node) {
        return pipeline.size();
    }
    for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
        const auto& stage = pipeline[stage_idx];
        if (!stage.node || stage.node.get() != node) {
            continue;
        }
        if (port < stage.outputs.size()) {
            return stage_idx;
        }
    }
    return pipeline.size();
}

}  // namespace

bool is_view_op(const InferStage& stage) {
    if (!stage.stage) {
        return false;
    }
    return select_tensor_layout_plan(stage.stage->type(), stage.node).view_only;
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
                         const std::shared_ptr<const ov::Model>& runtime_model,
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
        const auto source = resolve_output_source(outputs, runtime_model, out_idx);
        auto src_node = source.node;
        if (!src_node) {
            OPENVINO_THROW(error_prefix, ": remote output source node is null");
        }
        if (auto it = node_map.find(src_node.get()); it != node_map.end()) {
            size_t src_port = source.port;
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
        if (auto* tensor = find_pipeline_output(pipeline, src_node.get(), source.port, nullptr)) {
            const auto& src_tensor = remote_outputs[out_idx]->gpu_tensor();
            tensor->buf = src_tensor.buf;
            tensor->shape = src_tensor.shape;
            tensor->expected_type = src_tensor.expected_type;
            continue;
        }
        OPENVINO_THROW(error_prefix, ": failed to bind remote output ", out_idx, " (pipeline incomplete)");
    }
}

std::vector<InferStage> build_bound_pipeline(
    const std::vector<PipelineStageDesc>& descs,
    GpuBufferManager* buffer_manager,
    void* profiler,
    bool profiling_enabled,
    const std::shared_ptr<const ov::Model>& runtime_model,
    const std::vector<ov::Output<const ov::Node>>& outputs,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
    const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_inputs,
    GpuBackend expected_backend,
    const char* error_prefix) {
    auto pipeline = build_infer_pipeline(descs, buffer_manager, profiler, profiling_enabled);
    materialize_constant_outputs(pipeline,
                                 buffer_manager,
                                 runtime_model,
                                 outputs,
                                 node_map,
                                 param_map,
                                 error_prefix);
    normalize_remote_outputs(remote_outputs, expected_backend, error_prefix);
    bind_remote_outputs(outputs,
                        runtime_model,
                        node_map,
                        param_map,
                        remote_outputs,
                        remote_inputs,
                        pipeline,
                        error_prefix);
    return pipeline;
}

void prepare_reusable_execution_plan(
    PreparedInferExecutionPlan& plan,
    const std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map) {
    if (plan.stages.size() == pipeline.size()) {
        bool compatible = true;
        for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
            if (plan.stages[stage_idx].input_refs.size() != pipeline[stage_idx].inputs.size()) {
                compatible = false;
                break;
            }
        }
        if (compatible) {
            return;
        }
    }

    plan.stages.assign(pipeline.size(), {});
    for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
        const auto& stage = pipeline[stage_idx];
        auto& prepared = plan.stages[stage_idx];
        prepared.input_refs.clear();
        prepared.input_refs.reserve(stage.inputs.size());
        prepared.resolved_inputs.clear();
        prepared.resolved_inputs.reserve(stage.inputs.size());

        for (const auto& link : stage.inputs) {
            PreparedStageInputRef ref{};
            GpuTensor* resolved = nullptr;
            if (!link.node) {
                ref.kind = PreparedStageInputKind::None;
            } else if (auto pit = param_map.find(link.node.get()); pit != param_map.end()) {
                ref.kind = PreparedStageInputKind::Parameter;
                ref.index = pit->second;
            } else if (auto it = node_map.find(link.node.get()); it != node_map.end()) {
                ref.kind = PreparedStageInputKind::StageOutput;
                ref.index = it->second;
                ref.port = link.port;
                const auto& src_stage = pipeline[it->second];
                if (link.port < src_stage.outputs.size()) {
                    resolved = src_stage.outputs[link.port].get();
                }
            }
            prepared.input_refs.push_back(ref);
            prepared.resolved_inputs.push_back(resolved);
        }
    }
}

void prepare_reusable_output_plan(
    PreparedInferOutputPlan& plan,
    const std::vector<ov::Output<const ov::Node>>& public_outputs,
    const std::shared_ptr<const ov::Model>& runtime_model,
    const std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    const char* error_prefix) {
    if (plan.outputs.size() == public_outputs.size()) {
        bool compatible = true;
        for (size_t out_idx = 0; out_idx < public_outputs.size(); ++out_idx) {
            const auto source = resolve_output_source(public_outputs, runtime_model, out_idx);
            const auto& prepared = plan.outputs[out_idx];
            if (prepared.source.node != source.node || prepared.source.port != source.port) {
                compatible = false;
                break;
            }
        }
        if (compatible) {
            return;
        }
    }

    plan.outputs.assign(public_outputs.size(), {});
    for (size_t out_idx = 0; out_idx < public_outputs.size(); ++out_idx) {
        auto& prepared = plan.outputs[out_idx];
        prepared.source = resolve_output_source(public_outputs, runtime_model, out_idx);
        if (!prepared.source.node) {
            continue;
        }

        if (auto it = node_map.find(prepared.source.node.get()); it != node_map.end()) {
            prepared.kind = PreparedOutputSourceKind::StageOutput;
            prepared.index = it->second;
            prepared.port = prepared.source.port;
        } else if (auto stage_idx = find_pipeline_stage_index(pipeline,
                                                              prepared.source.node.get(),
                                                              prepared.source.port);
                   stage_idx < pipeline.size()) {
            prepared.kind = PreparedOutputSourceKind::StageOutput;
            prepared.index = stage_idx;
            prepared.port = prepared.source.port;
        } else if (auto pit = param_map.find(prepared.source.node.get()); pit != param_map.end()) {
            prepared.kind = PreparedOutputSourceKind::Parameter;
            prepared.index = pit->second;
        }

        if (prepared.kind == PreparedOutputSourceKind::StageOutput && prepared.index < pipeline.size() &&
            prepared.port < pipeline[prepared.index].outputs.size() &&
            pipeline[prepared.index].outputs[prepared.port]) {
            const auto& tensor = *pipeline[prepared.index].outputs[prepared.port];
            prepared.static_shape = resolve_output_shape(public_outputs, prepared.source, tensor, out_idx);
            prepared.static_type = resolve_output_element_type(prepared.source, tensor, error_prefix);
            continue;
        }

        if (prepared.source.node->get_output_partial_shape(prepared.source.port).is_static()) {
            prepared.static_shape = prepared.source.node->get_output_shape(prepared.source.port);
        } else if (out_idx < public_outputs.size() && public_outputs[out_idx].get_partial_shape().is_static()) {
            prepared.static_shape = public_outputs[out_idx].get_shape();
        }
        prepared.static_type = prepared.source.node->get_output_element_type(prepared.source.port);
    }
}

ov::Shape resolve_output_shape(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                               const OutputSource& source,
                               const GpuTensor& tensor,
                               size_t out_idx) {
    if (!tensor.shape.empty()) {
        return tensor.shape;
    }
    if (source.node && source.node->get_output_partial_shape(source.port).is_static()) {
        return source.node->get_output_shape(source.port);
    }
    if (out_idx < public_outputs.size() && public_outputs[out_idx].get_partial_shape().is_static()) {
        return public_outputs[out_idx].get_shape();
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
                                   const char* error_prefix) {
    OutputViewInfo info;
    info.source = resolve_output_source(public_outputs, runtime_model, out_idx);
    info.shape = resolve_output_shape(public_outputs, info.source, tensor, out_idx);
    if (tensor.shape.empty() && !info.shape.empty()) {
        tensor.shape = info.shape;
    }
    info.type = resolve_output_element_type(info.source, tensor, error_prefix);
    if (!tensor.shape.empty()) {
        info.shape = tensor.shape;
    }
    return info;
}

GpuTensor* find_pipeline_output(std::vector<InferStage>& pipeline,
                                const ov::Node* node,
                                size_t port,
                                const char* error_prefix) {
    if (!node) {
        return nullptr;
    }
    for (auto& stage : pipeline) {
        if (!stage.node || stage.node.get() != node) {
            continue;
        }
        OPENVINO_ASSERT(port < stage.outputs.size(),
                        error_prefix ? error_prefix : "GFX",
                        ": output port out of range");
        return stage.outputs[port].get();
    }
    return nullptr;
}

}  // namespace gfx_plugin
}  // namespace ov
