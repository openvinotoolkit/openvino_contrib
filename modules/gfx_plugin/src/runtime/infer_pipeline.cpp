// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/infer_pipeline.hpp"

#include "runtime/gfx_stage_runtime_values.hpp"
#include "openvino/op/constant.hpp"

#include <limits>
#include <string_view>
#include <utility>

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
    stage.output_sources.push_back({source.node, source.port});
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

void prepare_stage_runtime_executable(InferStage& stage,
                                      size_t stage_index,
                                      GpuBufferManager* buffer_manager,
                                      const std::shared_ptr<RuntimeSession>& runtime_session) {
    if (!stage.stage) {
        return;
    }
    OPENVINO_ASSERT(runtime_session,
                    "GFX: runtime session is required to prepare stage ",
                    stage_index);
    std::vector<GpuTensor*> placeholder_inputs(stage.inputs.size(), nullptr);
    std::vector<GpuTensor*> outputs;
    outputs.reserve(stage.outputs.size());
    for (auto& output : stage.outputs) {
        outputs.push_back(output.get());
    }

    stage.runtime_session = runtime_session;
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    OPENVINO_ASSERT(descriptor,
                    "GFX: compiler-owned runtime stage descriptor is required "
                    "to prepare pipeline stage ",
                    stage_index);
    auto bindings =
        ResourceBindingTable::for_stage(placeholder_inputs, outputs, *descriptor);
    PreparedKernelExecutable prepared(*descriptor);
    prepared.prepare(*stage.stage, buffer_manager, std::move(bindings));
    stage.prepared_executable =
        std::make_unique<PreparedKernelExecutable>(std::move(prepared));
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

struct StageOutputAllocationPlan {
    bool needs_buffer = false;
    bool workspace_managed = false;
    GpuBufferDesc desc{};
};

struct ActiveStageOutputSlot {
    size_t slot = StageOutputBufferWorkspace::npos;
    size_t last_use = 0;
};

void assign_runtime_shapes_for_stage(InferStage& stage,
                                     const std::vector<GpuTensor*>& inputs,
                                     const char* error_prefix) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    const std::string_view runtime_shape_rule =
        descriptor ? std::string_view(descriptor->runtime_shape_rule)
                   : std::string_view("static_or_descriptor");
    if (!stage.node || runtime_shape_rule == "static_or_descriptor") {
        for (size_t out_idx = 0; out_idx < stage.outputs.size(); ++out_idx) {
            ensure_stage_output_shape(stage, out_idx);
        }
        return;
    }

    std::vector<GpuTensor*> outputs;
    outputs.reserve(stage.outputs.size());
    for (auto& out : stage.outputs) {
        outputs.push_back(out.get());
    }

    RuntimeInputResolver runtime_inputs;
    runtime_inputs.inputs = &inputs;
    runtime_inputs.descriptor = descriptor;
    runtime_inputs.node = stage.node;
    const auto stage_name = stage.node->get_friendly_name();

    if (runtime_shape_rule == "concat") {
        const auto plan = plan_concat_runtime_values(runtime_inputs, *stage.node, stage_name);
        assign_runtime_value_outputs(plan.values, outputs);
        return;
    }

    if (runtime_shape_rule == "broadcast") {
        const ov::Shape in_shape = runtime_inputs.shape(0);
        OPENVINO_ASSERT(!in_shape.empty(),
                        error_prefix,
                        ": Broadcast input shape is unknown for stage ",
                        stage_name);
        const auto plan = plan_broadcast_runtime_values(runtime_inputs, *stage.node, in_shape, stage_name);
        assign_runtime_value_outputs(plan, outputs);
        return;
    }

    if (runtime_shape_rule == "select") {
        const auto plan = plan_select_runtime_values(runtime_inputs, *stage.node, stage_name);
        OPENVINO_ASSERT(plan.valid(),
                        error_prefix,
                        ": Select runtime shapes are unknown for stage ",
                        stage_name);
        assign_runtime_value_outputs(plan.values, outputs);
        return;
    }

    if (runtime_shape_rule == "shape_of") {
        const auto plan = plan_shapeof_runtime_values(runtime_inputs, stage.node.get(), stage_name);
        assign_runtime_value_outputs(plan, outputs);
        return;
    }

    if (runtime_shape_rule == "slice") {
        const bool requires_runtime_shape_args =
            descriptor && descriptor->requires_runtime_shape_args;
        const auto plan = plan_slice_runtime_values(runtime_inputs,
                                                    outputs,
                                                    requires_runtime_shape_args,
                                                    stage_name);
        assign_runtime_value_outputs(plan.values, outputs);
        return;
    }

    if (runtime_shape_rule == "range") {
        const auto plan = plan_range_runtime_values(runtime_inputs, stage.node.get(), stage_name);
        assign_runtime_value_outputs(plan, outputs);
        return;
    }

    if (runtime_shape_rule == "tile") {
        const auto plan = plan_tile_runtime_values(runtime_inputs, outputs, stage_name);
        OPENVINO_ASSERT(plan.valid(),
                        error_prefix,
                        ": Tile runtime shape is unknown for stage ",
                        stage_name);
        assign_runtime_value_outputs(plan.values, outputs);
        return;
    }

    OPENVINO_THROW(error_prefix,
                   ": unsupported compiler-owned runtime shape rule '",
                   std::string(runtime_shape_rule),
                   "' for stage ",
                   stage_name);
}

}  // namespace

bool is_view_op(const InferStage& stage) {
    if (const auto* descriptor = runtime_stage_descriptor_or_null(stage)) {
        return descriptor->tensor_view_only;
    }
    return false;
}

ov::Shape ensure_stage_output_shape(InferStage& stage, size_t out_idx) {
    if (out_idx >= stage.outputs.size()) {
        return {};
    }
    auto& out = stage.outputs[out_idx];
    if (out->shape.empty() && stage.node &&
        out_idx < stage.node->get_output_size() &&
        stage.node->get_output_partial_shape(out_idx).is_static()) {
        out->shape = stage.node->get_output_shape(out_idx);
    }
    if (out->shape.empty() && out_idx < stage.output_sources.size()) {
        const auto& source = stage.output_sources[out_idx];
        if (source.node && source.port < source.node->get_output_size() &&
            source.node->get_output_partial_shape(source.port).is_static()) {
            out->shape = source.node->get_output_shape(source.port);
        }
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
    if (stage.node && out_idx < stage.node->get_output_size()) {
        return stage.node->get_output_element_type(out_idx);
    }
    if (out_idx < stage.output_sources.size()) {
        const auto& source = stage.output_sources[out_idx];
        if (source.node && source.port < source.node->get_output_size()) {
            return source.node->get_output_element_type(source.port);
        }
    }
    OPENVINO_THROW(error_prefix, ": stage output type is not known");
}

void normalize_remote_tensor(GfxRemoteTensor& remote,
                             const compiler::BackendTarget& expected_target,
                             const char* error_prefix) {
    OPENVINO_ASSERT(remote.backend() == expected_target.backend(),
                    error_prefix,
                    ": remote tensor backend mismatch");
    OPENVINO_ASSERT(remote.target().is_compatible_with_fingerprint(expected_target.fingerprint()),
                    error_prefix,
                    ": remote tensor target mismatch");
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
                              const compiler::BackendTarget& expected_target,
                              const char* error_prefix) {
    for (auto& remote : remote_outputs) {
        if (!remote) {
            continue;
        }
        normalize_remote_tensor(*remote, expected_target, error_prefix);
    }
}

std::vector<InferStage> build_infer_pipeline(const std::vector<PipelineStageDesc>& descs,
                                             GpuBufferManager* buffer_manager,
                                             void* profiler,
                                             bool profiling_enabled,
                                             std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor) {
    std::vector<InferStage> pipeline;
    pipeline.reserve(descs.size());
    OPENVINO_ASSERT(runtime_descriptor,
                    "GFX: compiler-owned runtime executable descriptor is "
                    "required to build infer pipeline");
    auto runtime_session = std::make_shared<RuntimeSession>(std::move(runtime_descriptor));
    for (size_t stage_id = 0; stage_id < descs.size(); ++stage_id) {
        const auto& desc = descs[stage_id];
        OPENVINO_ASSERT(
            desc.runtime_stage_index != PipelineStageDesc::npos,
            "GFX: compiler-owned runtime stage index is required for pipeline "
            "stage ",
            stage_id);
        const size_t runtime_stage_index = desc.runtime_stage_index;
        OPENVINO_ASSERT(runtime_stage_index < runtime_session->stage_count(),
                        "GFX: runtime executable descriptor stage index ",
                        runtime_stage_index,
                        " is out of range for pipeline stage ",
                        stage_id);
        InferStage stage;
        stage.node = desc.node;
        stage.stage = desc.stage->clone();
        OPENVINO_ASSERT(stage.stage, "GFX: failed to clone stage for ", desc.node->get_friendly_name());
        stage.runtime_stage_index = runtime_stage_index;
        stage.runtime_stage_descriptor =
            desc.runtime_descriptor
                ? desc.runtime_descriptor
                : std::make_shared<RuntimeStageExecutableDescriptor>(
                      runtime_session->stage_descriptor(runtime_stage_index));
        stage.inputs = desc.inputs;
        stage.output_aliases = desc.output_aliases;
        stage.output_lifetimes = desc.output_lifetimes;
        stage.outputs.reserve(desc.outputs.size());
        stage.output_is_model_output.reserve(desc.outputs.size());
        stage.output_sources.reserve(desc.outputs.size());
        stage.direct_stateful_assign_variable_ids.reserve(desc.outputs.size());
        for (const auto& out_desc : desc.outputs) {
            auto out_tensor = std::make_unique<GpuTensor>();
            out_tensor->shape = out_desc.shape;
            out_tensor->expected_type = out_desc.type;
            stage.outputs.emplace_back(std::move(out_tensor));
            stage.output_is_model_output.push_back(out_desc.is_model_output);
            stage.output_sources.push_back({out_desc.source_node ? out_desc.source_node : desc.node,
                                            out_desc.source_port});
            stage.direct_stateful_assign_variable_ids.push_back(
                out_desc.direct_stateful_assign_variable_id);
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
        prepare_stage_runtime_executable(stage,
                                         runtime_stage_index,
                                         buffer_manager,
                                         runtime_session);
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
    const compiler::BackendTarget& expected_target,
    std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor,
    const char* error_prefix) {
    auto pipeline = build_infer_pipeline(descs,
                                         buffer_manager,
                                         profiler,
                                         profiling_enabled,
                                         std::move(runtime_descriptor));
    materialize_constant_outputs(pipeline,
                                 buffer_manager,
                                 runtime_model,
                                 outputs,
                                 node_map,
                                 param_map,
                                 error_prefix);
    normalize_remote_outputs(remote_outputs, expected_target, error_prefix);
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
            } else if (size_t producer_idx = 0, producer_output = 0;
                       find_pipeline_output_ref(pipeline, link.node.get(), link.port, producer_idx, producer_output)) {
                ref.kind = PreparedStageInputKind::StageOutput;
                ref.index = producer_idx;
                ref.port = producer_output;
                resolved = pipeline[producer_idx].outputs[producer_output].get();
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

void assign_runtime_stage_output_shapes(
    std::vector<InferStage>& pipeline,
    PreparedInferExecutionPlan& plan,
    const std::vector<GpuTensor>& input_tensors,
    const char* error_prefix) {
    OPENVINO_ASSERT(plan.stages.size() == pipeline.size(),
                    error_prefix,
                    ": prepared execution plan does not match pipeline");

    for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
        auto& stage = pipeline[stage_idx];
        auto& prepared = plan.stages[stage_idx];
        if (prepared.resolved_inputs.size() != prepared.input_refs.size()) {
            prepared.resolved_inputs.resize(prepared.input_refs.size(), nullptr);
        }

        for (size_t input_idx = 0; input_idx < prepared.input_refs.size(); ++input_idx) {
            const auto& ref = prepared.input_refs[input_idx];
            auto* resolved = prepared.resolved_inputs[input_idx];
            switch (ref.kind) {
            case PreparedStageInputKind::Parameter:
                resolved = ref.index < input_tensors.size()
                               ? const_cast<GpuTensor*>(&input_tensors[ref.index])
                               : nullptr;
                break;
            case PreparedStageInputKind::StageOutput:
                if (ref.index < pipeline.size() &&
                    ref.port < pipeline[ref.index].outputs.size()) {
                    resolved = pipeline[ref.index].outputs[ref.port].get();
                } else {
                    resolved = nullptr;
                }
                break;
            case PreparedStageInputKind::None:
            default:
                resolved = nullptr;
                break;
            }
            prepared.resolved_inputs[input_idx] = resolved;
        }

        assign_runtime_shapes_for_stage(stage, prepared.resolved_inputs, error_prefix);
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
            bool matched_alias = false;
            for (const auto& alias : stage.output_aliases) {
                if (alias.node.get() == node && alias.source_port == port) {
                    OPENVINO_ASSERT(alias.output_port < stage.outputs.size(),
                                    error_prefix ? error_prefix : "GFX",
                                    ": output alias port out of range");
                    return stage.outputs[alias.output_port].get();
                }
            }
            if (!matched_alias) {
                continue;
            }
        }
        OPENVINO_ASSERT(port < stage.outputs.size(),
                        error_prefix ? error_prefix : "GFX",
                        ": output port out of range");
        return stage.outputs[port].get();
    }
    return nullptr;
}

void allocate_stage_outputs(std::vector<InferStage>& pipeline,
                            std::vector<std::vector<BufferHandle>>& handles,
                            GpuBufferPool& pool,
                            const StageOutputDescInitializer& describe_output,
                            StageOutputBufferWorkspace* workspace,
                            const char* error_prefix) {
    if (handles.size() != pipeline.size()) {
        handles.assign(pipeline.size(), {});
    }

    if (workspace) {
        std::vector<std::vector<StageOutputAllocationPlan>> output_plan(pipeline.size());
        std::vector<std::vector<size_t>> last_use(pipeline.size());
        workspace->output_slots.assign(pipeline.size(), {});
        workspace->last_workspace_outputs = 0;
        workspace->last_direct_outputs = 0;
        workspace->last_slots_used = 0;
        workspace->last_peak_live_slots = 0;
        size_t max_new_workspace_slots = 0;
        for (const auto& stage : pipeline) {
            max_new_workspace_slots += stage.outputs.size();
        }
        workspace->handles.reserve(workspace->handles.size() + max_new_workspace_slots);

        std::unordered_map<NodePortKey, StageOutputRef, NodePortKeyHash> output_by_source;
        output_by_source.reserve(pipeline.size());

        for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
            auto& stage = pipeline[stage_idx];
            auto register_output_source =
                [&](const std::shared_ptr<const ov::Node>& source_node,
                    size_t source_port,
                    size_t output_port) {
                    if (!source_node || output_port >= stage.outputs.size()) {
                        return;
                    }
                    output_by_source[{source_node.get(), source_port}] =
                        {stage_idx, output_port};
                };
            if (stage.node) {
                for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
                    register_output_source(stage.node, oi, oi);
                }
            }
            for (size_t oi = 0; oi < stage.output_sources.size(); ++oi) {
                const auto& source = stage.output_sources[oi];
                register_output_source(source.node, source.port, oi);
            }
            for (const auto& alias : stage.output_aliases) {
                register_output_source(alias.node,
                                       alias.source_port,
                                       alias.output_port);
            }
            auto& stage_handles = handles[stage_idx];
            if (stage_handles.size() < stage.outputs.size()) {
                stage_handles.resize(stage.outputs.size());
            }
            output_plan[stage_idx].resize(stage.outputs.size());
            last_use[stage_idx].assign(stage.outputs.size(), stage_idx);
            workspace->output_slots[stage_idx].assign(
                stage.outputs.size(), StageOutputBufferWorkspace::npos);

            for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
                auto& out_ref = stage.outputs[oi];
                if (!out_ref || out_ref->buf.valid()) {
                    continue;
                }
                GpuBufferDesc desc{};
                if (!describe_output(stage, oi, *out_ref, desc, error_prefix)) {
                    continue;
                }
                auto& plan = output_plan[stage_idx][oi];
                plan.needs_buffer = true;
                plan.desc = desc;
                plan.workspace_managed = desc.usage == BufferUsage::Intermediate &&
                                         !desc.cpu_read &&
                                         !desc.cpu_write &&
                                         !stage_output_has_multiple_graph_consumers(stage, oi);
                if (!plan.workspace_managed) {
                    continue;
                }
                if (stage_handles[oi].valid()) {
                    pool.release(stage_handles[oi]);
                }
            }
        }

        for (size_t consumer_idx = 0; consumer_idx < pipeline.size(); ++consumer_idx) {
            const auto& consumer = pipeline[consumer_idx];
            for (const auto& link : consumer.inputs) {
                if (!link.node) {
                    continue;
                }
                auto it = output_by_source.find({link.node.get(), link.port});
                if (it == output_by_source.end()) {
                    continue;
                }
                const size_t producer_idx = it->second.stage;
                const size_t producer_output = it->second.output;
                if (producer_output >= last_use[producer_idx].size()) {
                    continue;
                }
                last_use[producer_idx][producer_output] =
                    std::max(last_use[producer_idx][producer_output],
                             consumer_idx);
            }
        }

        for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
            const auto& stage = pipeline[stage_idx];
            for (size_t oi = 0;
                 oi < stage.output_is_model_output.size() &&
                 oi < last_use[stage_idx].size();
                 ++oi) {
                if (stage.output_is_model_output[oi]) {
                    last_use[stage_idx][oi] = pipeline.size();
                }
            }
            for (size_t oi = 0; oi < last_use[stage_idx].size(); ++oi) {
                if (const auto* region =
                        runtime_output_memory_region_or_null(stage, oi)) {
                    last_use[stage_idx][oi] =
                        std::max(last_use[stage_idx][oi], region->last_stage);
                }
            }
        }

        for (size_t rev = pipeline.size(); rev > 0; --rev) {
            const size_t stage_idx = rev - 1;
            const auto& stage = pipeline[stage_idx];
            if (!stage_outputs_may_alias_inputs(stage) ||
                last_use[stage_idx].empty()) {
                continue;
            }
            size_t view_last_use = stage_idx;
            for (const auto use : last_use[stage_idx]) {
                view_last_use = std::max(view_last_use, use);
            }
            for (const auto& link : stage.inputs) {
                if (!link.node) {
                    continue;
                }
                auto it = output_by_source.find({link.node.get(), link.port});
                if (it == output_by_source.end()) {
                    continue;
                }
                const size_t producer_idx = it->second.stage;
                const size_t producer_output = it->second.output;
                if (producer_output < last_use[producer_idx].size()) {
                    last_use[producer_idx][producer_output] =
                        std::max(last_use[producer_idx][producer_output],
                                 view_last_use);
                }
            }
        }

        std::vector<size_t> free_slots;
        free_slots.reserve(workspace->handles.size());
        for (size_t slot = 0; slot < workspace->handles.size(); ++slot) {
            free_slots.push_back(slot);
        }
        std::vector<ActiveStageOutputSlot> active_slots;

        auto host_visible = [](const GpuBufferDesc& desc) {
            return desc.cpu_read || desc.cpu_write || !desc.prefer_device_local;
        };
        auto compatible = [&](const BufferHandle& handle,
                              const GpuBufferDesc& desc) {
            return handle.valid() &&
                   handle.capacity >= desc.bytes &&
                   handle.buf.type == desc.type &&
                   handle.buf.host_visible == host_visible(desc);
        };
        auto slot_is_live = [&](size_t slot) {
            return std::any_of(active_slots.begin(),
                               active_slots.end(),
                               [&](const ActiveStageOutputSlot& active) {
                                   return active.slot == slot;
                               });
        };
        auto remove_live_free_slots = [&]() {
            free_slots.erase(std::remove_if(free_slots.begin(),
                                            free_slots.end(),
                                            [&](size_t slot) {
                                                return slot_is_live(slot);
                                            }),
                             free_slots.end());
        };
        auto acquire_slot = [&](const GpuBufferDesc& desc) {
            remove_live_free_slots();
            if (desc.bytes == 0) {
                if (!free_slots.empty()) {
                    const size_t slot = free_slots.back();
                    free_slots.pop_back();
                    return slot;
                }
                workspace->handles.emplace_back();
                return workspace->handles.size() - 1;
            }
            size_t best_pos = StageOutputBufferWorkspace::npos;
            size_t best_capacity = std::numeric_limits<size_t>::max();
            for (size_t pos = 0; pos < free_slots.size(); ++pos) {
                const size_t slot = free_slots[pos];
                if (slot_is_live(slot)) {
                    continue;
                }
                const auto& handle = workspace->handles[slot];
                if (!compatible(handle, desc)) {
                    continue;
                }
                if (handle.capacity < best_capacity) {
                    best_capacity = handle.capacity;
                    best_pos = pos;
                }
            }
            if (best_pos == StageOutputBufferWorkspace::npos) {
                for (size_t pos = 0; pos < free_slots.size(); ++pos) {
                    const size_t slot = free_slots[pos];
                    if (slot_is_live(slot)) {
                        continue;
                    }
                    if (!workspace->handles[slot].valid()) {
                        best_pos = pos;
                        break;
                    }
                }
            }
            if (best_pos != StageOutputBufferWorkspace::npos) {
                const size_t slot = free_slots[best_pos];
                free_slots.erase(free_slots.begin() +
                                 static_cast<std::ptrdiff_t>(best_pos));
                return slot;
            }
            workspace->handles.emplace_back();
            return workspace->handles.size() - 1;
        };
        auto update_peak_live = [&](size_t extra_internal_live = 0) {
            workspace->last_peak_live_slots =
                std::max(workspace->last_peak_live_slots,
                         active_slots.size() + extra_internal_live);
        };
        auto protect_current_stage_input_slots = [&](const InferStage& stage) {
            std::vector<size_t> protected_slots;
            for (const auto& link : stage.inputs) {
                if (!link.node) {
                    continue;
                }
                auto it = output_by_source.find({link.node.get(), link.port});
                if (it == output_by_source.end()) {
                    continue;
                }
                const size_t producer_idx = it->second.stage;
                const size_t producer_output = it->second.output;
                if (producer_idx >= workspace->output_slots.size() ||
                    producer_output >=
                        workspace->output_slots[producer_idx].size()) {
                    continue;
                }
                const size_t slot =
                    workspace->output_slots[producer_idx][producer_output];
                if (slot == StageOutputBufferWorkspace::npos) {
                    continue;
                }
                auto free_it =
                    std::find(free_slots.begin(), free_slots.end(), slot);
                if (free_it == free_slots.end()) {
                    continue;
                }
                protected_slots.push_back(slot);
                free_slots.erase(free_it);
            }
            return protected_slots;
        };

        for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
            auto active_it = active_slots.begin();
            while (active_it != active_slots.end()) {
                if (active_it->last_use < stage_idx) {
                    free_slots.push_back(active_it->slot);
                    active_it = active_slots.erase(active_it);
                } else {
                    ++active_it;
                }
            }

            auto& stage = pipeline[stage_idx];
            auto protected_input_slots = protect_current_stage_input_slots(stage);
            auto& stage_handles = handles[stage_idx];
            const bool has_internal_lifetimes =
                stage.output_lifetimes.size() >= stage.outputs.size();
            if (has_internal_lifetimes) {
                const auto& internal_lifetimes = stage.output_lifetimes;
                struct InternalOutput {
                    size_t output = 0;
                    size_t produced_at = 0;
                    size_t last_used_at = 0;
                    bool escapes_stage = false;
                    size_t escape_last_use = 0;
                };
                std::vector<bool> escapes_stage(stage.outputs.size(), false);
                std::vector<size_t> escape_last_use(stage.outputs.size(), stage_idx);
                for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
                    if (last_use[stage_idx][oi] > stage_idx) {
                        escapes_stage[oi] = true;
                        escape_last_use[oi] = last_use[stage_idx][oi];
                    }
                }
                bool alias_escape_changed = true;
                while (alias_escape_changed) {
                    alias_escape_changed = false;
                    for (size_t oi = 0;
                         oi < internal_lifetimes.size() &&
                         oi < escapes_stage.size();
                         ++oi) {
                        if (!escapes_stage[oi]) {
                            continue;
                        }
                        const size_t storage_source =
                            internal_lifetimes[oi].storage_source_output;
                        if (storage_source == PipelineStageDesc::OutputLifetime::npos ||
                            storage_source >= escapes_stage.size()) {
                            continue;
                        }
                        const size_t propagated_last_use =
                            std::max(escape_last_use[storage_source],
                                     escape_last_use[oi]);
                        if (!escapes_stage[storage_source] ||
                            escape_last_use[storage_source] != propagated_last_use) {
                            escapes_stage[storage_source] = true;
                            escape_last_use[storage_source] = propagated_last_use;
                            alias_escape_changed = true;
                        }
                    }
                }
                std::vector<InternalOutput> internal_outputs;
                internal_outputs.reserve(stage.outputs.size());
                for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
                    auto& out_ref = stage.outputs[oi];
                    if (!out_ref || out_ref->buf.valid()) {
                        continue;
                    }
                    const auto& plan = output_plan[stage_idx][oi];
                    if (!plan.needs_buffer) {
                        continue;
                    }
                    if (!plan.workspace_managed) {
                        ++workspace->last_direct_outputs;
                        out_ref->buf = pool.ensure(stage_handles[oi], plan.desc);
                        continue;
                    }
                    const auto& lifetime = internal_lifetimes[oi];
                    if (!lifetime.valid() || !lifetime.requires_buffer) {
                        continue;
                    }
                    internal_outputs.push_back({oi,
                                                lifetime.produced_at,
                                                lifetime.last_used_at,
                                                escapes_stage[oi],
                                                escape_last_use[oi]});
                }
                std::sort(internal_outputs.begin(),
                          internal_outputs.end(),
                          [](const InternalOutput& lhs,
                             const InternalOutput& rhs) {
                              if (lhs.produced_at != rhs.produced_at) {
                                  return lhs.produced_at < rhs.produced_at;
                              }
                              return lhs.output < rhs.output;
                          });

                std::vector<ActiveStageOutputSlot> internal_active_slots;
                for (const auto& output : internal_outputs) {
                    auto internal_it = internal_active_slots.begin();
                    while (internal_it != internal_active_slots.end()) {
                        if (internal_it->last_use < output.produced_at) {
                            free_slots.push_back(internal_it->slot);
                            internal_it = internal_active_slots.erase(internal_it);
                        } else {
                            ++internal_it;
                        }
                    }

                    auto& out_ref = stage.outputs[output.output];
                    const auto& plan = output_plan[stage_idx][output.output];
                    const size_t slot = acquire_slot(plan.desc);
                    if (plan.needs_buffer) {
                        out_ref->buf =
                            pool.ensure(workspace->handles[slot], plan.desc);
                    }
                    workspace->output_slots[stage_idx][output.output] = slot;
                    workspace->last_slots_used =
                        std::max(workspace->last_slots_used, slot + 1);
                    ++workspace->last_workspace_outputs;
                    if (output.escapes_stage) {
                        active_slots.push_back({slot, output.escape_last_use});
                        update_peak_live();
                    } else {
                        internal_active_slots.push_back({slot, output.last_used_at});
                        update_peak_live(internal_active_slots.size());
                    }
                }
                for (const auto& active : internal_active_slots) {
                    free_slots.push_back(active.slot);
                }
                free_slots.insert(free_slots.end(),
                                  protected_input_slots.begin(),
                                  protected_input_slots.end());
                continue;
            }

            for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
                auto& out_ref = stage.outputs[oi];
                if (!out_ref || out_ref->buf.valid()) {
                    continue;
                }
                const auto& plan = output_plan[stage_idx][oi];
                if (!plan.needs_buffer) {
                    continue;
                }
                if (!plan.workspace_managed) {
                    ++workspace->last_direct_outputs;
                    out_ref->buf = pool.ensure(stage_handles[oi], plan.desc);
                    continue;
                }
                const size_t slot = acquire_slot(plan.desc);
                if (plan.needs_buffer) {
                    out_ref->buf =
                        pool.ensure(workspace->handles[slot], plan.desc);
                }
                workspace->output_slots[stage_idx][oi] = slot;
                workspace->last_slots_used =
                    std::max(workspace->last_slots_used, slot + 1);
                ++workspace->last_workspace_outputs;
                active_slots.push_back({slot, last_use[stage_idx][oi]});
                update_peak_live();
            }
            free_slots.insert(free_slots.end(),
                              protected_input_slots.begin(),
                              protected_input_slots.end());
        }
        return;
    }

    for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
        auto& stage = pipeline[stage_idx];
        auto& stage_handles = handles[stage_idx];
        if (stage_handles.size() < stage.outputs.size()) {
            stage_handles.resize(stage.outputs.size());
        }
        for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
            auto& out_ref = stage.outputs[oi];
            if (!out_ref || out_ref->buf.valid()) {
                continue;
            }
            GpuBufferDesc desc{};
            if (!describe_output(stage, oi, *out_ref, desc, error_prefix)) {
                continue;
            }
            out_ref->buf = pool.ensure(stage_handles[oi], desc);
        }
    }
}

}  // namespace gfx_plugin
}  // namespace ov
