// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/infer_pipeline.hpp"

#include "runtime/runtime_shape_materializer.hpp"
#include "runtime/tensor_binding_contract.hpp"

#include <limits>
#include <utility>

namespace ov {
namespace gfx_plugin {

namespace {

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

std::string stage_descriptor_name(const InferStage& stage) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (descriptor && !descriptor->stage_name.empty()) {
        return descriptor->stage_name;
    }
    return stage.stage ? stage.stage->name() : std::string("<null>");
}

std::string stage_descriptor_op_family(const InferStage& stage) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (descriptor && !descriptor->op_family.empty()) {
        return descriptor->op_family;
    }
    return stage.stage ? stage.stage->type() : std::string("<null>");
}

bool descriptor_output_static_shape(const InferStage& stage,
                                    size_t output_idx,
                                    ov::Shape& shape) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (!descriptor || output_idx >= descriptor->output_bindings.size()) {
        return false;
    }
    return parse_static_shape_contract(
        descriptor->output_bindings[output_idx].partial_shape, shape);
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

struct StageOutputRef {
    size_t stage = PipelineStageTensorRef::npos;
    size_t output = PipelineStageTensorRef::npos;
};

bool resolve_stage_output_ref(const PipelineStageTensorRef& ref,
                              const std::vector<InferStage>& pipeline,
                              StageOutputRef& resolved) {
    if (ref.kind != PipelineStageTensorRefKind::StageOutput ||
        ref.index >= pipeline.size()) {
        return false;
    }
    const size_t output_port =
        ref.port == PipelineStageTensorRef::npos ? 0 : ref.port;
    if (output_port >= pipeline[ref.index].outputs.size()) {
        return false;
    }
    resolved.stage = ref.index;
    resolved.output = output_port;
    return true;
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
    if (out->shape.empty()) {
        ov::Shape descriptor_shape;
        if (descriptor_output_static_shape(stage, out_idx, descriptor_shape)) {
            out->shape = std::move(descriptor_shape);
        }
    }
    return out->shape;
}

ov::element::Type resolve_stage_output_type(const InferStage& stage,
                                            const GpuTensor& out,
                                            size_t out_idx,
                                            const char* error_prefix) {
    (void)stage;
    (void)out_idx;
    if (out.expected_type != ov::element::dynamic) {
        return out.expected_type;
    }
    if (out.buf.type != ov::element::dynamic) {
        return out.buf.type;
    }
    OPENVINO_THROW(error_prefix,
                   ": stage output type is not known from compiler descriptor or tensor binding");
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
        stage.stage = desc.stage->clone();
        const auto& descriptor = runtime_session->stage_descriptor(runtime_stage_index);
        const std::string stage_name =
            !descriptor.stage_name.empty()
                ? descriptor.stage_name
                : std::string("<unnamed>");
        OPENVINO_ASSERT(stage.stage, "GFX: failed to clone stage for ", stage_name);
        stage.runtime_stage_index = runtime_stage_index;
        stage.runtime_stage_descriptor =
            desc.runtime_descriptor
                ? desc.runtime_descriptor
                : std::make_shared<RuntimeStageExecutableDescriptor>(
                      descriptor);
        stage.inputs = desc.inputs;
        stage.output_lifetimes = desc.output_lifetimes;
        stage.outputs.reserve(desc.outputs.size());
        stage.output_is_model_output.reserve(desc.outputs.size());
        stage.direct_stateful_assign_variable_ids.reserve(desc.outputs.size());
        for (const auto& out_desc : desc.outputs) {
            auto out_tensor = std::make_unique<GpuTensor>();
            out_tensor->shape = out_desc.shape;
            out_tensor->expected_type = out_desc.type;
            stage.outputs.emplace_back(std::move(out_tensor));
            stage.output_is_model_output.push_back(out_desc.is_model_output);
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
            stage.stage->set_profiler(profiler,
                                      static_cast<uint32_t>(stage_id),
                                      stage_descriptor_name(stage),
                                      stage_descriptor_op_family(stage));
        }
        prepare_stage_runtime_executable(stage,
                                         runtime_stage_index,
                                         buffer_manager,
                                         runtime_session);
        pipeline.emplace_back(std::move(stage));
    }
    return pipeline;
}

void bind_remote_outputs(const PreparedInferOutputPlan& output_plan,
                         const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                         const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_inputs,
                         std::vector<InferStage>& pipeline,
                         const char* error_prefix) {
    if (remote_outputs.empty()) {
        return;
    }
    for (size_t out_idx = 0; out_idx < output_plan.outputs.size(); ++out_idx) {
        if (out_idx >= remote_outputs.size() || !remote_outputs[out_idx]) {
            continue;
        }
        const auto& prepared = output_plan.outputs[out_idx];
        if (prepared.kind == PreparedOutputSourceKind::StageOutput) {
            OPENVINO_ASSERT(prepared.index < pipeline.size(),
                            error_prefix, ": remote output stage index out of range");
            auto& outs = pipeline[prepared.index].outputs;
            OPENVINO_ASSERT(prepared.port < outs.size(),
                            error_prefix, ": remote output port out of range");
            auto& dst = outs[prepared.port];
            const auto& src_tensor = remote_outputs[out_idx]->gpu_tensor();
            dst->buf = src_tensor.buf;
            dst->shape = src_tensor.shape;
            dst->expected_type = src_tensor.expected_type;
            continue;
        }
        if (prepared.kind == PreparedOutputSourceKind::Parameter) {
            const size_t input_idx = prepared.index;
            if (input_idx < remote_inputs.size() && remote_inputs[input_idx]) {
                auto in_buf = remote_inputs[input_idx]->gpu_tensor().buf.buffer;
                auto out_buf = remote_outputs[out_idx]->gpu_tensor().buf.buffer;
                OPENVINO_ASSERT(in_buf == out_buf,
                                error_prefix, ": remote output must alias remote input for passthrough outputs");
                continue;
            }
            OPENVINO_THROW(error_prefix, ": remote output cannot be bound to non-remote input passthrough");
        }
        OPENVINO_THROW(error_prefix, ": remote output descriptor missing for output ", out_idx);
    }
}

std::vector<InferStage> build_bound_pipeline(
    const std::vector<PipelineStageDesc>& descs,
    GpuBufferManager* buffer_manager,
    void* profiler,
    bool profiling_enabled,
    std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor,
    const char* error_prefix) {
    (void)error_prefix;
    return build_infer_pipeline(descs,
                                buffer_manager,
                                profiler,
                                profiling_enabled,
                                std::move(runtime_descriptor));
}

void prepare_reusable_execution_plan(
    PreparedInferExecutionPlan& plan,
    const std::vector<InferStage>& pipeline) {
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
            if (link.source_ref.kind == PipelineStageTensorRefKind::Parameter) {
                OPENVINO_ASSERT(link.source_ref.index != PipelineStageTensorRef::npos,
                                "GFX: compiler-owned parameter input ref is incomplete");
                ref.kind = PreparedStageInputKind::Parameter;
                ref.index = link.source_ref.index;
                ref.port = link.source_ref.port == PipelineStageTensorRef::npos
                               ? 0
                               : link.source_ref.port;
            } else if (link.source_ref.kind == PipelineStageTensorRefKind::StageOutput) {
                OPENVINO_ASSERT(link.source_ref.index < pipeline.size(),
                                "GFX: compiler-owned stage input ref points outside the pipeline");
                ref.kind = PreparedStageInputKind::StageOutput;
                ref.index = link.source_ref.index;
                ref.port = link.source_ref.port == PipelineStageTensorRef::npos
                               ? 0
                               : link.source_ref.port;
                const auto& src_stage = pipeline[ref.index];
                OPENVINO_ASSERT(ref.port < src_stage.outputs.size(),
                                "GFX: compiler-owned stage input ref points outside producer outputs");
                resolved = src_stage.outputs[ref.port].get();
            } else {
                ref.kind = PreparedStageInputKind::None;
                if (link.source_ref.kind != PipelineStageTensorRefKind::None) {
                    OPENVINO_THROW("GFX: compiler-owned stage input ref has unsupported kind");
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

        materialize_runtime_stage_output_shapes(stage,
                                                prepared.resolved_inputs,
                                                error_prefix);
    }
}

void prepare_reusable_output_plan(
    PreparedInferOutputPlan& plan,
    const RuntimeExecutableDescriptor& runtime_descriptor,
    const std::vector<InferStage>& pipeline,
    const char* error_prefix) {
    const auto& public_outputs = runtime_descriptor.public_outputs;
    if (plan.outputs.size() == public_outputs.size()) {
        bool compatible = true;
        for (size_t out_idx = 0; out_idx < public_outputs.size(); ++out_idx) {
            const auto& descriptor = public_outputs[out_idx];
            const auto& prepared = plan.outputs[out_idx];
            const bool same_kind =
                (descriptor.kind == RuntimePublicOutputSourceKind::Parameter &&
                 prepared.kind == PreparedOutputSourceKind::Parameter) ||
                (descriptor.kind == RuntimePublicOutputSourceKind::StageOutput &&
                 prepared.kind == PreparedOutputSourceKind::StageOutput) ||
                (descriptor.kind == RuntimePublicOutputSourceKind::None &&
                 prepared.kind == PreparedOutputSourceKind::None);
            if (!same_kind || prepared.index != descriptor.index ||
                prepared.port != descriptor.port ||
                prepared.static_shape != descriptor.static_shape ||
                prepared.static_type != descriptor.static_type) {
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
        const auto& descriptor = public_outputs[out_idx];
        prepared.index = descriptor.index;
        prepared.port = descriptor.port;
        prepared.static_shape = descriptor.static_shape;
        prepared.static_type = descriptor.static_type;
        if (descriptor.kind == RuntimePublicOutputSourceKind::StageOutput) {
            prepared.kind = PreparedOutputSourceKind::StageOutput;
        } else if (descriptor.kind == RuntimePublicOutputSourceKind::Parameter) {
            prepared.kind = PreparedOutputSourceKind::Parameter;
        } else {
            prepared.kind = PreparedOutputSourceKind::None;
        }

        if (prepared.kind == PreparedOutputSourceKind::StageOutput && prepared.index < pipeline.size() &&
            prepared.port < pipeline[prepared.index].outputs.size() &&
            pipeline[prepared.index].outputs[prepared.port]) {
            const auto& tensor = *pipeline[prepared.index].outputs[prepared.port];
            if (prepared.static_shape.empty()) {
                prepared.static_shape = tensor.shape;
            }
            if (prepared.static_type == ov::element::dynamic) {
                prepared.static_type =
                    resolve_output_element_type(tensor, error_prefix);
            }
        }
        if (prepared.kind == PreparedOutputSourceKind::StageOutput) {
            OPENVINO_ASSERT(prepared.index < pipeline.size(),
                            error_prefix,
                            ": public output descriptor stage index out of range at ",
                            out_idx);
            OPENVINO_ASSERT(prepared.port < pipeline[prepared.index].outputs.size(),
                            error_prefix,
                            ": public output descriptor port out of range at ",
                            out_idx);
        } else if (prepared.kind == PreparedOutputSourceKind::Parameter) {
            OPENVINO_ASSERT(prepared.index != PipelineStageTensorRef::npos,
                            error_prefix,
                            ": public output descriptor parameter index is missing at ",
                            out_idx);
        } else {
            OPENVINO_THROW(error_prefix,
                           ": public output descriptor source is missing at ",
                           out_idx);
        }
        prepared.source = {prepared.kind, prepared.index, prepared.port};
    }
}

ov::element::Type resolve_output_element_type(const GpuTensor& tensor,
                                              const char* error_prefix) {
    if (tensor.expected_type != ov::element::dynamic) {
        return tensor.expected_type;
    }
    if (tensor.buf.type != ov::element::dynamic) {
        return tensor.buf.type;
    }
    OPENVINO_THROW(error_prefix,
                   ": output element type is not known from compiler descriptor or tensor binding");
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
        std::vector<std::vector<size_t>> output_consumer_count(pipeline.size());
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

        for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
            auto& stage = pipeline[stage_idx];
            auto& stage_handles = handles[stage_idx];
            if (stage_handles.size() < stage.outputs.size()) {
                stage_handles.resize(stage.outputs.size());
            }
            output_plan[stage_idx].resize(stage.outputs.size());
            last_use[stage_idx].assign(stage.outputs.size(), stage_idx);
            output_consumer_count[stage_idx].assign(stage.outputs.size(), 0);
            workspace->output_slots[stage_idx].assign(
                stage.outputs.size(), StageOutputBufferWorkspace::npos);
        }

        for (size_t consumer_idx = 0; consumer_idx < pipeline.size(); ++consumer_idx) {
            const auto& consumer = pipeline[consumer_idx];
            for (const auto& link : consumer.inputs) {
                StageOutputRef producer_ref;
                if (!resolve_stage_output_ref(link.source_ref, pipeline, producer_ref) ||
                    producer_ref.stage == consumer_idx ||
                    producer_ref.output >= last_use[producer_ref.stage].size()) {
                    continue;
                }
                last_use[producer_ref.stage][producer_ref.output] =
                    std::max(last_use[producer_ref.stage][producer_ref.output],
                             consumer_idx);
                ++output_consumer_count[producer_ref.stage][producer_ref.output];
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
                StageOutputRef producer_ref;
                if (!resolve_stage_output_ref(link.source_ref, pipeline, producer_ref) ||
                    producer_ref.output >= last_use[producer_ref.stage].size()) {
                    continue;
                }
                last_use[producer_ref.stage][producer_ref.output] =
                    std::max(last_use[producer_ref.stage][producer_ref.output],
                             view_last_use);
            }
        }

        for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
            auto& stage = pipeline[stage_idx];
            auto& stage_handles = handles[stage_idx];
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
                plan.workspace_managed = runtime_output_uses_transient_arena(stage, oi) &&
                                         desc.usage == BufferUsage::Intermediate &&
                                         !desc.cpu_read &&
                                         !desc.cpu_write &&
                                         output_consumer_count[stage_idx][oi] <= 1u;
                if (!plan.workspace_managed) {
                    continue;
                }
                if (stage_handles[oi].valid()) {
                    pool.release(stage_handles[oi]);
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
                StageOutputRef producer_ref;
                if (!resolve_stage_output_ref(link.source_ref, pipeline, producer_ref)) {
                    continue;
                }
                const size_t producer_idx = producer_ref.stage;
                const size_t producer_output = producer_ref.output;
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
