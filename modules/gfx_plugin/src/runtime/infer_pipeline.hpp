// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "compiler/backend_target.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/infer_pipeline_state.hpp"

namespace ov {
namespace gfx_plugin {

bool is_view_op(const InferStage& stage);

std::vector<InferStage> build_infer_pipeline(const std::vector<PipelineStageDesc>& descs,
                                             GpuBufferManager* buffer_manager,
                                             void* profiler,
                                             bool profiling_enabled,
                                             std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor);

void bind_remote_outputs(const PreparedInferOutputPlan& output_plan,
                         const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                         const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_inputs,
                         std::vector<InferStage>& pipeline,
                         const char* error_prefix = "GFX");

ov::Shape ensure_stage_output_shape(InferStage& stage, size_t out_idx);

ov::element::Type resolve_stage_output_type(const InferStage& stage,
                                            const GpuTensor& out,
                                            size_t out_idx,
                                            const char* error_prefix = "GFX");

void normalize_remote_tensor(GfxRemoteTensor& remote,
                             const compiler::BackendTarget& expected_target,
                             const char* error_prefix);

void normalize_remote_outputs(std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                              const compiler::BackendTarget& expected_target,
                              const char* error_prefix);

std::vector<InferStage> build_bound_pipeline(
    const std::vector<PipelineStageDesc>& descs,
    GpuBufferManager* buffer_manager,
    void* profiler,
    bool profiling_enabled,
    std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor,
    const char* error_prefix = "GFX");

void prepare_reusable_execution_plan(
    PreparedInferExecutionPlan& plan,
    const std::vector<InferStage>& pipeline);

void assign_runtime_stage_output_shapes(
    std::vector<InferStage>& pipeline,
    PreparedInferExecutionPlan& plan,
    const std::vector<GpuTensor>& input_tensors,
    const char* error_prefix = "GFX");

void prepare_reusable_output_plan(
    PreparedInferOutputPlan& plan,
    const RuntimeExecutableDescriptor& runtime_descriptor,
    const std::vector<InferStage>& pipeline,
    const char* error_prefix = "GFX");

ov::element::Type resolve_output_element_type(const GpuTensor& tensor,
                                              const char* error_prefix);

inline const RuntimeStageExecutableDescriptor*
runtime_stage_descriptor_or_null(const InferStage& stage) {
    if (stage.runtime_stage_descriptor) {
        return stage.runtime_stage_descriptor.get();
    }
    if (!stage.runtime_session ||
        stage.runtime_stage_index == PipelineStageDesc::npos) {
        return nullptr;
    }
    const auto& runtime_descriptor = stage.runtime_session->descriptor();
    if (stage.runtime_stage_index >= runtime_descriptor.stages.size()) {
        return nullptr;
    }
    return &runtime_descriptor.stages[stage.runtime_stage_index];
}

inline bool stage_outputs_may_alias_inputs(const InferStage& stage) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (!descriptor) {
        return false;
    }
    if (descriptor->tensor_view_only) {
        return true;
    }
    const auto& memory_plan = stage.runtime_session->descriptor().memory_plan;
    for (const auto& output : descriptor->output_bindings) {
        const auto alias_it =
            std::find_if(memory_plan.alias_groups.begin(),
                         memory_plan.alias_groups.end(),
                         [&](const RuntimeMemoryAliasGroupDescriptor& group) {
                             return group.group_id == output.alias_group;
                         });
        if (alias_it != memory_plan.alias_groups.end() &&
            alias_it->output_aliasing) {
            return true;
        }
    }
    return false;
}

inline const RuntimeMemoryRegionDescriptor*
runtime_output_memory_region_or_null(const InferStage& stage, size_t output_index) {
    const auto* stage_descriptor = runtime_stage_descriptor_or_null(stage);
    if (!stage_descriptor || !stage.runtime_session) {
        return nullptr;
    }
    const auto& runtime_descriptor = stage.runtime_session->descriptor();
    if (output_index >= stage_descriptor->output_bindings.size()) {
        return nullptr;
    }
    const auto& region_id =
        stage_descriptor->output_bindings[output_index].memory_region_id;
    if (region_id.empty()) {
        return nullptr;
    }
    const auto& regions = runtime_descriptor.memory_plan.regions;
    const auto it = std::find_if(
        regions.begin(), regions.end(),
        [&](const RuntimeMemoryRegionDescriptor& region) {
            return region.region_id == region_id;
        });
    return it == regions.end() ? nullptr : &*it;
}

bool runtime_output_uses_transient_arena(const InferStage& stage, size_t output_index);

bool apply_runtime_output_memory_contract(const InferStage& stage,
                                          size_t output_index,
                                          GpuBufferDesc& desc,
                                          GpuTensor& output,
                                          const char* error_prefix = "GFX");

using StageOutputDescInitializer = std::function<bool(InferStage&,
                                                      size_t,
                                                      GpuTensor&,
                                                      GpuBufferDesc&,
                                                      const char*)>;

void allocate_stage_outputs(std::vector<InferStage>& pipeline,
                            std::vector<std::vector<BufferHandle>>& handles,
                            GpuBufferPool& pool,
                            const StageOutputDescInitializer& describe_output,
                            StageOutputBufferWorkspace* workspace = nullptr,
                            const char* error_prefix = "GFX");

template <typename RemoteResolver, typename HostResolver, typename RemoteHandler, typename HostHandler>
inline void for_each_input_tensor(size_t input_count,
                                  const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_inputs,
                                  RemoteResolver&& resolve_remote,
                                  HostResolver&& resolve_host,
                                  RemoteHandler&& on_remote,
                                  HostHandler&& on_host) {
    for (size_t idx = 0; idx < input_count; ++idx) {
        if (idx < remote_inputs.size() && remote_inputs[idx]) {
            auto dev = resolve_remote(idx);
            on_remote(idx, dev);
            continue;
        }
        auto host = resolve_host(idx);
        on_host(idx, host);
    }
}

inline GpuTensor* lookup_runtime_input_tensor(std::vector<GpuTensor>& input_tensors,
                                              size_t input_idx) {
    if (input_idx < input_tensors.size()) {
        return &input_tensors[input_idx];
    }
    return nullptr;
}

template <typename InputLookup,
          typename HostOverrideGetter,
          typename RemoteHandler,
          typename LocalHandler>
inline void for_each_output_tensor(std::vector<InferStage>& pipeline,
                                   InputLookup&& input_lookup,
                                   const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                                   HostOverrideGetter&& host_override_getter,
                                   RemoteHandler&& on_remote,
                                   LocalHandler&& on_local,
                                   const PreparedInferOutputPlan& prepared_plan,
                                   bool allow_missing,
                                   const char* error_prefix = "GFX") {
    auto resolve_prepared_output_tensor =
        [&](const PreparedOutputBinding& prepared) -> GpuTensor* {
        switch (prepared.kind) {
        case PreparedOutputSourceKind::Parameter:
            return input_lookup(prepared.index);
        case PreparedOutputSourceKind::StageOutput:
            if (prepared.index >= pipeline.size()) {
                return nullptr;
            }
            if (prepared.port >= pipeline[prepared.index].outputs.size()) {
                return nullptr;
            }
            return pipeline[prepared.index].outputs[prepared.port].get();
        case PreparedOutputSourceKind::None:
        default:
            return nullptr;
        }
    };

    auto fill_prepared_output_view =
        [&](OutputViewInfo& info, const PreparedOutputBinding& prepared, GpuTensor& tensor, size_t out_idx) {
            info.source = prepared.source;
            info.shape = !prepared.static_shape.empty()
                             ? prepared.static_shape
                             : tensor.shape;
            if (tensor.shape.empty() && !info.shape.empty()) {
                tensor.shape = info.shape;
            }
            info.type = prepared.static_type != ov::element::dynamic
                            ? prepared.static_type
                            : resolve_output_element_type(tensor, error_prefix);
            if (!tensor.shape.empty()) {
                info.shape = tensor.shape;
            }
        };

    for (size_t out_idx = 0; out_idx < prepared_plan.outputs.size(); ++out_idx) {
        if (out_idx < remote_outputs.size() && remote_outputs[out_idx]) {
            on_remote(out_idx, remote_outputs[out_idx]);
            continue;
        }
        const auto& prepared = prepared_plan.outputs[out_idx];
        auto* dev = resolve_prepared_output_tensor(prepared);
        if (!dev || !dev->buf.valid()) {
            if (allow_missing) {
                continue;
            }
            OPENVINO_THROW(error_prefix, ": output tensor missing (prepared pipeline incomplete)");
        }
        OutputViewInfo info;
        fill_prepared_output_view(info, prepared, *dev, out_idx);
        auto host_override = host_override_getter(out_idx, info.type, info.shape, error_prefix);
        on_local(out_idx, *dev, info, host_override);
    }
}

template <typename InputLookup, typename StageFn>
inline void execute_pipeline(std::vector<InferStage>& pipeline,
                             InputLookup&& input_lookup,
                             StageFn&& on_stage,
                             PreparedInferExecutionPlan* prepared_plan = nullptr) {
    auto output_refs = [](InferStage& stage) {
        std::vector<GpuTensor*> outputs;
        outputs.reserve(stage.outputs.size());
        for (auto& output : stage.outputs) {
            outputs.push_back(output.get());
        }
        return outputs;
    };
    auto bind_prepared_runtime_resources = [&](InferStage& stage,
                                               std::vector<GpuTensor*>& resolved) {
        if (!stage.runtime_session || !stage.prepared_executable) {
            return;
        }
        auto bindings =
            ResourceBindingTable::for_stage(resolved,
                                            output_refs(stage),
                                            stage.prepared_executable->descriptor());
        stage.prepared_executable->bind(std::move(bindings));
    };
    auto resolve_prepared_inputs = [&](PreparedStageExecution& prepared) -> std::vector<GpuTensor*>& {
        for (size_t input_idx = 0; input_idx < prepared.input_refs.size(); ++input_idx) {
            const auto& ref = prepared.input_refs[input_idx];
            auto& resolved = prepared.resolved_inputs[input_idx];
            switch (ref.kind) {
            case PreparedStageInputKind::Parameter:
                resolved = input_lookup(ref.index);
                break;
            case PreparedStageInputKind::StageOutput:
            case PreparedStageInputKind::None:
            default:
                break;
            }
        }
        return prepared.resolved_inputs;
    };

    PreparedInferExecutionPlan local_plan;
    if (!prepared_plan || prepared_plan->stages.size() != pipeline.size()) {
        prepare_reusable_execution_plan(local_plan, pipeline);
        prepared_plan = &local_plan;
    }

    for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
        auto& stage = pipeline[stage_idx];
        auto& resolved = resolve_prepared_inputs(prepared_plan->stages[stage_idx]);
        if (!stage.stage) {
            continue;
        }
        bind_prepared_runtime_resources(stage, resolved);
        stage.stage->set_inputs(resolved);
        on_stage(stage, resolved);
    }
}

template <typename InputLookup>
inline void prewarm_pipeline_runtime_state(std::vector<InferStage>& pipeline,
                                           InputLookup&& input_lookup,
                                           PreparedInferExecutionPlan* prepared_plan = nullptr) {
    execute_pipeline(pipeline,
                     std::forward<InputLookup>(input_lookup),
                     [](InferStage& stage, const std::vector<GpuTensor*>&) {
                         if (stage.stage) {
                             stage.stage->prewarm_runtime_state();
                         }
                     },
                     prepared_plan);
}

}  // namespace gfx_plugin
}  // namespace ov
