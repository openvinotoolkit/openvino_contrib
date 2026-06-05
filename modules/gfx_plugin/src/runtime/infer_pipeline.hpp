// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
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

void bind_remote_outputs(const std::vector<ov::Output<const ov::Node>>& outputs,
                         const std::shared_ptr<const ov::Model>& runtime_model,
                         const std::unordered_map<const ov::Node*, size_t>& node_map,
                         const std::unordered_map<const ov::Node*, size_t>& param_map,
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
    const std::shared_ptr<const ov::Model>& runtime_model,
    const std::vector<ov::Output<const ov::Node>>& outputs,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
    const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_inputs,
    const compiler::BackendTarget& expected_target,
    std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor,
    const char* error_prefix = "GFX");

void prepare_reusable_execution_plan(
    PreparedInferExecutionPlan& plan,
    const std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map);

void assign_runtime_stage_output_shapes(
    std::vector<InferStage>& pipeline,
    PreparedInferExecutionPlan& plan,
    const std::vector<GpuTensor>& input_tensors,
    const char* error_prefix = "GFX");

void prepare_reusable_output_plan(
    PreparedInferOutputPlan& plan,
    const std::vector<ov::Output<const ov::Node>>& public_outputs,
    const std::shared_ptr<const ov::Model>& runtime_model,
    const std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    const char* error_prefix = "GFX");

ov::Shape resolve_output_shape(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                               const OutputSource& source,
                               const GpuTensor& tensor,
                               size_t out_idx);

ov::element::Type resolve_output_element_type(const OutputSource& source,
                                              const GpuTensor& tensor,
                                              const char* error_prefix);

OutputSource resolve_output_source(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                                   const std::shared_ptr<const ov::Model>& runtime_model,
                                   size_t out_idx);

OutputViewInfo resolve_output_view(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                                   const std::shared_ptr<const ov::Model>& runtime_model,
                                   GpuTensor& tensor,
                                   size_t out_idx,
                                   const char* error_prefix);

GpuTensor* find_pipeline_output(std::vector<InferStage>& pipeline,
                                const ov::Node* node,
                                size_t port,
                                const char* error_prefix = "GFX");

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

struct StageOutputRef {
    size_t stage = 0;
    size_t output = 0;
};

struct NodePortKey {
    const ov::Node* node = nullptr;
    size_t port = 0;

    bool operator==(const NodePortKey& other) const {
        return node == other.node && port == other.port;
    }
};

struct NodePortKeyHash {
    size_t operator()(const NodePortKey& key) const {
        size_t h1 = std::hash<const ov::Node*>()(key.node);
        size_t h2 = std::hash<size_t>()(key.port);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

inline bool find_pipeline_output_ref(const std::vector<InferStage>& pipeline,
                                     const ov::Node* node,
                                     size_t port,
                                     size_t& stage_index,
                                     size_t& output_index) {
    if (!node) {
        return false;
    }
    for (size_t si = 0; si < pipeline.size(); ++si) {
        const auto& stage = pipeline[si];
        auto matches = [&](const std::shared_ptr<const ov::Node>& source_node,
                           size_t source_port,
                           size_t stage_output) {
            if (source_node.get() != node || source_port != port || stage_output >= stage.outputs.size()) {
                return false;
            }
            stage_index = si;
            output_index = stage_output;
            return true;
        };
        if (stage.node) {
            for (size_t oi = 0; oi < stage.outputs.size(); ++oi) {
                if (matches(stage.node, oi, oi)) {
                    return true;
                }
            }
        }
        for (size_t oi = 0; oi < stage.output_sources.size(); ++oi) {
            const auto& source = stage.output_sources[oi];
            if (matches(source.node, source.port, oi)) {
                return true;
            }
        }
        for (const auto& alias : stage.output_aliases) {
            if (matches(alias.node, alias.source_port, alias.output_port)) {
                return true;
            }
        }
    }
    return false;
}

inline bool stage_output_has_multiple_graph_consumers(const InferStage& stage, size_t output_index) {
    auto has_multiple_consumers = [](const std::shared_ptr<const ov::Node>& node, size_t port) {
        return node && port < node->get_output_size() && node->output(port).get_target_inputs().size() > 1;
    };
    if (output_index < stage.output_sources.size()) {
        const auto& source = stage.output_sources[output_index];
        if (has_multiple_consumers(source.node, source.port)) {
            return true;
        }
    }
    if (has_multiple_consumers(stage.node, output_index)) {
        return true;
    }
    for (const auto& alias : stage.output_aliases) {
        if (alias.output_port == output_index && has_multiple_consumers(alias.node, alias.source_port)) {
            return true;
        }
    }
    return false;
}

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
inline void for_each_output_tensor(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                                   const std::shared_ptr<const ov::Model>& runtime_model,
                                   const std::unordered_map<const ov::Node*, size_t>& node_map,
                                   const std::unordered_map<const ov::Node*, size_t>& param_map,
                                   std::vector<InferStage>& pipeline,
                                   InputLookup&& input_lookup,
                                   const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                                   HostOverrideGetter&& host_override_getter,
                                   RemoteHandler&& on_remote,
                                   LocalHandler&& on_local,
                                   const PreparedInferOutputPlan* prepared_plan,
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
                             : resolve_output_shape(public_outputs, prepared.source, tensor, out_idx);
            if (tensor.shape.empty() && !info.shape.empty()) {
                tensor.shape = info.shape;
            }
            info.type = prepared.static_type != ov::element::dynamic
                            ? prepared.static_type
                            : resolve_output_element_type(prepared.source, tensor, error_prefix);
            if (!tensor.shape.empty()) {
                info.shape = tensor.shape;
            }
        };

    if (prepared_plan && prepared_plan->outputs.size() == public_outputs.size()) {
        for (size_t out_idx = 0; out_idx < public_outputs.size(); ++out_idx) {
            if (out_idx < remote_outputs.size() && remote_outputs[out_idx]) {
                on_remote(out_idx, remote_outputs[out_idx]);
                continue;
            }
            const auto& prepared = prepared_plan->outputs[out_idx];
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
        return;
    }

    for (size_t out_idx = 0; out_idx < public_outputs.size(); ++out_idx) {
        if (out_idx < remote_outputs.size() && remote_outputs[out_idx]) {
            on_remote(out_idx, remote_outputs[out_idx]);
            continue;
        }
        auto* dev = resolve_output_tensor(public_outputs,
                                          runtime_model,
                                          node_map,
                                          param_map,
                                          pipeline,
                                          input_lookup,
                                          out_idx,
                                          allow_missing,
                                          error_prefix);
        if (!dev || !dev->buf.valid()) {
            if (allow_missing) {
                continue;
            }
            OPENVINO_THROW(error_prefix, ": output tensor missing (pipeline incomplete)");
        }
        auto info = resolve_output_view(public_outputs,
                                        runtime_model,
                                        *dev,
                                        out_idx,
                                        error_prefix);
        auto host_override = host_override_getter(out_idx, info.type, info.shape, error_prefix);
        on_local(out_idx, *dev, info, host_override);
    }
}

template <typename InputLookup>
inline std::vector<GpuTensor*> resolve_stage_inputs(
    const InferStage& stage,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    const std::vector<InferStage>& pipeline,
    InputLookup&& input_lookup) {
    std::vector<GpuTensor*> resolved;
    resolved.reserve(stage.inputs.size());
    for (const auto& link : stage.inputs) {
        if (!link.node) {
            resolved.push_back(nullptr);
            continue;
        }
        if (auto itp = param_map.find(link.node.get()); itp != param_map.end()) {
            resolved.push_back(input_lookup(itp->second));
            continue;
        }
        size_t stage_idx = 0;
        size_t output_idx = 0;
        if (find_pipeline_output_ref(pipeline, link.node.get(), link.port, stage_idx, output_idx)) {
            resolved.push_back(pipeline[stage_idx].outputs[output_idx].get());
            continue;
        }
        if (auto it = node_map.find(link.node.get()); it != node_map.end()) {
            const auto& src_stage = pipeline[it->second];
            GpuTensor* tensor = nullptr;
            if (link.port < src_stage.outputs.size()) {
                tensor = src_stage.outputs[link.port].get();
            }
            resolved.push_back(tensor);
            continue;
        }
        resolved.push_back(nullptr);  // constants handled inside ops
    }
    return resolved;
}

template <typename InputLookup, typename StageFn>
inline void execute_pipeline(std::vector<InferStage>& pipeline,
                             const std::unordered_map<const ov::Node*, size_t>& node_map,
                             const std::unordered_map<const ov::Node*, size_t>& param_map,
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

    if (prepared_plan && prepared_plan->stages.size() == pipeline.size()) {
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
        return;
    }

    for (auto& stage : pipeline) {
        auto resolved = resolve_stage_inputs(stage,
                                             node_map,
                                             param_map,
                                             pipeline,
                                             std::forward<InputLookup>(input_lookup));
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
                                           const std::unordered_map<const ov::Node*, size_t>& node_map,
                                           const std::unordered_map<const ov::Node*, size_t>& param_map,
                                           InputLookup&& input_lookup,
                                           PreparedInferExecutionPlan* prepared_plan = nullptr) {
    execute_pipeline(pipeline,
                     node_map,
                     param_map,
                     std::forward<InputLookup>(input_lookup),
                     [](InferStage& stage, const std::vector<GpuTensor*>&) {
                         if (stage.stage) {
                             stage.stage->prewarm_runtime_state();
                         }
                     },
                     prepared_plan);
}

template <typename InputLookup>
inline GpuTensor* resolve_output_tensor(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                                        const std::shared_ptr<const ov::Model>& runtime_model,
                                        const std::unordered_map<const ov::Node*, size_t>& node_map,
                                        const std::unordered_map<const ov::Node*, size_t>& param_map,
                                        std::vector<InferStage>& pipeline,
                                        InputLookup&& input_lookup,
                                        size_t out_idx,
                                        bool allow_missing,
                                        const char* error_prefix = "GFX") {
    const auto src = resolve_output_source(public_outputs, runtime_model, out_idx);
    if (!src.node) {
        if (allow_missing) {
            return nullptr;
        }
        OPENVINO_THROW(error_prefix, ": output source node is null for index ", out_idx);
    }
    if (auto it = node_map.find(src.node.get()); it != node_map.end()) {
        auto& outs = pipeline[it->second].outputs;
        OPENVINO_ASSERT(src.port < outs.size(), error_prefix, ": output port out of range");
        return outs[src.port].get();
    }
    if (auto* tensor = find_pipeline_output(pipeline, src.node.get(), src.port, error_prefix)) {
        return tensor;
    }
    if (auto pit = param_map.find(src.node.get()); pit != param_map.end()) {
        auto* input_tensor = input_lookup(pit->second);
        if (!input_tensor && !allow_missing) {
            OPENVINO_THROW(error_prefix, ": input tensor missing for passthrough output ", out_idx);
        }
        return input_tensor;
    }
    if (allow_missing) {
        return nullptr;
    }
    OPENVINO_THROW(error_prefix, ": failed to resolve output ", out_idx, " (pipeline incomplete)");
}

}  // namespace gfx_plugin
}  // namespace ov
