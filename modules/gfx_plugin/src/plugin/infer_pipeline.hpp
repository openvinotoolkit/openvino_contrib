// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "runtime/gpu_buffer_pool.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

struct InferStage {
    std::shared_ptr<const ov::Node> node;
    std::unique_ptr<GpuStage> stage;
    std::vector<std::unique_ptr<GpuTensor>> outputs;
    std::vector<bool> output_is_model_output;
    std::vector<PipelineStageDesc::InputLink> inputs;
};

struct StageOutputBufferWorkspace {
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    std::vector<BufferHandle> handles;
    std::vector<std::vector<size_t>> output_slots;
    size_t last_workspace_outputs = 0;
    size_t last_legacy_outputs = 0;
    size_t last_slots_used = 0;
    size_t last_peak_live_slots = 0;
};

enum class PreparedStageInputKind {
    None,
    Parameter,
    StageOutput,
};

struct PreparedStageInputRef {
    PreparedStageInputKind kind = PreparedStageInputKind::None;
    size_t index = 0;
    size_t port = 0;
};

struct PreparedStageExecution {
    std::vector<PreparedStageInputRef> input_refs;
    std::vector<GpuTensor*> resolved_inputs;
};

struct PreparedInferExecutionPlan {
    std::vector<PreparedStageExecution> stages;
};

struct OutputSource {
    std::shared_ptr<const ov::Node> node;
    size_t port = 0;
};

enum class PreparedOutputSourceKind {
    None,
    Parameter,
    StageOutput,
};

struct PreparedOutputBinding {
    PreparedOutputSourceKind kind = PreparedOutputSourceKind::None;
    size_t index = 0;
    size_t port = 0;
    OutputSource source;
    ov::Shape static_shape;
    ov::element::Type static_type = ov::element::dynamic;
};

struct PreparedInferOutputPlan {
    std::vector<PreparedOutputBinding> outputs;
};

bool is_view_op(const InferStage& stage);

std::vector<InferStage> build_infer_pipeline(const std::vector<PipelineStageDesc>& descs,
                                             GpuBufferManager* buffer_manager,
                                             void* profiler,
                                             bool profiling_enabled);

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

struct OutputViewInfo {
    OutputSource source;
    ov::Shape shape;
    ov::element::Type type = ov::element::dynamic;
};

void normalize_remote_tensor(GfxRemoteTensor& remote,
                             GpuBackend expected_backend,
                             const char* error_prefix);

void normalize_remote_outputs(std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
                              GpuBackend expected_backend,
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
    GpuBackend expected_backend,
    const char* error_prefix = "GFX");

void prepare_reusable_execution_plan(
    PreparedInferExecutionPlan& plan,
    const std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map);

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

template <typename DescribeOutput>
inline void allocate_stage_outputs(std::vector<InferStage>& pipeline,
                                   std::vector<std::vector<BufferHandle>>& handles,
                                   GpuBufferPool& pool,
                                   DescribeOutput&& describe_output,
                                   StageOutputBufferWorkspace* workspace = nullptr,
                                   const char* error_prefix = "GFX") {
    if (handles.size() != pipeline.size()) {
        handles.assign(pipeline.size(), {});
    }

    if (workspace) {
        struct OutputPlan {
            bool needs_buffer = false;
            bool workspace_managed = false;
            GpuBufferDesc desc{};
        };
        struct ActiveSlot {
            size_t slot = StageOutputBufferWorkspace::npos;
            size_t last_use = 0;
        };

        std::vector<std::vector<OutputPlan>> output_plan(pipeline.size());
        std::vector<std::vector<size_t>> last_use(pipeline.size());
        workspace->output_slots.assign(pipeline.size(), {});
        workspace->last_workspace_outputs = 0;
        workspace->last_legacy_outputs = 0;
        workspace->last_slots_used = 0;
        workspace->last_peak_live_slots = 0;
        size_t max_new_workspace_slots = 0;
        for (const auto& stage : pipeline) {
            max_new_workspace_slots += stage.outputs.size();
        }
        workspace->handles.reserve(workspace->handles.size() + max_new_workspace_slots);

        std::unordered_map<const ov::Node*, size_t> stage_by_node;
        stage_by_node.reserve(pipeline.size());

        for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
            auto& stage = pipeline[stage_idx];
            if (stage.node) {
                stage_by_node[stage.node.get()] = stage_idx;
            }
            auto& stage_handles = handles[stage_idx];
            if (stage_handles.size() < stage.outputs.size()) {
                stage_handles.resize(stage.outputs.size());
            }
            output_plan[stage_idx].resize(stage.outputs.size());
            last_use[stage_idx].assign(stage.outputs.size(), stage_idx);
            workspace->output_slots[stage_idx].assign(stage.outputs.size(), StageOutputBufferWorkspace::npos);

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
                                         desc.prefer_device_local &&
                                         !desc.cpu_read &&
                                         !desc.cpu_write;
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
                auto it = stage_by_node.find(link.node.get());
                if (it == stage_by_node.end()) {
                    continue;
                }
                const size_t producer_idx = it->second;
                if (link.port >= last_use[producer_idx].size()) {
                    continue;
                }
                last_use[producer_idx][link.port] =
                    std::max(last_use[producer_idx][link.port], consumer_idx);
            }
        }

        for (size_t stage_idx = 0; stage_idx < pipeline.size(); ++stage_idx) {
            const auto& stage = pipeline[stage_idx];
            for (size_t oi = 0; oi < stage.output_is_model_output.size() && oi < last_use[stage_idx].size(); ++oi) {
                if (stage.output_is_model_output[oi]) {
                    last_use[stage_idx][oi] = pipeline.size();
                }
            }
        }

        for (size_t rev = pipeline.size(); rev > 0; --rev) {
            const size_t stage_idx = rev - 1;
            const auto& stage = pipeline[stage_idx];
            if (!is_view_op(stage) || last_use[stage_idx].empty()) {
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
                auto it = stage_by_node.find(link.node.get());
                if (it == stage_by_node.end()) {
                    continue;
                }
                const size_t producer_idx = it->second;
                if (link.port < last_use[producer_idx].size()) {
                    last_use[producer_idx][link.port] =
                        std::max(last_use[producer_idx][link.port], view_last_use);
                }
            }
        }

        std::vector<size_t> free_slots;
        free_slots.reserve(workspace->handles.size());
        for (size_t slot = 0; slot < workspace->handles.size(); ++slot) {
            free_slots.push_back(slot);
        }
        std::vector<ActiveSlot> active_slots;

        auto host_visible = [](const GpuBufferDesc& desc) {
            return desc.cpu_read || desc.cpu_write || !desc.prefer_device_local;
        };
        auto compatible = [&](const BufferHandle& handle, const GpuBufferDesc& desc) {
            return handle.valid() &&
                   handle.capacity >= desc.bytes &&
                   handle.buf.type == desc.type &&
                   handle.buf.host_visible == host_visible(desc);
        };
        auto acquire_slot = [&](const GpuBufferDesc& desc) {
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
                    if (!workspace->handles[slot].valid()) {
                        best_pos = pos;
                        break;
                    }
                }
            }
            if (best_pos != StageOutputBufferWorkspace::npos) {
                const size_t slot = free_slots[best_pos];
                free_slots.erase(free_slots.begin() + static_cast<std::ptrdiff_t>(best_pos));
                return slot;
            }
            workspace->handles.emplace_back();
            return workspace->handles.size() - 1;
        };
        auto update_peak_live = [&](size_t extra_internal_live = 0) {
            workspace->last_peak_live_slots =
                std::max(workspace->last_peak_live_slots, active_slots.size() + extra_internal_live);
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
            auto& stage_handles = handles[stage_idx];
            std::vector<GpuStageOutputLifetime> internal_lifetimes;
            const bool has_internal_lifetimes =
                stage.stage &&
                stage.stage->describe_output_lifetimes(internal_lifetimes) &&
                internal_lifetimes.size() >= stage.outputs.size();
            if (has_internal_lifetimes) {
                struct InternalOutput {
                    size_t output = 0;
                    size_t produced_at = 0;
                    size_t last_used_at = 0;
                    bool escapes_stage = false;
                };
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
                        ++workspace->last_legacy_outputs;
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
                                                last_use[stage_idx][oi] > stage_idx});
                }
                std::sort(internal_outputs.begin(),
                          internal_outputs.end(),
                          [](const InternalOutput& lhs, const InternalOutput& rhs) {
                              if (lhs.produced_at != rhs.produced_at) {
                                  return lhs.produced_at < rhs.produced_at;
                              }
                              return lhs.output < rhs.output;
                          });

                std::vector<ActiveSlot> internal_active_slots;
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
                        out_ref->buf = pool.ensure(workspace->handles[slot], plan.desc);
                    }
                    workspace->output_slots[stage_idx][output.output] = slot;
                    workspace->last_slots_used = std::max(workspace->last_slots_used, slot + 1);
                    ++workspace->last_workspace_outputs;
                    if (output.escapes_stage) {
                        active_slots.push_back({slot, last_use[stage_idx][output.output]});
                        update_peak_live();
                    } else {
                        internal_active_slots.push_back({slot, output.last_used_at});
                        update_peak_live(internal_active_slots.size());
                    }
                }
                for (const auto& active : internal_active_slots) {
                    free_slots.push_back(active.slot);
                }
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
                    ++workspace->last_legacy_outputs;
                    out_ref->buf = pool.ensure(stage_handles[oi], plan.desc);
                    continue;
                }
                const size_t slot = acquire_slot(plan.desc);
                if (plan.needs_buffer) {
                    out_ref->buf = pool.ensure(workspace->handles[slot], plan.desc);
                }
                workspace->output_slots[stage_idx][oi] = slot;
                workspace->last_slots_used = std::max(workspace->last_slots_used, slot + 1);
                ++workspace->last_workspace_outputs;
                active_slots.push_back({slot, last_use[stage_idx][oi]});
                update_peak_live();
            }
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
