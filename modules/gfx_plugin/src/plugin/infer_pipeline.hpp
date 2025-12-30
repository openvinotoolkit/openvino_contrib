// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

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

bool is_view_op(const InferStage& stage);

std::vector<InferStage> build_infer_pipeline(const std::vector<PipelineStageDesc>& descs,
                                             GpuBufferManager* buffer_manager,
                                             void* profiler,
                                             bool profiling_enabled);

void bind_remote_outputs(const std::vector<ov::Output<const ov::Node>>& outputs,
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

struct OutputSource {
    std::shared_ptr<const ov::Node> node;
    size_t port = 0;
};

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
    const std::vector<ov::Output<const ov::Node>>& outputs,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_outputs,
    const std::vector<std::shared_ptr<GfxRemoteTensor>>& remote_inputs,
    GpuBackend expected_backend,
    const char* error_prefix = "GFX");

ov::Shape resolve_output_shape(const std::vector<ov::Output<const ov::Node>>& public_outputs,
                               const OutputSource& source,
                               const GpuTensor& tensor,
                               size_t out_idx,
                               bool allow_fallback_one);

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
                                   bool allow_fallback_one,
                                   const char* error_prefix);

template <typename DescribeOutput>
inline void allocate_stage_outputs(std::vector<InferStage>& pipeline,
                                   std::vector<std::vector<BufferHandle>>& handles,
                                   GpuBufferPool& pool,
                                   DescribeOutput&& describe_output,
                                   const char* error_prefix = "GFX") {
    if (handles.size() != pipeline.size()) {
        handles.assign(pipeline.size(), {});
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
                                   bool allow_missing,
                                   bool allow_fallback_one,
                                   const char* error_prefix = "GFX") {
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
                                        allow_fallback_one,
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
