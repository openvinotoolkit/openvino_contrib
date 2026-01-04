// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/compiled_model.hpp"

#include "openvino/gfx_plugin/infer_request.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "openvino/gfx_plugin/properties.hpp"
#include "openvino/gfx_plugin/plugin.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "plugin/compiled_model_backend_resources.hpp"
#include "plugin/backend_factory.hpp"
#include "plugin/gfx_property_lists.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "plugin/model_serialization.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gfx_batchnorm.hpp"
#include "runtime/gfx_bias.hpp"
#include "runtime/fused_sequence_stage.hpp"
#include "openvino/gfx_plugin/profiling.hpp"
#include "plugin/backend_state.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

#if ENABLE_GFX_MLIR
#include "transforms/fusion_pass.hpp"
#endif

#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
namespace ov {
namespace gfx_plugin {

namespace {

bool read_const_f32(const std::shared_ptr<const ov::op::v0::Constant>& constant,
                    std::vector<float>& out) {
    if (!constant) {
        return false;
    }
    const auto et = constant->get_element_type();
    const size_t count = shape_size(constant->get_shape());
    out.resize(count);
    if (count == 0) {
        return false;
    }
    if (et == ov::element::f32) {
        const float* src = constant->get_data_ptr<float>();
        std::copy(src, src + count, out.begin());
        return true;
    }
    if (et == ov::element::f16) {
        const ov::float16* src = constant->get_data_ptr<ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(src[i]);
        }
        return true;
    }
    return false;
}

bool extract_bias_params(const std::shared_ptr<const ov::Node>& node, BiasParams& out) {
    auto add = ov::as_type_ptr<const ov::op::v1::Add>(node);
    if (!add) {
        return false;
    }

    std::shared_ptr<const ov::op::v0::Constant> bias_const;
    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(add->get_input_node_shared_ptr(0))) {
        bias_const = c;
    } else if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(add->get_input_node_shared_ptr(1))) {
        bias_const = c;
    } else {
        return false;
    }

    BiasParams params{};
    if (!read_const_f32(bias_const, params.values)) {
        return false;
    }
    params.element_type = bias_const->get_element_type();
    params.shape.clear();
    const auto& shape = bias_const->get_shape();
    params.shape.reserve(shape.size());
    for (auto dim : shape) {
        params.shape.push_back(static_cast<int64_t>(dim));
    }
    if (params.values.empty()) {
        return false;
    }
    const size_t expected = shape_size(shape);
    if (expected != params.values.size()) {
        return false;
    }
    out = std::move(params);
    return true;
}

bool extract_batchnorm_params(const std::shared_ptr<const ov::Node>& node, BatchNormParams& out) {
    auto bn = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node);
    if (!bn) {
        return false;
    }

    auto gamma = std::dynamic_pointer_cast<const ov::op::v0::Constant>(bn->get_input_node_shared_ptr(1));
    auto beta = std::dynamic_pointer_cast<const ov::op::v0::Constant>(bn->get_input_node_shared_ptr(2));
    auto mean = std::dynamic_pointer_cast<const ov::op::v0::Constant>(bn->get_input_node_shared_ptr(3));
    auto var = std::dynamic_pointer_cast<const ov::op::v0::Constant>(bn->get_input_node_shared_ptr(4));
    if (!gamma || !beta || !mean || !var) {
        return false;
    }

    BatchNormParams params{};
    if (!read_const_f32(gamma, params.gamma) ||
        !read_const_f32(beta, params.beta) ||
        !read_const_f32(mean, params.mean) ||
        !read_const_f32(var, params.var)) {
        return false;
    }

    const size_t channels = params.gamma.size();
    if (channels == 0 ||
        params.beta.size() != channels ||
        params.mean.size() != channels ||
        params.var.size() != channels) {
        return false;
    }

    if (bn->get_input_partial_shape(0).rank().is_static()) {
        const auto& in_shape = bn->get_input_partial_shape(0);
        if (in_shape.rank().get_length() >= 2 && in_shape[1].is_static()) {
            const size_t expected = static_cast<size_t>(in_shape[1].get_length());
            if (expected != channels) {
                return false;
            }
        }
    }

    params.epsilon = static_cast<float>(bn->get_eps_value());
    out = std::move(params);
    return true;
}

}  // namespace

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<const ov::Model>& original_model,
                             const ov::AnyMap& properties,
                             const ov::SoPtr<ov::IRemoteContext>& context)
    : ov::ICompiledModel(model, plugin, context),
      m_runtime_model(model),
      m_original_model(original_model ? original_model : model) {
    // Initialize inference precision from user-supplied properties (default: f32).
    if (auto it = properties.find(ov::hint::inference_precision.name()); it != properties.end()) {
        m_inference_precision = it->second.as<ov::element::Type>();
    }
    if (auto it = properties.find(ov::enable_profiling.name()); it != properties.end()) {
        m_enable_profiling = parse_bool_property(it->second, ov::enable_profiling.name());
    } else {
        // Honour legacy PERF_COUNT=true if provided under a different key.
        if (auto it2 = properties.find("PERF_COUNT"); it2 != properties.end()) {
            m_enable_profiling = parse_bool_property(it2->second, "PERF_COUNT");
        }
    }
    if (auto it = properties.find(kGfxEnableFusionProperty); it != properties.end()) {
        m_enable_fusion = parse_bool_property(it->second, kGfxEnableFusionProperty);
    }
    if (auto it = properties.find(kGfxProfilingLevelProperty); it != properties.end()) {
        m_profiling_level = parse_profiling_level(it->second);
        m_profiling_level_set = true;
    }
    if (auto it = properties.find(ov::loaded_from_cache.name()); it != properties.end()) {
        m_loaded_from_cache = it->second.as<bool>();
    }
    ov::AnyMap resolved_props = properties;
    const auto request = get_backend_request(resolved_props);
    auto resolved = resolve_backend_for_properties(resolved_props, /*log_fallback=*/true, "CompiledModel");
    if (context) {
        auto gfx_ctx = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
        OPENVINO_ASSERT(gfx_ctx, "GFX: remote context type mismatch");
        const auto ctx_backend = gfx_ctx->backend();
        if (request.explicit_request && request.kind != ctx_backend) {
            OPENVINO_THROW("GFX: backend mismatch between properties (",
                           backend_to_string(request.kind),
                           ") and remote context (",
                           backend_to_string(ctx_backend),
                           ")");
        }
        resolved.backend = ctx_backend;
        resolved.backend_name = backend_to_string(ctx_backend);
        resolved_props[kGfxBackendProperty] = resolved.backend_name;
    }
    m_backend = resolved.backend;
    m_backend_name = resolved.backend_name;

    // Preserve user properties; store inference_precision as ov::element::Type.
    for (const auto& kv : resolved_props) {
        if (kv.first == ov::hint::inference_precision.name()) {
            m_config[kv.first] = m_inference_precision;
        } else if (kv.first == ov::enable_profiling.name() || kv.first == "PERF_COUNT") {
            m_config[kv.first] = m_enable_profiling;
        } else if (kv.first == kGfxProfilingLevelProperty) {
            m_config[kv.first] = kv.second;
        } else if (kv.first == kGfxEnableFusionProperty) {
            m_config[kv.first] = m_enable_fusion;
        } else {
            m_config[kv.first] = kv.second;
        }
    }
    m_config[kGfxBackendProperty] = m_backend_name;
    GFX_LOG_INFO("CompiledModel", "Creating backend state for " << m_backend_name);
    m_backend_state = create_backend_state(m_backend, properties, context);
    GFX_LOG_INFO("CompiledModel", "Backend state created");
    if (m_backend == GpuBackend::Vulkan) {
        GFX_LOG_INFO("CompiledModel", "Vulkan backend selected");
    }

    // Build GpuStage pipeline eagerly; fail early if unsupported ops encountered.
    GFX_LOG_INFO("CompiledModel", "Building stage pipeline");
    build_op_pipeline();
}

CompiledModel::~CompiledModel() {
    if (m_backend_state) {
        m_backend_state->release();
    }
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(shared_from_this());
}

void CompiledModel::export_model(std::ostream& model) const {
    const auto source = m_original_model ? m_original_model : m_runtime_model;
    write_model_to_stream(source, model);
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    for (const auto& kv : properties) {
        if (kv.first == ov::hint::inference_precision.name()) {
            m_inference_precision = kv.second.as<ov::element::Type>();
            m_config[kv.first] = m_inference_precision;
        } else if (apply_profiling_property(kv.first,
                                            kv.second,
                                            m_enable_profiling,
                                            m_profiling_level,
                                            m_profiling_level_set,
                                            m_config)) {
            // handled
        } else if (kv.first == kGfxEnableFusionProperty) {
            m_enable_fusion = parse_bool_property(kv.second, kv.first);
            m_config[kv.first] = m_enable_fusion;
        } else {
            OPENVINO_THROW("CompiledModel unsupported property: ", kv.first);
        }
    }
}

ProfilingLevel CompiledModel::profiling_level() const {
    if (!m_enable_profiling) {
        return ProfilingLevel::Off;
    }
    if (m_profiling_level_set) {
        return m_profiling_level;
    }
    return ProfilingLevel::Standard;
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    // Read-only properties we currently expose for integration with tools like benchmark_app
    if (ov::model_name == name) {
        return decltype(ov::model_name)::value_type{m_runtime_model->get_friendly_name()};
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_plugin()->get_device_name()};
    } else if (ov::optimal_number_of_infer_requests == name) {
        // Single-stream synchronous execution for now
        return decltype(ov::optimal_number_of_infer_requests)::value_type{1};
    } else if (ov::loaded_from_cache == name) {
        return decltype(ov::loaded_from_cache)::value_type{m_loaded_from_cache};
    } else if (ov::supported_properties == name) {
        auto props = gfx_compiled_model_supported_properties();
        return decltype(ov::supported_properties)::value_type(props.begin(), props.end());
    }

    if (name == kGfxBackendProperty) {
        return m_backend_name;
    }
    if (name == "PERF_COUNT") {
        return m_enable_profiling;
    }

    if (auto it = m_config.find(name); it != m_config.end()) {
        return it->second;
    }

    if (name == kGfxMemStatsProperty) {
        return m_backend_state ? m_backend_state->get_mem_stats() : ov::Any{};
    }
    if (name == kGfxProfilingReportProperty) {
        return last_profiling_report_json();
    }
    if (name == kGfxProfilingLevelProperty) {
        return static_cast<int>(profiling_level());
    }

    if (name == ov::hint::inference_precision.name()) {
        return m_inference_precision;
    }
    if (name == ov::enable_profiling.name()) {
        return m_enable_profiling;
    }

    OPENVINO_THROW("CompiledModel unsupported property: ", name);
}

void CompiledModel::update_last_profiling_report_json(std::string report_json) const {
    std::lock_guard<std::mutex> lock(m_report_mutex);
    m_last_report_json = std::move(report_json);
}

std::string CompiledModel::last_profiling_report_json() const {
    std::lock_guard<std::mutex> lock(m_report_mutex);
    return m_last_report_json;
}

void CompiledModel::build_op_pipeline() {
    if (!m_runtime_model) {
        GFX_LOG_WARN("OpFactory", "Cannot build pipeline: runtime model is null");
        return;
    }

    if (!backend_has_const_manager(backend_state())) {
        GFX_LOG_WARN("OpFactory", "Cannot build pipeline: const manager is null");
        return;
    }

    const auto resources = get_backend_resources(backend_state());
    auto* backend_state = m_backend_state.get();
    OPENVINO_ASSERT(backend_state, "GFX: backend state is not initialized");
    GFX_LOG_INFO("StageFactory",
                 "Building pipeline for backend=" << m_backend_name
                                                 << " ops=" << m_runtime_model->get_ops().size());
    // Map Parameter nodes to input indices.
    for (size_t i = 0; i < m_runtime_model->inputs().size(); ++i) {
        m_param_index[m_runtime_model->inputs()[i].get_node()] = i;
    }

    // Track model outputs for bookkeeping (outputs remain device-only).
    std::unordered_map<const ov::Node*, std::vector<bool>> model_outputs;
    for (const auto& result : m_runtime_model->get_results()) {
        auto src = result->input_value(0).get_node_shared_ptr();
        const size_t port = result->input_value(0).get_index();
        auto& flags = model_outputs[src.get()];
        if (flags.empty()) {
            flags.resize(src->get_output_size(), false);
        }
        if (port < flags.size()) {
            flags[port] = true;
        }
    }

    const auto ordered_ops = m_runtime_model->get_ordered_ops();
    GFX_LOG_INFO("StageFactory", "Ordered ops count=" << ordered_ops.size());
    m_pipeline.reserve(ordered_ops.size());

#if ENABLE_GFX_MLIR
    FusionPlan fusion_plan;
    std::unordered_map<size_t, const FusionGroup*> fusion_primary;
    if (m_enable_fusion) {
        FusionConfig fusion_cfg;
        fusion_cfg.enable_fusion = true;
        fusion_plan = build_fusion_plan(m_runtime_model, fusion_cfg);
        fusion_primary.reserve(fusion_plan.groups.size());
        for (const auto& group : fusion_plan.groups) {
            if (group.node_indices.size() < 2) {
                continue;
            }
            fusion_primary[group.node_indices.front()] = &group;
        }
        if (gfx_log_debug_enabled()) {
            GFX_LOG_DEBUG("Fusion", "Fusion enabled: groups=" << fusion_plan.groups.size());
        }
    } else if (gfx_log_debug_enabled()) {
        GFX_LOG_DEBUG("Fusion", "Fusion disabled via " << kGfxEnableFusionProperty);
    }
#endif

    std::unordered_set<size_t> fused_indices;
    fused_indices.reserve(ordered_ops.size());
    auto merge_model_outputs = [&](PipelineStageDesc& stage_desc, const ov::Node* node) {
        auto it = model_outputs.find(node);
        if (it == model_outputs.end()) {
            return;
        }
        const auto& flags = it->second;
        for (size_t oi = 0; oi < stage_desc.outputs.size() && oi < flags.size(); ++oi) {
            stage_desc.outputs[oi].is_model_output = stage_desc.outputs[oi].is_model_output || flags[oi];
        }
    };

    for (size_t op_index = 0; op_index < ordered_ops.size(); ++op_index) {
        const auto& node = ordered_ops[op_index];
        if (ov::as_type_ptr<ov::op::v0::Parameter>(node) ||
            ov::as_type_ptr<ov::op::v0::Result>(node) ||
            ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            continue;
        }

        if (fused_indices.count(op_index)) {
            continue;
        }

#if ENABLE_GFX_MLIR
        auto f_it = fusion_primary.find(op_index);
        if (f_it != fusion_primary.end() && f_it->second) {
            const auto* group = f_it->second;
            if (group->kind == "Attention" || group->kind == "AttentionScaleMask") {
                const size_t stage_count = group->node_indices.size();
                if (stage_count >= 3) {
                    struct InputKey {
                        const ov::Node* node = nullptr;
                        size_t port = 0;
                        bool operator==(const InputKey& other) const {
                            return node == other.node && port == other.port;
                        }
                    };
                    struct InputKeyHash {
                        size_t operator()(const InputKey& key) const {
                            size_t h1 = std::hash<const ov::Node*>()(key.node);
                            size_t h2 = std::hash<size_t>()(key.port);
                            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
                        }
                    };

                    std::unordered_map<const ov::Node*, size_t> stage_index;
                    stage_index.reserve(stage_count);
                    for (size_t i = 0; i < stage_count; ++i) {
                        const auto idx = group->node_indices[i];
                        if (idx < ordered_ops.size()) {
                            stage_index[ordered_ops[idx].get()] = i;
                        }
                    }

                    std::vector<size_t> stage_output_index(stage_count, 0);
                    for (size_t i = 0; i < stage_count; ++i) {
                        stage_output_index[i] = (i + 1);
                    }
                    stage_output_index[stage_count - 1] = 0;

                    std::unordered_map<InputKey, size_t, InputKeyHash> external_map;
                    std::vector<PipelineStageDesc::InputLink> fused_inputs;
                    std::vector<FusedStageInfo> fused_stages;
                    fused_stages.reserve(stage_count);

                    bool can_fuse = true;
                    for (size_t i = 0; i < stage_count; ++i) {
                        const size_t idx = group->node_indices[i];
                        if (idx >= ordered_ops.size()) {
                            can_fuse = false;
                            break;
                        }
                        const auto& stage_node = ordered_ops[idx];
                        auto stage = backend_state->create_stage(stage_node);
                        if (!stage) {
                            can_fuse = false;
                            break;
                        }

                        FusedStageInfo info;
                        info.stage = std::move(stage);
                        info.output_index = stage_output_index[i];
                        info.inputs.reserve(stage_node->get_input_size());
                        for (const auto& iv : stage_node->input_values()) {
                            auto src_node = iv.get_node();
                            const auto it_stage = stage_index.find(src_node);
                            if (it_stage != stage_index.end()) {
                                const size_t src_stage = it_stage->second;
                                info.inputs.push_back({FusedInputKind::Output,
                                                       stage_output_index[src_stage]});
                                continue;
                            }
                            if (ov::as_type_ptr<const ov::op::v0::Constant>(iv.get_node_shared_ptr())) {
                                info.inputs.push_back({FusedInputKind::None, 0});
                                continue;
                            }
                            InputKey key{src_node, iv.get_index()};
                            auto it_ext = external_map.find(key);
                            size_t ext_idx = 0;
                            if (it_ext == external_map.end()) {
                                ext_idx = fused_inputs.size();
                                external_map.emplace(key, ext_idx);
                                fused_inputs.push_back({iv.get_node_shared_ptr(), iv.get_index()});
                            } else {
                                ext_idx = it_ext->second;
                            }
                            info.inputs.push_back({FusedInputKind::External, ext_idx});
                        }
                        fused_stages.emplace_back(std::move(info));
                    }

                    if (can_fuse && fused_stages.size() == stage_count) {
                        PipelineStageDesc stage_desc;
                        const auto& final_node = ordered_ops[group->node_indices.back()];
                        stage_desc.node = final_node;
                        stage_desc.stage = std::make_unique<FusedSequenceStage>(
                            std::move(fused_stages),
                            final_node ? final_node->get_friendly_name() : std::string("fused_attention"),
                            "FusedAttention");
                        stage_desc.inputs = std::move(fused_inputs);
                        stage_desc.outputs.reserve(stage_count);

                        auto is_model_output = [&](const ov::Node* n) -> bool {
                            auto it = model_outputs.find(n);
                            if (it == model_outputs.end() || it->second.empty()) {
                                return false;
                            }
                            return it->second[0];
                        };

                        for (size_t out_idx = 0; out_idx < stage_count; ++out_idx) {
                            const ov::Node* out_node = nullptr;
                            if (out_idx == 0) {
                                out_node = final_node.get();
                            } else {
                                const size_t stage_idx = out_idx - 1;
                                const size_t node_idx = group->node_indices[stage_idx];
                                if (node_idx < ordered_ops.size()) {
                                    out_node = ordered_ops[node_idx].get();
                                }
                            }
                            OutputDesc out_desc{};
                            if (out_node && out_node->get_output_partial_shape(0).is_static()) {
                                out_desc.shape = out_node->get_output_shape(0);
                            }
                            if (out_node) {
                                out_desc.type = out_node->get_output_element_type(0);
                                out_desc.is_model_output = is_model_output(out_node);
                            }
                            stage_desc.outputs.emplace_back(std::move(out_desc));
                        }

                        const size_t idx = m_pipeline.size();
                        m_pipeline.emplace_back(std::move(stage_desc));
                        for (const auto node_idx : group->node_indices) {
                            if (node_idx < ordered_ops.size()) {
                                fused_indices.insert(node_idx);
                                const auto& fused_node = ordered_ops[node_idx];
                                m_node_to_stage[fused_node.get()] = idx;
                            }
                        }
                        continue;
                    }
                }
            }
        }
#endif

        if (gfx_log_debug_enabled()) {
            GFX_LOG_DEBUG("StageFactory",
                          "Preparing stage for " << node->get_type_name()
                                                 << " name=" << node->get_friendly_name());
        }
        auto gpu_stage = backend_state->create_stage(node);
        OPENVINO_ASSERT(gpu_stage,
                        "GFX: unsupported op in MLIR pipeline: ",
                        node->get_friendly_name(),
                        " (",
                        node->get_type_name(),
                        ")");

        PipelineStageDesc stage_desc;
        stage_desc.node = node;
        stage_desc.stage = std::move(gpu_stage);
        const size_t out_count = node->get_output_size();
        stage_desc.outputs.reserve(out_count);
        for (size_t oi = 0; oi < out_count; ++oi) {
            OutputDesc out_desc{};
            if (node->get_output_partial_shape(oi).is_static()) {
                out_desc.shape = node->get_output_shape(oi);
            }
            out_desc.type = node->get_output_element_type(oi);
            stage_desc.outputs.emplace_back(std::move(out_desc));
        }
        merge_model_outputs(stage_desc, node.get());
        for (const auto& iv : node->input_values()) {
            stage_desc.inputs.push_back({iv.get_node_shared_ptr(), iv.get_index()});
        }
        if (gfx_log_debug_enabled()) {
            GFX_LOG_DEBUG("StageFactory",
                          "Created GpuStage for " << node->get_type_name()
                                                  << " name=" << node->get_friendly_name());
        }

        const size_t idx = m_pipeline.size();
        m_node_to_stage[node.get()] = idx;
        m_pipeline.emplace_back(std::move(stage_desc));

#if ENABLE_GFX_MLIR
        auto f_it2 = fusion_primary.find(op_index);
        if (f_it2 != fusion_primary.end() && f_it2->second) {
            const auto* group = f_it2->second;
            auto& stage = m_pipeline[idx];
            auto mark_fused = [&](size_t fused_idx) {
                if (fused_idx >= ordered_ops.size()) {
                    return;
                }
                fused_indices.insert(fused_idx);
                const auto& fused_node = ordered_ops[fused_idx];
                m_node_to_stage[fused_node.get()] = idx;
                merge_model_outputs(stage, fused_node.get());
            };

            auto mark_fused_tail = [&](size_t start_idx) {
                for (size_t i = start_idx; i < group->node_indices.size(); ++i) {
                    mark_fused(group->node_indices[i]);
                }
            };
            if (group->kind == "ConvBatchNorm" || group->kind == "ConvBatchNormAct") {
                if (group->node_indices.size() >= 2) {
                    const size_t bn_idx = group->node_indices[1];
                    if (bn_idx < ordered_ops.size()) {
                        BatchNormParams bn_params{};
                        const bool bn_fused = extract_batchnorm_params(ordered_ops[bn_idx], bn_params) &&
                                              stage.stage->fuse_batchnorm(bn_params);
                        if (bn_fused) {
                            mark_fused(bn_idx);
                        }
                        if (bn_fused && group->kind == "ConvBatchNormAct" &&
                            group->activation.has_value() && group->node_indices.size() >= 3) {
                            if (stage.stage->fuse_activation(*group->activation, group->activation_alpha)) {
                                mark_fused_tail(2);
                            }
                        }
                    }
                }
            } else if (group->kind == "ConvBias" || group->kind == "ConvBiasActivation") {
                bool bias_fused = false;
                if (group->node_indices.size() >= 2) {
                    const size_t add_idx = group->node_indices[1];
                    if (add_idx < ordered_ops.size()) {
                        BiasParams bias_params{};
                        if (extract_bias_params(ordered_ops[add_idx], bias_params) &&
                            stage.stage->fuse_bias(bias_params)) {
                            bias_fused = true;
                            mark_fused(add_idx);
                        }
                    }
                }
                if (bias_fused && group->kind == "ConvBiasActivation" &&
                    group->activation.has_value() && group->node_indices.size() >= 3) {
                    if (stage.stage->fuse_activation(*group->activation, group->activation_alpha)) {
                        mark_fused_tail(2);
                    }
                }
            } else if (group->kind == "ConvActivation") {
                if (group->activation.has_value() && group->node_indices.size() >= 2) {
                    if (stage.stage->fuse_activation(*group->activation, group->activation_alpha)) {
                        mark_fused_tail(1);
                    }
                }
            } else if (group->kind == "EltwiseBias" || group->kind == "EltwiseBiasActivation") {
                bool bias_fused = false;
                if (group->node_indices.size() >= 2) {
                    const size_t add_idx = group->node_indices[1];
                    if (add_idx < ordered_ops.size()) {
                        BiasParams bias_params{};
                        if (extract_bias_params(ordered_ops[add_idx], bias_params) &&
                            stage.stage->fuse_bias(bias_params)) {
                            bias_fused = true;
                            mark_fused(add_idx);
                        }
                    }
                }
                if (bias_fused && group->kind == "EltwiseBiasActivation" &&
                    group->activation.has_value() && group->node_indices.size() >= 3) {
                    if (stage.stage->fuse_activation(*group->activation, group->activation_alpha)) {
                        mark_fused_tail(2);
                    }
                }
            } else if (group->kind == "EltwiseActivation") {
                if (group->activation.has_value() && group->node_indices.size() >= 2) {
                    if (stage.stage->fuse_activation(*group->activation, group->activation_alpha)) {
                        mark_fused_tail(1);
                    }
                }
            } else if (group->kind == "MatMulBias" || group->kind == "MatMulBiasActivation") {
                bool bias_fused = false;
                if (group->node_indices.size() >= 2) {
                    const size_t add_idx = group->node_indices[1];
                    if (add_idx < ordered_ops.size()) {
                        BiasParams bias_params{};
                        if (extract_bias_params(ordered_ops[add_idx], bias_params) &&
                            stage.stage->fuse_bias(bias_params)) {
                            bias_fused = true;
                            mark_fused(add_idx);
                        }
                    }
                }
                if (bias_fused && group->kind == "MatMulBiasActivation" &&
                    group->activation.has_value() && group->node_indices.size() >= 3) {
                    if (stage.stage->fuse_activation(*group->activation, group->activation_alpha)) {
                        mark_fused_tail(2);
                    }
                }
            } else if (group->kind == "MatMulActivation") {
                if (group->activation.has_value() && group->node_indices.size() >= 2) {
                    if (stage.stage->fuse_activation(*group->activation, group->activation_alpha)) {
                        mark_fused_tail(1);
                    }
                }
            }
        }
#endif
    }

    for (auto& stage : m_pipeline) {
        std::vector<GpuTensor*> inputs;
        inputs.reserve(stage.node->get_input_size());
        for (const auto& link : stage.inputs) {
            (void)link;
            inputs.push_back(nullptr);  // Parameter/Constant or previous stage; not needed for compile
        }
        if (gfx_log_debug_enabled()) {
            GFX_LOG_DEBUG("StageFactory",
                          "Compiling stage for " << stage.node->get_type_name()
                                                 << " name=" << stage.node->get_friendly_name());
        }
        stage.stage->set_inputs(inputs);
        GpuBufferManager* buffer_manager = resources.const_manager;
        stage.stage->init(buffer_manager);
        stage.stage->compile(buffer_manager);
    }

    m_pipeline_built = true;
    GFX_LOG_INFO("StageFactory",
                 "Built GFX " << m_backend_name << " pipeline with " << m_pipeline.size() << " stages");
}

}  // namespace gfx_plugin
}  // namespace ov
