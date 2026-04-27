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
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/fused_sequence_stage.hpp"
#include "openvino/gfx_plugin/profiling.hpp"
#include "plugin/backend_state.hpp"
#include "runtime/gfx_precision.hpp"
#include "runtime/gfx_profiling_report.hpp"
#include "runtime/gfx_vulkan_pipeline_cache_scope.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "openvino/util/common_util.hpp"

#include "transforms/fusion_pass.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
namespace ov {
namespace gfx_plugin {

namespace {

std::vector<int64_t> evaluate_constant_i64(const ov::Output<ov::Node>& value) {
    auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(value.get_node_shared_ptr());
    OPENVINO_ASSERT(constant, "GFX: expected constant input");
    return constant->cast_vector<int64_t>();
}

bool is_supported_absorbing_consumer(const std::shared_ptr<const ov::Node>& node) {
    return ov::is_type<ov::op::v1::Add>(node.get()) ||
           ov::is_type<ov::op::v1::GroupConvolution>(node.get());
}

bool is_supported_absorbing_input(const std::shared_ptr<const ov::Node>& node, size_t input_idx) {
    if (ov::is_type<ov::op::v1::Add>(node.get())) {
        return input_idx < 2;
    }
    if (ov::is_type<ov::op::v1::GroupConvolution>(node.get())) {
        return input_idx == 0;
    }
    return false;
}

bool is_absorbable_transpose_candidate(const std::shared_ptr<const ov::Node>& node,
                                       const std::unordered_map<const ov::Node*, std::vector<bool>>& model_outputs,
                                       const std::unordered_set<const ov::Node*>& fused_nodes) {
    auto transpose = ov::as_type_ptr<const ov::op::v1::Transpose>(node);
    if (!transpose || transpose->get_output_size() != 1) {
        return false;
    }
    if (fused_nodes.count(node.get()) != 0) {
        return false;
    }
    if (auto it = model_outputs.find(node.get()); it != model_outputs.end()) {
        if (std::any_of(it->second.begin(), it->second.end(), [](bool value) { return value; })) {
            return false;
        }
    }
    if (!transpose->get_input_partial_shape(0).is_static() ||
        !transpose->get_output_partial_shape(0).is_static()) {
        return false;
    }
    auto source = transpose->input_value(0).get_node_shared_ptr();
    if (!source || ov::is_type<ov::op::v0::Constant>(source.get())) {
        return false;
    }
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
    // GFX always computes in fp16 internally.
    if (auto it = properties.find(ov::hint::inference_precision.name()); it != properties.end()) {
        (void)it;
    }
    m_inference_precision = gfx_default_inference_precision();
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
    const bool capture_compile_profile = m_enable_profiling && profiling_level() != ProfilingLevel::Off;
    GfxProfilingTrace compile_trace;
    GfxProfilingTrace* compile_trace_ptr = capture_compile_profile ? &compile_trace : nullptr;
    const auto compile_wall_start =
        capture_compile_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (compile_trace_ptr) {
        compile_trace_ptr->reset(profiling_level());
        compile_trace_ptr->set_backend(m_backend_name);
        compile_trace_ptr->set_counter_capability(false, false);
        compile_trace_ptr->set_counter("loaded_from_cache", m_loaded_from_cache ? 1 : 0);
    }

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
    gfx_log_info("CompiledModel") << "Creating backend state for " << m_backend_name;
    const auto backend_state_start =
        compile_trace_ptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    m_backend_state = create_backend_state(m_backend, properties, context);
    if (compile_trace_ptr) {
        compile_trace_ptr->increment_counter("backend_state_create_count");
        compile_trace_ptr->add_segment(
            "compile",
            "create_backend_state",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - backend_state_start)
                                      .count()));
    }
    gfx_log_info("CompiledModel") << "Backend state created";
    if (m_backend == GpuBackend::Vulkan) {
        gfx_log_info("CompiledModel") << "Vulkan backend selected";
    }

    // Build GpuStage pipeline eagerly; fail early if unsupported ops encountered.
    gfx_log_info("CompiledModel") << "Building stage pipeline";
    const auto cache_dir_it = resolved_props.find(ov::cache_dir.name());
    const bool has_vulkan_pipeline_cache_dir =
        m_backend == GpuBackend::Vulkan && cache_dir_it != resolved_props.end() && cache_dir_it->second.is<std::string>() &&
        !cache_dir_it->second.as<std::string>().empty();
    if (has_vulkan_pipeline_cache_dir) {
        ScopedVulkanPipelineCacheDir cache_scope(cache_dir_it->second.as<std::string>());
        build_op_pipeline(compile_trace_ptr);
    } else {
        build_op_pipeline(compile_trace_ptr);
    }
    if (compile_trace_ptr) {
        uint64_t total_cpu_us = 0;
        for (const auto& segment : compile_trace_ptr->report().segments) {
            total_cpu_us += segment.cpu_us;
        }
        compile_trace_ptr->set_total_cpu_us(total_cpu_us);
        compile_trace_ptr->set_total_gpu_us(0);
        compile_trace_ptr->set_total_wall_us(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - compile_wall_start)
                                      .count()));
        const auto compile_json = compile_trace_ptr->to_json();
        update_compile_profiling_report_json(compile_json);
        update_last_profiling_report_json(
            build_profiling_report_json(m_backend_name, profiling_level(), {}, {}, compile_json));
    }
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
            m_inference_precision = gfx_default_inference_precision();
            m_config[kv.first] = m_inference_precision;
        } else if (apply_profiling_property(kv.first,
                                            kv.second,
                                            m_enable_profiling,
                                            m_profiling_level,
                                            m_profiling_level_set,
                                            m_config)) {
            // handled
        } else if (kv.first == ov::cache_dir.name()) {
            m_config[kv.first] = kv.second.as<std::string>();
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
    if (name == ov::cache_dir.name()) {
        if (auto it = m_config.find(name); it != m_config.end()) {
            return it->second;
        }
        return std::string{};
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

void CompiledModel::update_compile_profiling_report_json(std::string report_json) const {
    std::lock_guard<std::mutex> lock(m_report_mutex);
    m_compile_report_json = std::move(report_json);
}

std::string CompiledModel::compile_profiling_report_json() const {
    std::lock_guard<std::mutex> lock(m_report_mutex);
    return m_compile_report_json;
}

void CompiledModel::build_op_pipeline(GfxProfilingTrace* compile_trace) {
    if (!m_runtime_model) {
        gfx_log_warn("OpFactory") << "Cannot build pipeline: runtime model is null";
        return;
    }

    if (!backend_has_const_manager(backend_state())) {
        gfx_log_warn("OpFactory") << "Cannot build pipeline: const manager is null";
        return;
    }

    const auto resources = get_backend_resources(backend_state());
    auto* backend_state = m_backend_state.get();
    OPENVINO_ASSERT(backend_state, "GFX: backend state is not initialized");
    gfx_log_info("StageFactory") << "Building pipeline for backend=" << m_backend_name
                                                 << " ops=" << m_runtime_model->get_ops().size();
    const auto build_start =
        compile_trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (compile_trace) {
        compile_trace->set_counter("runtime_model_op_count", static_cast<uint64_t>(m_runtime_model->get_ops().size()));
    }
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
    gfx_log_info("StageFactory") << "Ordered ops count=" << ordered_ops.size();
    m_pipeline.reserve(ordered_ops.size());

    FusionPlan fusion_plan;
    std::unordered_map<size_t, const FusionGroup*> fusion_primary;
    std::unordered_set<size_t> planned_fused_indices;
    std::unordered_set<const ov::Node*> fused_nodes;
    if (m_enable_fusion) {
        FusionConfig fusion_cfg;
        fusion_cfg.enable_fusion = true;
        fusion_cfg.debug_dump_ir = gfx_log_debug_enabled();
        fusion_plan = build_fusion_plan(m_runtime_model, fusion_cfg);
        if (compile_trace) {
            compile_trace->set_counter("fusion_group_count", static_cast<uint64_t>(fusion_plan.groups.size()));
        }
        fusion_primary.reserve(fusion_plan.groups.size());
        for (const auto& group : fusion_plan.groups) {
            if (group.node_indices.size() < 2) {
                continue;
            }
            if (gfx_log_debug_enabled()) {
                std::string node_list;
                for (size_t i = 0; i < group.node_indices.size(); ++i) {
                    const auto node_idx = group.node_indices[i];
                    if (node_idx >= ordered_ops.size()) {
                        continue;
                    }
                    const auto& fused_node = ordered_ops[node_idx];
                    if (!node_list.empty()) {
                        node_list += " | ";
                    }
                    node_list += "[" + std::to_string(node_idx) + "] " + fused_node->get_friendly_name() + " (" +
                                 fused_node->get_type_name() + ")";
                }
                gfx_log_debug("Fusion") << "group kind=" << group.kind
                                        << " size=" << group.node_indices.size()
                                        << " nodes=" << node_list;
            }
            const bool attention_group = group.kind == "Attention" ||
                                         group.kind == "AttentionScale" ||
                                         group.kind == "AttentionScaleMask";
            const size_t primary_idx = attention_group ? group.node_indices.back()
                                                       : group.node_indices.front();
            fusion_primary[primary_idx] = &group;
            const bool pre_fusion_supported = [&]() {
                if (group.kind != "EltwiseInputActivation" ||
                    !group.input_activation.has_value() ||
                    primary_idx >= ordered_ops.size()) {
                    return false;
                }
                auto probe_stage = backend_state->create_stage(ordered_ops[primary_idx]);
                return probe_stage &&
                       probe_stage->fuse_input_activation(group.input_activation_input,
                                                          *group.input_activation,
                                                          group.input_activation_alpha);
            }();
            for (const auto node_idx : group.node_indices) {
                if (node_idx < ordered_ops.size()) {
                    fused_nodes.insert(ordered_ops[node_idx].get());
                    if ((attention_group || pre_fusion_supported) && node_idx != primary_idx) {
                        planned_fused_indices.insert(node_idx);
                    }
                }
            }
        }
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("Fusion") << "Fusion enabled: groups=" << fusion_plan.groups.size();
        }
    } else if (gfx_log_debug_enabled()) {
        gfx_log_debug("Fusion") << "Fusion disabled via " << kGfxEnableFusionProperty;
    }

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

    std::unordered_map<const ov::Node*, std::unordered_map<size_t, GfxInputTransform>> absorbed_input_transforms;
    std::unordered_set<const ov::Node*> absorbed_transpose_nodes;
    for (const auto& node : ordered_ops) {
        if (!is_absorbable_transpose_candidate(node, model_outputs, fused_nodes)) {
            continue;
        }
        OPENVINO_ASSERT(node->get_output_size() == 1, "GFX: transpose absorption expects single-output transpose");
        const auto& consumers = node->output(0).get_target_inputs();
        if (consumers.size() != 1) {
            continue;
        }
        const auto& consumer_input = *consumers.begin();
        auto consumer = consumer_input.get_node()->shared_from_this();
        if (!consumer || !is_supported_absorbing_consumer(consumer) ||
            !is_supported_absorbing_input(consumer, consumer_input.get_index()) ||
            fused_nodes.count(consumer.get()) != 0) {
            continue;
        }
        auto source = node->input_value(0).get_node_shared_ptr();
        if (!source || !node->input_value(0).get_partial_shape().is_static()) {
            continue;
        }
        auto permutation = evaluate_constant_i64(node->input_value(1));
        if (permutation.size() != node->get_input_shape(0).size()) {
            continue;
        }
        GfxInputTransform transform;
        transform.source_shape = node->get_input_shape(0);
        transform.transpose_permutation = std::move(permutation);
        absorbed_input_transforms[consumer.get()][consumer_input.get_index()] = std::move(transform);
        absorbed_transpose_nodes.insert(node.get());
    }

    for (size_t op_index = 0; op_index < ordered_ops.size(); ++op_index) {
        const auto& node = ordered_ops[op_index];
        if (ov::as_type_ptr<ov::op::v0::Parameter>(node) ||
            ov::as_type_ptr<ov::op::v0::Result>(node) ||
            ov::as_type_ptr<ov::op::v0::Constant>(node) ||
            ov::is_decompression(node)) {
            continue;
        }

        if (absorbed_transpose_nodes.count(node.get()) != 0) {
            continue;
        }

        if (fused_indices.count(op_index)) {
            continue;
        }
        if (planned_fused_indices.count(op_index) && fusion_primary.find(op_index) == fusion_primary.end()) {
            continue;
        }

        auto f_it = fusion_primary.find(op_index);
        if (f_it != fusion_primary.end() && f_it->second) {
            const auto* group = f_it->second;
            if (group->kind == "Attention" || group->kind == "AttentionScale" || group->kind == "AttentionScaleMask") {
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

                    bool can_fuse = true;
                    std::vector<std::vector<size_t>> stage_output_slots(stage_count);
                    size_t fused_output_count = 1;
                    for (size_t i = 0; i < stage_count; ++i) {
                        const size_t idx = group->node_indices[i];
                        if (idx >= ordered_ops.size()) {
                            can_fuse = false;
                            break;
                        }
                        const auto& stage_node = ordered_ops[idx];
                        stage_output_slots[i].reserve(stage_node->get_output_size());
                        for (size_t port = 0; port < stage_node->get_output_size(); ++port) {
                            if (i + 1 == stage_count && port == 0) {
                                stage_output_slots[i].push_back(0);
                            } else {
                                stage_output_slots[i].push_back(fused_output_count++);
                            }
                        }
                    }
                    if (!can_fuse) {
                        continue;
                    }

                    std::unordered_map<InputKey, size_t, InputKeyHash> external_map;
                    std::vector<PipelineStageDesc::InputLink> fused_inputs;
                    std::vector<FusedStageInfo> fused_stages;
                    fused_stages.reserve(stage_count);

                    for (size_t i = 0; i < stage_count; ++i) {
                        const size_t idx = group->node_indices[i];
                        if (idx >= ordered_ops.size()) {
                            can_fuse = false;
                            break;
                        }
                        const auto& stage_node = ordered_ops[idx];
                        auto stage = backend_state->create_stage(stage_node);
                        if (compile_trace) {
                            compile_trace->increment_counter("stage_create_count");
                        }
                        if (!stage) {
                            can_fuse = false;
                            break;
                        }

                        FusedStageInfo info;
                        info.stage = std::move(stage);
                        info.output_indices = stage_output_slots[i];
                        info.inputs.reserve(stage_node->get_input_size());
                        for (const auto& iv : stage_node->input_values()) {
                            auto src_node = iv.get_node();
                            const auto it_stage = stage_index.find(src_node);
                            if (it_stage != stage_index.end()) {
                                const size_t src_stage = it_stage->second;
                                if (iv.get_index() >= stage_output_slots[src_stage].size()) {
                                    can_fuse = false;
                                    break;
                                }
                                info.inputs.push_back(
                                    {FusedInputKind::Output, stage_output_slots[src_stage][iv.get_index()]});
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
                        if (!can_fuse) {
                            break;
                        }
                        fused_stages.emplace_back(std::move(info));
                    }

                    if (can_fuse && fused_stages.size() == stage_count) {
                        PipelineStageDesc stage_desc;
                        const auto& final_node = ordered_ops[group->node_indices.back()];
                        if (compile_trace) {
                            compile_trace->increment_counter("fused_stage_count");
                            compile_trace->increment_counter("fused_node_count", static_cast<uint64_t>(stage_count));
                        }
                        stage_desc.node = final_node;
                        stage_desc.stage = std::make_unique<FusedSequenceStage>(
                            std::move(fused_stages),
                            final_node ? final_node->get_friendly_name() : std::string("fused_attention"),
                            "FusedAttention");
                        stage_desc.inputs = std::move(fused_inputs);
                        stage_desc.outputs.resize(fused_output_count);

                        auto is_model_output = [&](const ov::Node* n, size_t port) -> bool {
                            auto it = model_outputs.find(n);
                            if (it == model_outputs.end() || it->second.empty()) {
                                return false;
                            }
                            return port < it->second.size() ? it->second[port] : false;
                        };

                        for (size_t stage_idx = 0; stage_idx < stage_count; ++stage_idx) {
                            const size_t node_idx = group->node_indices[stage_idx];
                            if (node_idx >= ordered_ops.size()) {
                                continue;
                            }
                            const auto& out_node = ordered_ops[node_idx];
                            for (size_t port = 0; port < out_node->get_output_size(); ++port) {
                                const size_t slot = stage_output_slots[stage_idx][port];
                                auto& out_desc = stage_desc.outputs[slot];
                                if (out_node->get_output_partial_shape(port).is_static()) {
                                    out_desc.shape = out_node->get_output_shape(port);
                                }
                                out_desc.type = out_node->get_output_element_type(port);
                                out_desc.is_model_output = is_model_output(out_node.get(), port);
                            }
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

        if (gfx_log_debug_enabled()) {
            gfx_log_debug("StageFactory") << "Preparing stage for " << node->get_type_name()
                                                 << " name=" << node->get_friendly_name();
        }
        auto gpu_stage = backend_state->create_stage(node);
        if (compile_trace) {
            compile_trace->increment_counter("stage_create_count");
        }
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
        const auto absorbed_it = absorbed_input_transforms.find(node.get());
        for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
            const auto& iv = node->input_value(input_idx);
            auto linked_node = iv.get_node_shared_ptr();
            size_t linked_port = iv.get_index();
            if (absorbed_it != absorbed_input_transforms.end()) {
                auto transform_it = absorbed_it->second.find(input_idx);
                if (transform_it != absorbed_it->second.end()) {
                    auto transpose = ov::as_type_ptr<const ov::op::v1::Transpose>(linked_node);
                    OPENVINO_ASSERT(transpose,
                                    "GFX: absorbed transpose input is not a transpose for ",
                                    node->get_friendly_name());
                    linked_node = transpose->input_value(0).get_node_shared_ptr();
                    linked_port = transpose->input_value(0).get_index();
                    stage_desc.stage->set_input_transform(input_idx, transform_it->second);
                }
            }
            stage_desc.inputs.push_back({linked_node, linked_port});
        }
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("StageFactory") << "Created GpuStage for " << node->get_type_name()
                                                  << " name=" << node->get_friendly_name();
        }

        const size_t idx = m_pipeline.size();
        m_node_to_stage[node.get()] = idx;
        m_pipeline.emplace_back(std::move(stage_desc));

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
            if (group->input_activation.has_value()) {
                const size_t input_idx = group->input_activation_input;
                bool input_activation_ok = false;
                if (group->node_indices.size() > 1 &&
                    input_idx < stage.inputs.size() &&
                    stage.stage->fuse_input_activation(input_idx,
                                                       *group->input_activation,
                                                       group->input_activation_alpha)) {
                    const size_t act_idx = group->node_indices[1];
                    if (act_idx < ordered_ops.size()) {
                        const auto& act_node = ordered_ops[act_idx];
                        if (stage.inputs[input_idx].node.get() == act_node.get() &&
                            act_node->get_input_size() == 1) {
                            stage.inputs[input_idx].node = act_node->input_value(0).get_node_shared_ptr();
                            stage.inputs[input_idx].port = act_node->input_value(0).get_index();
                            mark_fused(act_idx);
                            input_activation_ok = true;
                        }
                    }
                }
                if (!input_activation_ok && gfx_log_debug_enabled()) {
                    gfx_log_debug("Fusion") << "Failed input activation fusion for "
                                            << (node ? node->get_friendly_name() : std::string("<null>"));
                }
            }
            size_t next_post_op_idx = 1;
            bool post_ops_ok = true;
            if (group->batchnorm.has_value()) {
                if (group->node_indices.size() <= next_post_op_idx ||
                    !stage.stage->fuse_batchnorm(*group->batchnorm)) {
                    post_ops_ok = false;
                } else {
                    mark_fused(group->node_indices[next_post_op_idx]);
                    ++next_post_op_idx;
                }
            }
            if (post_ops_ok && group->bias.has_value()) {
                if (group->node_indices.size() <= next_post_op_idx ||
                    !stage.stage->fuse_bias(*group->bias)) {
                    post_ops_ok = false;
                } else {
                    mark_fused(group->node_indices[next_post_op_idx]);
                    ++next_post_op_idx;
                }
            }
            if (post_ops_ok && group->activation.has_value() &&
                group->node_indices.size() > next_post_op_idx) {
                if (stage.stage->fuse_activation(*group->activation, group->activation_alpha)) {
                    mark_fused_tail(next_post_op_idx);
                }
            }
        }
    }

    for (auto& stage : m_pipeline) {
        std::vector<GpuTensor*> inputs;
        inputs.reserve(stage.node->get_input_size());
        for (const auto& link : stage.inputs) {
            (void)link;
            inputs.push_back(nullptr);  // Parameter/Constant or previous stage; not needed for compile
        }
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("StageFactory") << "Compiling stage for " << stage.node->get_type_name()
                                                 << " name=" << stage.node->get_friendly_name();
        }
        stage.stage->set_inputs(inputs);
        GpuBufferManager* buffer_manager = resources.const_manager;
        stage.stage->init(buffer_manager);
        const auto stage_compile_start =
            compile_trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto compile_scope_name = stage.node ? stage.node->get_friendly_name() : stage.stage->name();
        ScopedCompileProfilingContext compile_scope(compile_trace, compile_scope_name);
        stage.stage->compile(buffer_manager);
        if (compile_trace) {
            compile_trace->increment_counter("stage_compile_count");
            compile_trace->add_segment(
                "compile",
                stage.node ? stage.node->get_friendly_name() : stage.stage->name(),
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - stage_compile_start)
                                          .count()));
        }
    }

    m_pipeline_built = true;
    if (compile_trace) {
        compile_trace->set_counter("pipeline_stage_count", static_cast<uint64_t>(m_pipeline.size()));
        compile_trace->add_segment(
            "compile",
            "build_op_pipeline",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - build_start)
                                      .count()));
    }
    gfx_log_info("StageFactory") << "Built GFX " << m_backend_name << " pipeline with " << m_pipeline.size() << " stages";
}

}  // namespace gfx_plugin
}  // namespace ov
