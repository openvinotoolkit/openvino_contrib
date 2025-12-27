// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include "infer_request.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "backends/metal/plugin/properties.hpp"
#include "plugin.hpp"
#include "remote_stub.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "backends/metal/runtime/memory.hpp"
#include "runtime/profiling/gfx_profiler_config.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

#include <cstdlib>
#include <string>
namespace ov {
namespace gfx_plugin {

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
        m_enable_profiling = it->second.as<bool>();
    } else {
        // Honour legacy PERF_COUNT=true if provided under a different key.
        if (auto it2 = properties.find("PERF_COUNT"); it2 != properties.end()) {
            m_enable_profiling = it2->second.as<bool>();
        }
    }
    if (auto it = properties.find(kGfxProfilingLevelProperty); it != properties.end()) {
        m_profiling_level = parse_profiling_level(it->second);
        m_profiling_level_set = true;
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
    if (!backend_supported(m_backend)) {
        OPENVINO_THROW("GFX ", backend_to_string(m_backend), " backend is not available in this build.");
    }
    m_backend_name = resolved.backend_name;

    // Preserve user properties; store inference_precision as ov::element::Type.
    for (const auto& kv : resolved_props) {
        if (kv.first == ov::hint::inference_precision.name()) {
            m_config[kv.first] = m_inference_precision;
        } else if (kv.first == ov::enable_profiling.name() || kv.first == "PERF_COUNT") {
            m_config[kv.first] = m_enable_profiling;
        } else if (kv.first == kGfxProfilingLevelProperty) {
            m_config[kv.first] = kv.second;
        } else {
            m_config[kv.first] = kv.second;
        }
    }
    m_config[kGfxBackendProperty] = m_backend_name;
    if (m_backend == GpuBackend::Metal) {
        // Create buffer manager bound to selected Metal device (context or DEVICE_ID).
        MetalDeviceHandle dev = nullptr;
        if (context) {
            auto metal_ctx = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
            OPENVINO_ASSERT(metal_ctx, "GFX: remote context type mismatch");
            OPENVINO_ASSERT(metal_ctx->backend() == GpuBackend::Metal,
                            "GFX: remote context backend mismatch (expected Metal)");
            dev = metal_ctx->device_handle();
        }
        if (!dev) {
            int device_id = parse_device_id(properties);
            dev = metal_get_device_by_id(device_id);
        }
        m_device = dev;
        m_command_queue = metal_create_command_queue(m_device);
        OPENVINO_ASSERT(m_command_queue, "GFX: failed to create command queue");
        m_caps = query_metal_device_caps(m_device);
        m_alloc_core = std::make_unique<MetalAllocatorCore>(m_device, m_caps);
        m_persistent_heaps = std::make_unique<MetalHeapPool>(*m_alloc_core);
        m_persistent_freelist = std::make_unique<MetalFreeList>();
        m_persistent_staging = std::make_unique<MetalStagingPool>(*m_alloc_core);
        m_persistent_alloc = std::make_unique<MetalAllocator>(*m_alloc_core,
                                                              *m_persistent_heaps,
                                                              *m_persistent_freelist,
                                                              *m_persistent_staging,
                                                              m_caps);
        m_const_cache = std::make_unique<MetalConstCache>(*m_persistent_alloc);
        m_const_manager = std::make_shared<MetalBufferManager>(*m_alloc_core, m_const_cache.get());
    } else {
        GFX_LOG_INFO("CompiledModel", "Vulkan backend selected");
    }

    // Build GpuStage pipeline eagerly; fail early if unsupported ops encountered.
    build_op_pipeline();
}

CompiledModel::~CompiledModel() {
    if (m_command_queue) {
        metal_release_command_queue(m_command_queue);
        m_command_queue = nullptr;
    }
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(shared_from_this());
}

void CompiledModel::export_model(std::ostream& /*model*/) const {
    OPENVINO_THROW("GFX plugin export_model is not implemented yet");
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    for (const auto& kv : properties) {
        if (kv.first == ov::hint::inference_precision.name()) {
            m_inference_precision = kv.second.as<ov::element::Type>();
            m_config[kv.first] = m_inference_precision;
        } else if (kv.first == ov::enable_profiling.name()) {
            m_enable_profiling = kv.second.as<bool>();
            m_config[kv.first] = m_enable_profiling;
        } else if (kv.first == "PERF_COUNT") {
            m_enable_profiling = kv.second.as<bool>();
            m_config[ov::enable_profiling.name()] = m_enable_profiling;
            m_config[kv.first] = kv.second;
        } else if (kv.first == kGfxProfilingLevelProperty) {
            m_profiling_level = parse_profiling_level(kv.second);
            m_profiling_level_set = true;
            m_config[kv.first] = kv.second;
        } else {
            m_config[kv.first] = kv.second;
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
    const auto default_ro_properties = []() {
        return std::vector<ov::PropertyName>{ov::model_name,
                                             ov::supported_properties,
                                             ov::execution_devices,
                                             ov::loaded_from_cache,
                                             ov::optimal_number_of_infer_requests};
    };

    if (ov::model_name == name) {
        return decltype(ov::model_name)::value_type{m_runtime_model->get_friendly_name()};
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_plugin()->get_device_name()};
    } else if (ov::optimal_number_of_infer_requests == name) {
        // Single-stream synchronous execution for now
        return decltype(ov::optimal_number_of_infer_requests)::value_type{1};
    } else if (ov::loaded_from_cache == name) {
        return decltype(ov::loaded_from_cache)::value_type{false};
    } else if (ov::supported_properties == name) {
        auto props = default_ro_properties();
        props.push_back(ov::PropertyName{ov::hint::inference_precision.name(), ov::PropertyMutability::RW});
        props.push_back(ov::PropertyName{ov::enable_profiling.name(), ov::PropertyMutability::RW});
        props.push_back(ov::PropertyName{"PERF_COUNT", ov::PropertyMutability::RW});
        props.push_back(ov::PropertyName{kGfxProfilingLevelProperty, ov::PropertyMutability::RW});
        props.push_back(ov::PropertyName{kGfxBackendProperty, ov::PropertyMutability::RO});
        props.push_back(ov::PropertyName{kGfxProfilingReportProperty, ov::PropertyMutability::RO});
        props.push_back(ov::PropertyName{kGfxMemStatsProperty, ov::PropertyMutability::RO});
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
        return ov::Any{m_last_stats};
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

void CompiledModel::update_last_profiling_report(const MetalProfilingReport& report) const {
    std::lock_guard<std::mutex> lock(m_report_mutex);
    m_last_report_json = report.to_json();
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

    if (!m_const_manager && m_backend == GpuBackend::Metal) {
        GFX_LOG_WARN("OpFactory", "Cannot build pipeline: const manager is null");
        return;
    }

    const auto device = m_device;
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
    m_pipeline.reserve(ordered_ops.size());

    const bool allow_conv_relu_fusion = false;  // TODO: re-enable after fused path validation.
    for (const auto& node : ordered_ops) {
        if (ov::as_type_ptr<ov::op::v0::Parameter>(node) ||
            ov::as_type_ptr<ov::op::v0::Result>(node) ||
            ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            continue;
        }

        if (allow_conv_relu_fusion) {
            if (auto relu = ov::as_type_ptr<const ov::op::v0::Relu>(node)) {
            auto input = relu->input_value(0).get_node_shared_ptr();
            auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(input);
            if (conv && input->output(0).get_target_inputs().size() == 1) {
                auto it = m_node_to_stage.find(input.get());
                if (it != m_node_to_stage.end()) {
                    auto& conv_stage = m_pipeline[it->second];
                    if (conv_stage.stage->fuse_activation(ActivationKind::Relu, 0.0f)) {
                        m_node_to_stage[node.get()] = it->second;
                        if (auto mo = model_outputs.find(node.get()); mo != model_outputs.end()) {
                            for (size_t oi = 0; oi < mo->second.size() && oi < conv_stage.outputs.size(); ++oi) {
                                conv_stage.outputs[oi].is_model_output =
                                    conv_stage.outputs[oi].is_model_output || mo->second[oi];
                            }
                        }
                        continue;  // skip creating a separate Relu stage
                    }
                }
            }
            }
        }

        auto gpu_stage = GpuStageFactory::create(node, m_backend, device, m_command_queue);
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
        if (auto it = model_outputs.find(node.get()); it != model_outputs.end()) {
            const auto& flags = it->second;
            for (size_t oi = 0; oi < out_count && oi < flags.size(); ++oi) {
                stage_desc.outputs[oi].is_model_output = flags[oi];
            }
        }
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
    }

    for (auto& stage : m_pipeline) {
        std::vector<GpuTensor*> inputs;
        inputs.reserve(stage.node->get_input_size());
        for (const auto& link : stage.inputs) {
            (void)link;
            inputs.push_back(nullptr);  // Parameter/Constant or previous stage; not needed for compile
        }
        stage.stage->set_inputs(inputs);
        stage.stage->init(m_const_manager.get());
        stage.stage->compile(m_const_manager.get());
    }

    m_pipeline_built = true;
    GFX_LOG_INFO("StageFactory",
                 "Built GFX " << m_backend_name << " pipeline with " << m_pipeline.size() << " stages");
}

}  // namespace gfx_plugin
}  // namespace ov
