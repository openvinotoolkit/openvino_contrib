// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include "infer_request.hpp"
#include "plugin.hpp"
#include "runtime/mlir_backend.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_memory.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

#include <cstdlib>
#include <string>

namespace ov {
namespace metal_plugin {

namespace {

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value)
        return false;
    const std::string lowered = ov::util::to_lower(std::string(value));
    return !(lowered == "0" || lowered == "false" || lowered == "off" || lowered == "no");
}

}  // namespace

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<const ov::Model>& original_model,
                             const ov::AnyMap& properties)
    : ov::ICompiledModel(model, plugin),
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

    // Force MLIR-only execution: disable MetalOp pipeline unconditionally.
    m_use_op_factory = false;

    // Preserve user properties; store inference_precision as ov::element::Type.
    for (const auto& kv : properties) {
        if (kv.first == ov::hint::inference_precision.name()) {
            m_config[kv.first] = m_inference_precision;
        } else if (kv.first == ov::enable_profiling.name() || kv.first == "PERF_COUNT") {
            m_config[kv.first] = m_enable_profiling;
        } else {
            m_config[kv.first] = kv.second;
        }
    }
    m_config["METAL_ENABLE_OP_FACTORY"] = m_use_op_factory;

    m_backend = std::make_unique<MlirBackend>(model, m_original_model, m_inference_precision);
    m_backend->set_profiling(m_enable_profiling);
    m_buffer_manager = m_backend->create_buffer_manager();
    if (metal_safe_debug_enabled()) {
        METAL_LOG_WARN("mlir", "[METAL SAFE_DEBUG] Backend created in safe-debug mode: GPU dispatch will be bypassed");
    }
    if (m_buffer_manager) {
        m_backend->preload_constants(*m_buffer_manager);
    }

    // MetalOp pipeline disabled; MLIR path only.
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(shared_from_this());
}

void CompiledModel::export_model(std::ostream& /*model*/) const {
    OPENVINO_THROW("METAL plugin export_model is not implemented yet");
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    for (const auto& kv : properties) {
        if (kv.first == ov::hint::inference_precision.name()) {
            m_inference_precision = kv.second.as<ov::element::Type>();
            m_config[kv.first] = m_inference_precision;
        } else if (kv.first == ov::enable_profiling.name()) {
            m_enable_profiling = kv.second.as<bool>();
            m_config[kv.first] = m_enable_profiling;
            if (m_backend) m_backend->set_profiling(m_enable_profiling);
        } else if (kv.first == "METAL_ENABLE_OP_FACTORY") {
            METAL_LOG_WARN("OpFactory", "METAL_ENABLE_OP_FACTORY ignored: MLIR-only mode enforced");
            m_use_op_factory = false;
            m_config[kv.first] = m_use_op_factory;
            m_pipeline.clear();
            m_node_to_stage.clear();
            m_pipeline_built = false;
        } else {
            m_config[kv.first] = kv.second;
        }
    }
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
        props.push_back(ov::PropertyName{"METAL_MEM_STATS", ov::PropertyMutability::RO});
        return decltype(ov::supported_properties)::value_type(props.begin(), props.end());
    }

    if (auto it = m_config.find(name); it != m_config.end()) {
        return it->second;
    }

    if (name == "METAL_MEM_STATS") {
        return m_buffer_manager ? ov::Any{m_buffer_manager->stats()} : ov::Any{MetalMemoryStats{}};
    }

    if (name == ov::hint::inference_precision.name()) {
        return m_inference_precision;
    }
    if (name == ov::enable_profiling.name()) {
        return m_enable_profiling;
    }

    OPENVINO_THROW("CompiledModel unsupported property: ", name);
}

void CompiledModel::build_op_pipeline() {
    // MLIR-only mode enforced: always skip MetalOp pipeline.
    METAL_LOG_DEBUG("OpFactory", "MLIR-only mode: skipping MetalOp pipeline build");
    m_pipeline.clear();
    m_node_to_stage.clear();
    m_pipeline_built = false;
    m_param_index.clear();
    m_use_op_factory = false;
    return;

    if (!m_runtime_model) {
        METAL_LOG_WARN("OpFactory", "Cannot build pipeline: runtime model is null");
        return;
    }

    if (!m_buffer_manager && m_backend) {
        m_buffer_manager = m_backend->create_buffer_manager();
    }
    if (!m_buffer_manager) {
        METAL_LOG_WARN("OpFactory", "Cannot build pipeline: buffer manager is null");
        return;
    }

    const auto device = m_buffer_manager->device();
    // Map Parameter nodes to input indices.
    for (size_t i = 0; i < m_runtime_model->inputs().size(); ++i) {
        m_param_index[m_runtime_model->inputs()[i].get_node()] = i;
    }

    bool all_supported = true;
    const auto ordered_ops = m_runtime_model->get_ordered_ops();
    m_pipeline.reserve(ordered_ops.size());

    for (const auto& node : ordered_ops) {
        if (ov::as_type_ptr<ov::op::v0::Parameter>(node) ||
            ov::as_type_ptr<ov::op::v0::Result>(node) ||
            ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            continue;
        }

        auto op = MetalOpFactory::create(node, device, /*queue*/ nullptr);
        if (!op) {
            METAL_LOG_DEBUG("OpFactory", "No MetalOp mapping for " << node->get_friendly_name()
                                                                  << " (" << node->get_type_name() << ")");
            all_supported = false;
            break;
        }

        PipelineStage stage;
        stage.node = node;
        stage.op = std::move(op);
        const size_t out_count = node->get_output_size();
        stage.outputs.reserve(out_count);
        for (size_t oi = 0; oi < out_count; ++oi) {
            auto out_tensor = std::make_unique<MetalTensor>();
            if (node->get_output_partial_shape(oi).is_static()) {
                out_tensor->shape = node->get_output_shape(oi);
            }
            out_tensor->expected_type = node->get_output_element_type(oi);
            stage.outputs.emplace_back(std::move(out_tensor));
        }
        for (const auto& iv : node->input_values()) {
            stage.inputs.push_back({iv.get_node_shared_ptr(), iv.get_index()});
        }

        // Bind outputs to op (single-output ops use first; multi-output ops handled by op subclass).
        if (stage.outputs.size() == 1) {
            stage.op->set_output(stage.outputs[0].get());
        } else {
            stage.op->set_outputs(stage.outputs);
        }

        const size_t idx = m_pipeline.size();
        m_node_to_stage[node.get()] = idx;
        m_pipeline.emplace_back(std::move(stage));
    }

    if (!all_supported) {
        m_pipeline.clear();
        m_node_to_stage.clear();
        m_pipeline_built = false;
        m_use_op_factory = false;
        METAL_LOG_INFO("OpFactory", "Pipeline disabled: unsupported op encountered");
        return;
    }

        for (auto& stage : m_pipeline) {
            std::vector<MetalTensor*> inputs;
            inputs.reserve(stage.node->get_input_size());
            for (const auto& link : stage.inputs) {
                auto source = link.node;
                auto it = m_node_to_stage.find(source.get());
                if (it != m_node_to_stage.end()) {
                    auto& src_stage = m_pipeline[it->second];
                    MetalTensor* tensor = nullptr;
                    if (link.port < src_stage.outputs.size()) {
                        tensor = src_stage.outputs[link.port].get();
                    }
                    inputs.push_back(tensor);
                } else {
                    inputs.push_back(nullptr);  // Parameter/Constant or unsupported op
                }
            }
            stage.op->set_inputs(inputs);
            stage.op->init(m_buffer_manager.get());
        }

    m_pipeline_built = true;
    m_use_op_factory = true;
    METAL_LOG_INFO("OpFactory", "Built MetalOp pipeline with " << m_pipeline.size() << " stages");
}

}  // namespace metal_plugin
}  // namespace ov
