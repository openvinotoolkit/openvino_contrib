// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include "infer_request.hpp"
#include "plugin.hpp"
#include "runtime/mlir_backend.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace metal_plugin {

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

    // Preserve user properties; store inference_precision as ov::element::Type.
    for (const auto& kv : properties) {
        if (kv.first == ov::hint::inference_precision.name()) {
            m_config[kv.first] = m_inference_precision;
        } else {
            m_config[kv.first] = kv.second;
        }
    }

    m_backend = std::make_unique<MlirBackend>(model, m_original_model, m_inference_precision);
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
        return decltype(ov::supported_properties)::value_type(props.begin(), props.end());
    }

    if (auto it = m_config.find(name); it != m_config.end()) {
        return it->second;
    }

    if (name == ov::hint::inference_precision.name()) {
        return m_inference_precision;
    }

    OPENVINO_THROW("CompiledModel unsupported property: ", name);
}

}  // namespace metal_plugin
}  // namespace ov
