// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include "infer_request.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace metal_plugin {

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const MPSGraphBuildResult& graph_info)
    : ov::ICompiledModel(model, plugin), m_graph_info(graph_info), m_runtime_model(model) {}

void CompiledModel::export_model(std::ostream& /*model*/) const {
    OPENVINO_THROW("METAL plugin export_model is not implemented yet");
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    for (const auto& kv : properties) {
        m_config[kv.first] = kv.second;
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    if (ov::supported_properties == name) {
        return std::vector<ov::PropertyName>{ov::supported_properties};
    }
    if (auto it = m_config.find(name); it != m_config.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported compiled model property: ", name);
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(shared_from_this());
}

}  // namespace metal_plugin
}  // namespace ov
