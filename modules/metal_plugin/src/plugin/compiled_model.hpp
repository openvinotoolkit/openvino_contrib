// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/icompiled_model.hpp"

#include "graph/mps_graph_builder.hpp"

namespace ov {
namespace metal_plugin {

class Plugin;
class InferRequest;

class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const MPSGraphBuildResult& graph_info);

    std::shared_ptr<const ov::Model> get_runtime_model() const override { return m_runtime_model; }
    void export_model(std::ostream& model) const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    const std::shared_ptr<void>& graph() const { return m_graph_info.graph; }
    const std::vector<void*>& input_tensors() const { return m_graph_info.input_tensors; }
    const std::vector<void*>& output_tensors() const { return m_graph_info.output_tensors; }
    GraphLayout layout() const { return m_graph_info.internal_layout; }

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    MPSGraphBuildResult m_graph_info;
    std::shared_ptr<const ov::Model> m_runtime_model;
    ov::AnyMap m_config;
};

}  // namespace metal_plugin
}  // namespace ov
