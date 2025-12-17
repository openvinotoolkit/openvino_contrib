// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/core/type/element_type.hpp"

#include "runtime/backend.hpp"
#include "runtime/mlir_backend.hpp"
#include "runtime/metal_memory.hpp"
#include "runtime/metal_op.hpp"
#include "runtime/metal_op_factory.hpp"

namespace ov {
namespace metal_plugin {

class Plugin;
class InferRequest;
struct PipelineStage {
    std::shared_ptr<const ov::Node> node;
    std::unique_ptr<MetalOp> op;
    struct InputLink {
        std::shared_ptr<const ov::Node> node;
        size_t port = 0;
    };
    std::vector<std::unique_ptr<MetalTensor>> outputs;
    std::vector<InputLink> inputs;
};

class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const std::shared_ptr<const ov::Model>& original_model = nullptr,
                  const ov::AnyMap& properties = {});

    std::shared_ptr<const ov::Model> get_runtime_model() const override { return m_runtime_model; }
    void export_model(std::ostream& model) const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    MetalBackend* backend() const { return m_backend.get(); }
    ov::element::Type get_inference_precision() const { return m_inference_precision; }
    std::shared_ptr<MetalBufferManager> buffer_manager() const { return m_buffer_manager; }
    const MetalMemoryStats& memory_stats() const { return m_buffer_manager ? m_buffer_manager->stats() : m_dummy_stats; }
    size_t op_pipeline_size() const { return m_pipeline.size(); }
    bool op_pipeline_built() const { return m_pipeline_built; }
    bool op_pipeline_enabled() const { return m_use_op_factory; }
    const std::vector<PipelineStage>& pipeline() const { return m_pipeline; }
    std::vector<PipelineStage>& pipeline_mutable() const { return const_cast<std::vector<PipelineStage>&>(m_pipeline); }
    const std::unordered_map<const ov::Node*, size_t>& node_to_stage() const { return m_node_to_stage; }
    const std::unordered_map<const ov::Node*, size_t>& parameter_index() const { return m_param_index; }

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    void build_op_pipeline();

    MetalBackendPtr m_backend;
    std::shared_ptr<MetalBufferManager> m_buffer_manager;
    std::shared_ptr<const ov::Model> m_runtime_model;
    std::shared_ptr<const ov::Model> m_original_model;
    ov::AnyMap m_config;
    ov::element::Type m_inference_precision{ov::element::f32};
    bool m_enable_profiling = false;
    bool m_use_op_factory = true;
    bool m_pipeline_built = false;
    mutable std::vector<PipelineStage> m_pipeline;
    std::unordered_map<const ov::Node*, size_t> m_node_to_stage;
    std::unordered_map<const ov::Node*, size_t> m_param_index;
    MetalMemoryStats m_dummy_stats{};
};

}  // namespace metal_plugin
}  // namespace ov
