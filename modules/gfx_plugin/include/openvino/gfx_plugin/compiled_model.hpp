// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/core/type/element_type.hpp"

#include "runtime/gpu_stage.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/gfx_precision.hpp"
#include "openvino/gfx_plugin/profiling.hpp"

#include "openvino/gfx_plugin/properties.hpp"

namespace ov {
namespace gfx_plugin {

class Plugin;
class InferRequest;
class GfxProfilingTrace;
struct BackendState;
struct OutputDesc {
    ov::Shape shape;
    ov::element::Type type = ov::element::dynamic;
    bool is_model_output = false;
};

struct PipelineStageDesc {
    std::shared_ptr<const ov::Node> node;
    std::unique_ptr<GpuStage> stage;  // compiled prototype
    struct InputLink {
        std::shared_ptr<const ov::Node> node;
        size_t port = 0;
    };
    std::vector<OutputDesc> outputs;
    std::vector<InputLink> inputs;
};

class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const std::shared_ptr<const ov::Model>& original_model = nullptr,
                  const ov::AnyMap& properties = {},
                  const ov::SoPtr<ov::IRemoteContext>& context = {});
    ~CompiledModel() override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override { return m_runtime_model; }
    void export_model(std::ostream& model) const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    ov::element::Type get_inference_precision() const { return m_inference_precision; }
    GpuBackend backend() const { return m_backend; }
    const std::string& backend_name() const { return m_backend_name; }
    BackendState* backend_state() { return m_backend_state.get(); }
    const BackendState* backend_state() const { return m_backend_state.get(); }
    bool enable_profiling() const { return m_enable_profiling; }
    ProfilingLevel profiling_level() const;
    size_t op_pipeline_size() const { return m_pipeline.size(); }
    bool op_pipeline_built() const { return m_pipeline_built; }
    const std::vector<PipelineStageDesc>& pipeline_desc() const { return m_pipeline; }
    const std::unordered_map<const ov::Node*, size_t>& node_to_stage() const { return m_node_to_stage; }
    const std::unordered_map<const ov::Node*, size_t>& parameter_index() const { return m_param_index; }
    void update_last_profiling_report_json(std::string report_json) const;
    std::string last_profiling_report_json() const;
    void update_compile_profiling_report_json(std::string report_json) const;
    std::string compile_profiling_report_json() const;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    void build_op_pipeline(GfxProfilingTrace* compile_trace = nullptr);

    std::unique_ptr<BackendState> m_backend_state;
    std::shared_ptr<const ov::Model> m_runtime_model;
    std::shared_ptr<const ov::Model> m_original_model;
    ov::AnyMap m_config;
    GpuBackend m_backend = GpuBackend::Metal;
    std::string m_backend_name{"metal"};
    ov::element::Type m_inference_precision{gfx_default_inference_precision()};
    bool m_enable_profiling = false;
    ProfilingLevel m_profiling_level = ProfilingLevel::Standard;
    bool m_profiling_level_set = false;
    bool m_enable_fusion = true;
    bool m_pipeline_built = false;
    bool m_loaded_from_cache = false;
    mutable std::vector<PipelineStageDesc> m_pipeline;
    std::unordered_map<const ov::Node*, size_t> m_node_to_stage;
    std::unordered_map<const ov::Node*, size_t> m_param_index;
    mutable std::mutex m_report_mutex;
    mutable std::string m_last_report_json;
    mutable std::string m_compile_report_json;
};

}  // namespace gfx_plugin
}  // namespace ov
