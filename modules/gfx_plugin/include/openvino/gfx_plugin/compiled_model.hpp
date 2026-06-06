// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "openvino/runtime/icompiled_model.hpp"

#include "common/gpu_backend.hpp"
#include "compiler/backend_target.hpp"
#include "openvino/gfx_plugin/profiling.hpp"
#include "runtime/gfx_precision.hpp"
#include "runtime/pipeline_stage_desc.hpp"

#include "openvino/gfx_plugin/properties.hpp"

namespace ov {
namespace gfx_plugin {

namespace compiler {
struct ExecutableBundle;
}  // namespace compiler

class Plugin;
class InferRequest;
class GfxProfilingTrace;
struct BackendState;
struct RuntimeExecutableDescriptor;

class CompiledModel : public ov::ICompiledModel {
public:
  CompiledModel(
      const std::shared_ptr<const ov::Model> &model,
      const std::shared_ptr<const ov::IPlugin> &plugin,
      const compiler::ExecutableBundle &executable,
      std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor,
      const compiler::BackendTarget &target,
      const std::shared_ptr<const ov::Model> &original_model = nullptr,
      const ov::AnyMap &properties = {},
      const ov::SoPtr<ov::IRemoteContext> &context = {});
  ~CompiledModel() override;

  std::shared_ptr<const ov::Model> get_runtime_model() const override {
    return m_runtime_model;
  }
  void export_model(std::ostream &model) const override;

  void set_property(const ov::AnyMap &properties) override;
  ov::Any get_property(const std::string &name) const override;

  ov::element::Type get_inference_precision() const {
    return m_inference_precision;
  }
  const compiler::BackendTarget &target() const { return m_target; }
  GpuBackend backend() const { return m_backend; }
  const std::string &backend_name() const { return m_backend_name; }
  BackendState *backend_state() { return m_backend_state.get(); }
  const BackendState *backend_state() const { return m_backend_state.get(); }
  bool enable_profiling() const { return m_enable_profiling; }
  ProfilingLevel profiling_level() const;
  size_t op_pipeline_size() const { return m_pipeline.size(); }
  bool op_pipeline_built() const { return m_pipeline_built; }
  const std::vector<PipelineStageDesc> &pipeline_desc() const {
    return m_pipeline;
  }
  std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor() const {
    return m_runtime_descriptor;
  }
  void update_last_profiling_report_json(std::string report_json) const;
  std::string last_profiling_report_json() const;
  void update_compile_profiling_report_json(std::string report_json) const;
  std::string compile_profiling_report_json() const;

protected:
  std::shared_ptr<ov::ISyncInferRequest>
  create_sync_infer_request() const override;

private:
  void build_op_pipeline(GfxProfilingTrace *compile_trace = nullptr);

  std::unique_ptr<BackendState> m_backend_state;
  std::shared_ptr<const ov::Model> m_runtime_model;
  std::shared_ptr<const ov::Model> m_original_model;
  ov::AnyMap m_config;
  compiler::BackendTarget m_target;
  GpuBackend m_backend = GpuBackend::Unknown;
  std::string m_backend_name;
  ov::element::Type m_inference_precision{gfx_default_inference_precision()};
  bool m_enable_profiling = false;
  ProfilingLevel m_profiling_level = ProfilingLevel::Standard;
  bool m_profiling_level_set = false;
  bool m_enable_fusion = true;
  bool m_diagnostic_f32_vendor_image = false;
  bool m_pipeline_built = false;
  bool m_loaded_from_cache = false;
  std::shared_ptr<const RuntimeExecutableDescriptor> m_runtime_descriptor;
  mutable std::vector<PipelineStageDesc> m_pipeline;
  mutable std::mutex m_report_mutex;
  mutable std::string m_last_report_json;
  mutable std::string m_compile_report_json;
};

} // namespace gfx_plugin
} // namespace ov
