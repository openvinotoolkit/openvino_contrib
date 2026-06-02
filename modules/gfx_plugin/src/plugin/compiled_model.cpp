// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/compiled_model.hpp"

#include "openvino/gfx_plugin/infer_request.hpp"
#include "openvino/gfx_plugin/plugin.hpp"
#include "openvino/gfx_plugin/profiling.hpp"
#include "openvino/gfx_plugin/properties.hpp"
#include "plugin/backend_factory.hpp"
#include "plugin/backend_state.hpp"
#include "plugin/compiled_model_backend_resources.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/gfx_property_lists.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "plugin/model_serialization.hpp"
#include "compiler/pipeline_stage_builder.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_precision.hpp"
#include "runtime/gfx_profiling_report.hpp"
#include "runtime/gfx_remote_context.hpp"

#include "openvino/core/except.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
namespace ov {
namespace gfx_plugin {

CompiledModel::CompiledModel(
    const std::shared_ptr<const ov::Model> &model,
    const std::shared_ptr<const ov::IPlugin> &plugin,
    const compiler::ExecutableBundle &executable,
    const std::shared_ptr<const ov::Model> &original_model,
    const ov::AnyMap &properties, const ov::SoPtr<ov::IRemoteContext> &context)
    : ov::ICompiledModel(model, plugin, context), m_runtime_model(model),
      m_original_model(original_model ? original_model : model) {
  auto runtime_descriptor = std::make_shared<RuntimeExecutableDescriptor>(
      RuntimeExecutableDescriptorBuilder{}.build(executable));
  OPENVINO_ASSERT(
      runtime_descriptor->valid(executable),
      "GFX: compiler executable bundle does not match runtime descriptor");
  m_runtime_descriptor = std::move(runtime_descriptor);

  // GFX exposes f16 as the performance preference, while codegen preserves
  // f32 arithmetic for declared f32 and fp32-sensitive stages.
  if (auto it = properties.find(ov::hint::inference_precision.name());
      it != properties.end()) {
    m_inference_precision = parse_inference_precision_property(
        it->second, ov::hint::inference_precision.name());
  } else {
    m_inference_precision = gfx_default_inference_precision();
  }
  if (auto it = properties.find(ov::enable_profiling.name());
      it != properties.end()) {
    m_enable_profiling =
        parse_bool_property(it->second, ov::enable_profiling.name());
  } else {
    // Honour legacy PERF_COUNT=true if provided under a different key.
    if (auto it2 = properties.find("PERF_COUNT"); it2 != properties.end()) {
      m_enable_profiling = parse_bool_property(it2->second, "PERF_COUNT");
    }
  }
  if (auto it = properties.find(kGfxEnableFusionProperty);
      it != properties.end()) {
    m_enable_fusion = parse_bool_property(it->second, kGfxEnableFusionProperty);
  }
  for (const auto &kv : properties) {
    if (is_diagnostic_f32_vendor_image_property(kv.first)) {
      m_diagnostic_f32_vendor_image = parse_bool_property(kv.second, kv.first);
      break;
    }
  }
  if (auto it = properties.find(kGfxProfilingLevelProperty);
      it != properties.end()) {
    m_profiling_level = parse_profiling_level(it->second);
    m_profiling_level_set = true;
  }
  if (auto it = properties.find(ov::loaded_from_cache.name());
      it != properties.end()) {
    m_loaded_from_cache = it->second.as<bool>();
  }
  ov::AnyMap resolved_props = properties;
  const auto request = get_backend_request(resolved_props);
  auto resolved = resolve_backend_for_properties(
      resolved_props, /*log_fallback=*/true, "CompiledModel");
  if (context) {
    auto gfx_ctx = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
    OPENVINO_ASSERT(gfx_ctx, "GFX: remote context type mismatch");
    const auto ctx_backend = gfx_ctx->backend();
    if (request.explicit_request && request.kind != ctx_backend) {
      OPENVINO_THROW("GFX: backend mismatch between properties (",
                     backend_to_string(request.kind), ") and remote context (",
                     backend_to_string(ctx_backend), ")");
    }
    resolved.backend = ctx_backend;
    resolved.backend_name = backend_to_string(ctx_backend);
    resolved_props[kGfxBackendProperty] = resolved.backend_name;
  }
  m_backend = resolved.backend;
  m_backend_name = resolved.backend_name;
  register_backend_profiling_trace_sinks(m_backend);
  const bool capture_compile_profile =
      m_enable_profiling && profiling_level() != ProfilingLevel::Off;
  GfxProfilingTrace compile_trace;
  GfxProfilingTrace *compile_trace_ptr =
      capture_compile_profile ? &compile_trace : nullptr;
  const auto compile_wall_start = capture_compile_profile
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  if (compile_trace_ptr) {
    compile_trace_ptr->reset(profiling_level());
    compile_trace_ptr->set_backend(m_backend_name);
    compile_trace_ptr->set_counter_capability(false, false);
    compile_trace_ptr->set_counter("loaded_from_cache",
                                   m_loaded_from_cache ? 1 : 0);
  }

  // Preserve user properties; store inference_precision as ov::element::Type.
  for (const auto &kv : resolved_props) {
    if (kv.first == ov::hint::inference_precision.name()) {
      m_config[kv.first] = m_inference_precision;
    } else if (kv.first == ov::enable_profiling.name() ||
               kv.first == "PERF_COUNT") {
      m_config[kv.first] = m_enable_profiling;
    } else if (kv.first == kGfxProfilingLevelProperty) {
      m_config[kv.first] = kv.second;
    } else if (kv.first == kGfxEnableFusionProperty) {
      m_config[kv.first] = m_enable_fusion;
    } else if (is_diagnostic_f32_vendor_image_property(kv.first)) {
      m_config[kv.first] = m_diagnostic_f32_vendor_image;
    } else {
      m_config[kv.first] = kv.second;
    }
  }
  m_config[kGfxBackendProperty] = m_backend_name;
  gfx_log_info("CompiledModel")
      << "Creating backend state for " << m_backend_name;
  const auto backend_state_start =
      compile_trace_ptr ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
  m_backend_state = create_backend_state(m_backend, properties, context);
  if (compile_trace_ptr) {
    compile_trace_ptr->increment_counter("backend_state_create_count");
    compile_trace_ptr->add_segment(
        "compile", "create_backend_state",
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - backend_state_start)
                .count()));
  }
  gfx_log_info("CompiledModel") << "Backend state created";
  // Build GpuStage pipeline eagerly; fail early if unsupported ops encountered.
  gfx_log_info("CompiledModel") << "Building stage pipeline";
  build_op_pipeline(compile_trace_ptr);
  if (compile_trace_ptr) {
    uint64_t total_cpu_us = 0;
    for (const auto &segment : compile_trace_ptr->report().segments) {
      total_cpu_us += segment.cpu_us;
    }
    compile_trace_ptr->set_total_cpu_us(total_cpu_us);
    compile_trace_ptr->set_total_gpu_us(0);
    compile_trace_ptr->set_total_wall_us(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - compile_wall_start)
            .count()));
    const auto compile_json = compile_trace_ptr->to_json();
    update_compile_profiling_report_json(compile_json);
    update_last_profiling_report_json(build_profiling_report_json(
        m_backend_name, profiling_level(), {}, {}, compile_json));
  }
}

CompiledModel::~CompiledModel() {
  if (m_backend_state) {
    m_backend_state->release();
  }
}

std::shared_ptr<ov::ISyncInferRequest>
CompiledModel::create_sync_infer_request() const {
  return std::make_shared<InferRequest>(shared_from_this());
}

void CompiledModel::export_model(std::ostream &model) const {
  const auto source = m_original_model ? m_original_model : m_runtime_model;
  write_model_to_stream(source, model);
}

void CompiledModel::set_property(const ov::AnyMap &properties) {
  for (const auto &kv : properties) {
    if (kv.first == ov::hint::inference_precision.name()) {
      m_inference_precision =
          parse_inference_precision_property(kv.second, kv.first);
      m_config[kv.first] = m_inference_precision;
    } else if (apply_profiling_property(kv.first, kv.second, m_enable_profiling,
                                        m_profiling_level,
                                        m_profiling_level_set, m_config)) {
      // handled
    } else if (kv.first == ov::cache_dir.name()) {
      m_config[kv.first] = kv.second.as<std::string>();
    } else if (kv.first == kGfxEnableFusionProperty) {
      m_enable_fusion = parse_bool_property(kv.second, kv.first);
      m_config[kv.first] = m_enable_fusion;
    } else if (is_diagnostic_f32_vendor_image_property(kv.first)) {
      m_diagnostic_f32_vendor_image = parse_bool_property(kv.second, kv.first);
      m_config[kv.first] = m_diagnostic_f32_vendor_image;
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

ov::Any CompiledModel::get_property(const std::string &name) const {
  // Read-only properties we currently expose for integration with tools like
  // benchmark_app
  if (ov::model_name == name) {
    return decltype(ov::model_name)::value_type{
        m_runtime_model->get_friendly_name()};
  } else if (ov::execution_devices == name) {
    return decltype(ov::execution_devices)::value_type{
        get_plugin()->get_device_name()};
  } else if (ov::optimal_number_of_infer_requests == name) {
    // Single-stream synchronous execution for now
    return decltype(ov::optimal_number_of_infer_requests)::value_type{1};
  } else if (ov::loaded_from_cache == name) {
    return decltype(ov::loaded_from_cache)::value_type{m_loaded_from_cache};
  } else if (ov::supported_properties == name) {
    auto props = gfx_compiled_model_supported_properties();
    return decltype(ov::supported_properties)::value_type(props.begin(),
                                                          props.end());
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

void CompiledModel::update_last_profiling_report_json(
    std::string report_json) const {
  std::lock_guard<std::mutex> lock(m_report_mutex);
  m_last_report_json = std::move(report_json);
}

std::string CompiledModel::last_profiling_report_json() const {
  std::lock_guard<std::mutex> lock(m_report_mutex);
  return m_last_report_json;
}

void CompiledModel::update_compile_profiling_report_json(
    std::string report_json) const {
  std::lock_guard<std::mutex> lock(m_report_mutex);
  m_compile_report_json = std::move(report_json);
}

std::string CompiledModel::compile_profiling_report_json() const {
  std::lock_guard<std::mutex> lock(m_report_mutex);
  return m_compile_report_json;
}

void CompiledModel::build_op_pipeline(GfxProfilingTrace *compile_trace) {
  if (!m_runtime_model) {
    gfx_log_warn("OpFactory") << "Cannot build pipeline: runtime model is null";
    return;
  }

  if (!backend_has_const_manager(backend_state())) {
    gfx_log_warn("OpFactory") << "Cannot build pipeline: const manager is null";
    return;
  }

  auto *backend_state = m_backend_state.get();
  OPENVINO_ASSERT(backend_state, "GFX: backend state is not initialized");
  OPENVINO_ASSERT(m_runtime_descriptor,
                  "GFX: compiled model is missing compiler-owned runtime "
                  "executable descriptor");
  compiler::PipelineStageBuildRequest request;
  request.runtime_model = m_runtime_model;
  request.stage_factory = backend_state;
  request.runtime_descriptor = m_runtime_descriptor.get();
  request.backend = m_backend;
  request.backend_name = m_backend_name;
  request.enable_fusion = m_enable_fusion;
  request.diagnostic_f32_vendor_image = m_diagnostic_f32_vendor_image;
  request.compile_trace = compile_trace;

  auto result = compiler::build_pipeline_stage_descriptors(request);
  m_pipeline = std::move(result.pipeline);
  m_node_to_stage = std::move(result.node_to_stage);
  m_param_index = std::move(result.param_index);
  m_pipeline_built = true;
}

} // namespace gfx_plugin
} // namespace ov
