// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/plugin.hpp"

#include <algorithm>
#include <cctype>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/backend_config.hpp"
#include "common/gfx_backend_utils.hpp"
#include "compiler/cache_import.hpp"
#include "compiler/cache_repository.hpp"
#include "compiler/gfx_compiler_service.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "plugin/compiled_model.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin/compiled_model_cache_contract.hpp"
#include "plugin/gfx_device_info.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/gfx_property_lists.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "plugin/remote_context_support.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_precision.hpp"
#include "runtime/gfx_remote_context.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::string make_compiled_runtime_properties(const ov::AnyMap &config) {
  const auto info = query_device_info_from_properties(
      config, /*log_fallback=*/false, "Plugin");
  std::ostringstream oss;
  oss << "backend=" << info.backend_name << ";device=" << info.device_name
      << ";id=" << info.device_id;
  return oss.str();
}

std::string read_cache_wire(std::istream &stream) {
  std::ostringstream wire;
  wire << stream.rdbuf();
  return wire.str();
}

std::string diagnostics_to_string(const std::vector<std::string> &diagnostics) {
  std::ostringstream os;
  for (size_t i = 0; i < diagnostics.size(); ++i) {
    if (i != 0) {
      os << "; ";
    }
    os << diagnostics[i];
  }
  return os.str();
}

} // namespace

Plugin::Plugin() {
  // Set device name exposed to Core
  m_device_name = "GFX";
  set_device_name(m_device_name);
  // Defaults for mutable properties expected by behavior tests
  m_config[ov::hint::num_requests.name()] = static_cast<uint32_t>(1);
  m_config[ov::hint::execution_mode.name()] =
      ov::hint::ExecutionMode::PERFORMANCE;
  m_config[ov::num_streams.name()] = ov::streams::Num(1);
  m_config[ov::inference_num_threads.name()] = static_cast<uint32_t>(1);
  m_config[ov::log::level.name()] = ov::log::Level::INFO;
  m_config[ov::hint::inference_precision.name()] =
      gfx_default_inference_precision();
  m_config[ov::internal::threads_per_stream.name()] = static_cast<uint32_t>(1);
  m_config[ov::internal::exclusive_async_requests.name()] = false;
  m_config[kGfxEnableFusionProperty] = true;
  m_config[kGfxDiagnosticF32MpsImageProperty] = false;
}

std::shared_ptr<ov::ICompiledModel>
Plugin::compile_model(const std::shared_ptr<const ov::Model> &model,
                      const ov::AnyMap &properties) const {
  return compile_model_impl(model, properties, {});
}

std::shared_ptr<ov::ICompiledModel>
Plugin::compile_model(const std::shared_ptr<const ov::Model> &model,
                      const ov::AnyMap &properties,
                      const ov::SoPtr<ov::IRemoteContext> &context) const {
  if (!context) {
    return compile_model(model, properties);
  }
  ov::AnyMap merged = properties;
  const auto ctx_backend = ov::util::to_lower(get_remote_backend(context));
  if (auto it = merged.find(kGfxBackendProperty); it != merged.end()) {
    const auto requested = ov::util::to_lower(it->second.as<std::string>());
    if (requested != ctx_backend) {
      OPENVINO_THROW("GFX: backend mismatch between properties (", requested,
                     ") and remote context (", ctx_backend, ")");
    }
  }
  merged[kGfxBackendProperty] = ctx_backend;
  merged[ov::device::id.name()] = get_remote_device_id(context);
  OPENVINO_ASSERT(model, "Model is null");

  if (is_hetero_subgraph(model)) {
    OPENVINO_THROW("GFX plugin does not support HETERO subgraphs yet");
  }
  return compile_model_impl(model, merged, context);
}

std::shared_ptr<ov::ICompiledModel>
Plugin::compile_model_impl(const std::shared_ptr<const ov::Model> &model,
                           const ov::AnyMap &properties,
                           const ov::SoPtr<ov::IRemoteContext> &context) const {
  OPENVINO_ASSERT(model, "Model is null");

  if (is_hetero_subgraph(model)) {
    OPENVINO_THROW("GFX plugin does not support HETERO subgraphs yet");
  }

  ov::AnyMap compile_properties = m_config;
  for (const auto &kv : properties) {
    compile_properties[kv.first] = kv.second;
  }
  if (!compiled_model_cache_roundtrip_supported() &&
      compile_properties.count(ov::cache_dir.name()) != 0) {
    throw_compiled_model_cache_roundtrip_unavailable(
        "compile_model(cache_dir)");
  }
  const auto resolved = resolve_backend_for_properties(
      compile_properties, /*log_fallback=*/true, "Plugin");
  auto compile_target = resolved.target;
  if (context) {
    auto gfx_ctx = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
    OPENVINO_ASSERT(gfx_ctx, "GFX: remote context type mismatch");
    if (resolved.explicit_request &&
        resolved.backend != gfx_ctx->target().backend()) {
      OPENVINO_THROW("GFX: backend mismatch between properties (",
                     backend_to_string(resolved.backend),
                     ") and remote context (",
                     backend_to_string(gfx_ctx->target().backend()), ")");
    }
    compile_target = gfx_ctx->target();
  }
  gfx_log_info("Plugin") << "Selected GFX backend target: "
                         << compile_target.debug_string();
  bool enable_fusion = true;
  if (auto it = compile_properties.find(kGfxEnableFusionProperty);
      it != compile_properties.end()) {
    enable_fusion = parse_bool_property(it->second, kGfxEnableFusionProperty);
  }

  const auto &backend_registry = compiler::BackendRegistry::default_registry();
  const auto backend_module = backend_registry.resolve(compile_target);
  OPENVINO_ASSERT(backend_module,
                  "GFX: selected backend target is not registered: ",
                  compile_target.debug_string());

  std::string cache_dir;
  if (auto it = compile_properties.find(ov::cache_dir.name());
      it != compile_properties.end()) {
    cache_dir = it->second.as<std::string>();
  }
  compiler::ArtifactCacheRepository cache_repository(cache_dir);
  compiler::CacheLookupRequest cache_request;
  cache_request.model = model.get();
  cache_request.target = &compile_target;
  cache_request.capabilities = &backend_module->capabilities();
  cache_request.enable_fusion = enable_fusion;
  if (cache_repository.enabled()) {
    const auto cached = cache_repository.load(cache_request);
    if (cached.hit()) {
      gfx_log_info("Plugin") << "Loaded GFX compiled CacheEnvelope from cache";
      const auto import_contract = compiler::make_cache_import_contract(
          cached.envelope, backend_registry);
      if (!import_contract.valid()) {
        OPENVINO_THROW("GFX: cached CacheEnvelope cannot be imported: ",
                       diagnostics_to_string(import_contract.diagnostics));
      }
      ov::AnyMap cached_properties = compile_properties;
      cached_properties[ov::loaded_from_cache.name()] = true;
      cached_properties[kGfxBackendProperty] =
          import_contract.target.backend_id();
      return std::make_shared<CompiledModel>(
          import_contract.runtime_model, shared_from_this(),
          import_contract.executable, import_contract.runtime_descriptor,
          import_contract.target, cached.envelope, model, cached_properties,
          context);
    }
    if (cached.rejected()) {
      OPENVINO_THROW("GFX: rejected compiled cache entry: ",
                     diagnostics_to_string(cached.diagnostics));
    }
    gfx_log_info("Plugin") << "GFX compiled cache miss: "
                           << diagnostics_to_string(cached.diagnostics);
  }

  const compiler::GfxCompilerService compiler_service;
  compiler::GfxCompileRequest compile_request;
  compile_request.model = model;
  compile_request.target = compile_target;
  compile_request.backend_name = compile_target.backend_id();
  compile_request.enable_fusion = enable_fusion;
  const auto compile_result = compiler_service.compile(compile_request);
  if (!compile_result.supported()) {
    OPENVINO_THROW(compile_result.unsupported_message());
  }
  ov::AnyMap compiled_properties = compile_properties;
  compiled_properties[ov::loaded_from_cache.name()] = false;
  compiled_properties[kGfxBackendProperty] = compile_result.target.backend_id();
  if (cache_repository.enabled()) {
    const auto store_result =
        cache_repository.store(cache_request, compile_result.cache_envelope);
    if (!store_result.success) {
      gfx_log_warn("Plugin") << "Failed to store GFX compiled CacheEnvelope: "
                             << diagnostics_to_string(store_result.diagnostics);
    } else {
      gfx_log_info("Plugin")
          << "Stored GFX compiled CacheEnvelope at " << store_result.location;
    }
  }

  return std::make_shared<CompiledModel>(
      compile_result.transformed_model, shared_from_this(),
      compile_result.executable, compile_result.runtime_descriptor,
      compile_result.target, compile_result.cache_envelope, model,
      compiled_properties, context);
}

std::shared_ptr<ov::ICompiledModel>
Plugin::import_model(std::istream &model, const ov::AnyMap &properties) const {
  return import_model(model, {}, properties);
}

std::shared_ptr<ov::ICompiledModel>
Plugin::import_model(std::istream &model,
                     const ov::SoPtr<ov::IRemoteContext> &context,
                     const ov::AnyMap &properties) const {
  const auto wire = read_cache_wire(model);
  if (wire.empty()) {
    OPENVINO_THROW("GFX: import_model received an empty CacheEnvelope");
  }
  auto parsed = compiler::deserialize_cache_envelope(wire);
  if (!parsed.valid()) {
    OPENVINO_THROW("GFX: invalid compiled CacheEnvelope: ",
                   diagnostics_to_string(parsed.diagnostics));
  }
  auto import_contract = compiler::make_cache_import_contract(
      parsed.envelope, compiler::BackendRegistry::default_registry());
  if (!import_contract.valid()) {
    OPENVINO_THROW("GFX: unsupported compiled CacheEnvelope import: ",
                   diagnostics_to_string(import_contract.diagnostics));
  }

  ov::AnyMap import_properties = m_config;
  for (const auto &kv : properties) {
    import_properties[kv.first] = kv.second;
  }
  import_properties[ov::loaded_from_cache.name()] = true;
  import_properties[kGfxBackendProperty] = import_contract.target.backend_id();

  return std::make_shared<CompiledModel>(
      import_contract.runtime_model, shared_from_this(),
      import_contract.executable, import_contract.runtime_descriptor,
      import_contract.target, parsed.envelope, import_contract.runtime_model,
      import_properties, context);
}

std::shared_ptr<ov::ICompiledModel>
Plugin::import_model(const ov::Tensor &model,
                     const ov::AnyMap &properties) const {
  ov::SharedStreamBuffer buffer{model.data(), model.get_byte_size()};
  std::istream stream{&buffer};
  return import_model(stream, properties);
}

std::shared_ptr<ov::ICompiledModel>
Plugin::import_model(const ov::Tensor &model,
                     const ov::SoPtr<ov::IRemoteContext> &context,
                     const ov::AnyMap &properties) const {
  ov::SharedStreamBuffer buffer{model.data(), model.get_byte_size()};
  std::istream stream{&buffer};
  return import_model(stream, context, properties);
}

ov::SupportedOpsMap
Plugin::query_model(const std::shared_ptr<const ov::Model> &model,
                    const ov::AnyMap &properties) const {
  OPENVINO_ASSERT(model, "Model is null");
  ov::AnyMap merged = m_config;
  for (const auto &kv : properties) {
    merged[kv.first] = kv.second;
  }
  const auto request = get_backend_request(merged);
  if (request.explicit_request && !backend_supported(request.kind)) {
    gfx_log_warn("Plugin") << "query_model: requested backend '"
                           << request.requested << "' is not supported";
    return {};
  }
  const auto resolved =
      resolve_backend_for_properties(merged, /*log_fallback=*/false, "Plugin");
  ov::SupportedOpsMap res;
  const compiler::GfxCompilerService compiler_service;
  const auto compile_result =
      compiler_service.compile({model, resolved.target});
  if (!compile_result.supported()) {
    // No partial fallback to CPU/HETERO: all-or-nothing support.
    return res;
  }
  for (const auto &node : model->get_ordered_ops()) {
    res.emplace(node->get_friendly_name(), get_device_name());
  }
  return res;
}

bool Plugin::is_hetero_subgraph(
    const std::shared_ptr<const ov::Model> &model) const {
  const auto fname = model->get_friendly_name();
  if (fname.find("Subgraph") != std::string::npos ||
      fname.find("HETERO") != std::string::npos) {
    return true;
  }
  // Inspect rt_info keys for hetero markers
  const auto &rt = model->get_rt_info();
  for (const auto &kv : rt) {
    const auto &key = kv.first;
    if (key.find("HETERO") != std::string::npos ||
        key.find("Subgraph") != std::string::npos) {
      return true;
    }
  }
  return false;
}

void Plugin::set_property(const ov::AnyMap &properties) {
  for (const auto &kv : properties) {
    if (apply_profiling_property(kv.first, kv.second, m_enable_profiling,
                                 m_profiling_level, m_profiling_level_set,
                                 m_config)) {
      continue;
    }
    if (kv.first == ov::hint::performance_mode.name()) {
      m_performance_mode = kv.second.as<ov::hint::PerformanceMode>();
      m_config[kv.first] = kv.second;
    } else if (kv.first == ov::device::id.name()) {
      // Accept numeric IDs or empty; reject arbitrary strings
      try {
        // allow both string and integral form
        auto id_any = kv.second;
        if (id_any.is<std::string>()) {
          auto s = id_any.as<std::string>();
          if (!s.empty()) {
            (void)std::stoi(s);
          }
        } else {
          (void)id_any.as<int>();
        }
        m_config[kv.first] = kv.second;
      } catch (const std::exception &e) {
        OPENVINO_THROW("Unsupported device id");
      }
    } else if (kv.first == ov::cache_dir.name()) {
      if (!compiled_model_cache_roundtrip_supported()) {
        throw_compiled_model_cache_roundtrip_unavailable(
            "set_property(cache_dir)");
      }
      m_config[kv.first] = kv.second.as<std::string>();
    } else if (kv.first == kGfxBackendProperty) {
      ov::AnyMap tmp{{kGfxBackendProperty, kv.second}};
      const auto backend = resolve_backend_name_from_properties(
          tmp, /*log_fallback=*/true, "Plugin");
      m_config[kv.first] = backend;
    } else if (kv.first == kGfxEnableFusionProperty) {
      m_config[kv.first] = parse_bool_property(kv.second, kv.first);
    } else if (kv.first == kGfxDiagnosticF32MpsImageProperty) {
      m_config[kv.first] = parse_bool_property(kv.second, kv.first);
    } else if (kv.first == ov::hint::inference_precision.name()) {
      m_config[kv.first] =
          parse_inference_precision_property(kv.second, kv.first);
    } else if (kv.first == ov::internal::threads_per_stream.name()) {
      m_config[kv.first] = kv.second.as<uint32_t>();
    } else if (kv.first == ov::hint::num_requests.name() ||
               kv.first == ov::hint::execution_mode.name() ||
               kv.first == ov::num_streams.name() ||
               kv.first == ov::inference_num_threads.name() ||
               kv.first == ov::log::level.name() ||
               kv.first == ov::internal::exclusive_async_requests.name()) {
      // Accepted but currently not acted upon; keep to satisfy behavior API
      // expectations.
      m_config[kv.first] = kv.second;
    } else {
      OPENVINO_THROW("Unsupported property: ", kv.first);
    }
  }
}

ov::Any Plugin::get_property(const std::string &name,
                             const ov::AnyMap &arguments) const {
  ov::AnyMap merged = m_config;
  for (const auto &kv : arguments) {
    merged[kv.first] = kv.second;
  }
  const auto device_info = [&]() {
    return query_device_info_from_properties(merged, /*log_fallback=*/false,
                                             "Plugin");
  };

  if (ov::supported_properties == name) {
    return gfx_plugin_supported_properties();
  } else if (ov::internal::supported_properties == name) {
    // Advertise internal properties this plugin understands (minimal set for
    // dev flow)
    return decltype(ov::internal::supported_properties)::value_type(
        gfx_internal_supported_properties());
  } else if (ov::internal::caching_properties == name) {
    if (!compiled_model_cache_roundtrip_supported()) {
      OPENVINO_THROW("Unsupported property: ", name);
    }
    return decltype(ov::internal::caching_properties)::value_type(
        gfx_caching_properties());
  } else if (ov::internal::compiled_model_runtime_properties == name) {
    return make_compiled_runtime_properties(merged);
  } else if (ov::internal::compiled_model_runtime_properties_supported ==
             name) {
    auto it =
        arguments.find(ov::internal::compiled_model_runtime_properties.name());
    if (it == arguments.end()) {
      return false;
    }
    const std::string expected = it->second.as<std::string>();
    return expected == make_compiled_runtime_properties(merged);
  } else if (ov::available_devices == name) {
    const auto info = device_info();
    if (info.available_devices.empty()) {
      return decltype(ov::available_devices)::value_type{{""}};
    }
    return decltype(ov::available_devices)::value_type{
        info.available_devices.begin(), info.available_devices.end()};
  } else if (ov::device::full_name == name) {
    const auto info = device_info();
    return decltype(ov::device::full_name)::value_type{
        info.full_name.empty() ? "GFX" : info.full_name};
  } else if (ov::device::architecture == name) {
    return decltype(ov::device::architecture)::value_type{"GFX"};
  } else if (ov::device::type == name) {
    const auto info = device_info();
    return decltype(ov::device::type)::value_type{info.device_type};
  } else if (ov::device::capabilities == name) {
    const auto info = device_info();
    return decltype(ov::device::capabilities)::value_type{
        info.capabilities.begin(), info.capabilities.end()};
  } else if (ov::device::id == name) {
    const auto info = device_info();
    return decltype(ov::device::id)::value_type{info.device_id};
  } else if (ov::hint::performance_mode == name) {
    return m_performance_mode;
  } else if (ov::enable_profiling == name) {
    return m_enable_profiling;
  } else if (ov::cache_dir == name) {
    if (!compiled_model_cache_roundtrip_supported()) {
      OPENVINO_THROW("Unsupported property: ", name);
    }
    if (auto it = merged.find(ov::cache_dir.name()); it != merged.end()) {
      return it->second.as<std::string>();
    }
    return std::string{};
  } else if (name == kGfxProfilingLevelProperty) {
    if (m_profiling_level_set) {
      return static_cast<int>(m_profiling_level);
    }
    return static_cast<int>(ProfilingLevel::Standard);
  } else if (ov::execution_devices == name) {
    return decltype(ov::execution_devices)::value_type{get_device_name()};
  } else if (name == kGfxBackendProperty) {
    if (auto it = merged.find(kGfxBackendProperty); it != merged.end()) {
      return it->second;
    }
    return resolve_backend_name_from_properties(merged, /*log_fallback=*/false,
                                                "Plugin");
  } else if (ov::internal::cache_header_alignment == name) {
    if (!compiled_model_cache_roundtrip_supported()) {
      OPENVINO_THROW("Unsupported property: ", name);
    }
    // Align cache header to 64 bytes to match Template expectations and
    // CacheHeaderAlignmentTests.
    return decltype(ov::internal::cache_header_alignment)::value_type{64u};
  } else if (ov::range_for_async_infer_requests == name) {
    // min, max, step
    return decltype(ov::range_for_async_infer_requests)::value_type{1, 1, 1};
  } else if (ov::hint::inference_precision == name) {
    if (auto it = merged.find(ov::hint::inference_precision.name());
        it != merged.end()) {
      return it->second.as<ov::element::Type>();
    }
    return gfx_default_inference_precision();
  }

  // Check user-provided properties preserved in m_config
  if (auto it = m_config.find(name); it != m_config.end()) {
    return it->second;
  }

  OPENVINO_THROW("Unsupported property: ", name);
}

ov::SoPtr<ov::IRemoteContext>
Plugin::create_context(const ov::AnyMap &remote_properties) const {
  ov::AnyMap merged = m_config;
  for (const auto &kv : remote_properties) {
    merged[kv.first] = kv.second;
  }
  return make_gfx_remote_context(get_device_name(), merged);
}

ov::SoPtr<ov::IRemoteContext>
Plugin::get_default_context(const ov::AnyMap &remote_properties) const {
  return create_context(remote_properties);
}

} // namespace gfx_plugin
} // namespace ov

// Plugin entry point
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_gfx_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::gfx_plugin::Plugin, version)
