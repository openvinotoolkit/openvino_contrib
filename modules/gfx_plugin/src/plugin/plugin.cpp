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

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "plugin/remote_context_support.hpp"
#include "plugin/gfx_backend_config.hpp"
#include "plugin/gfx_device_info.hpp"
#include "plugin/gfx_profiling_utils.hpp"
#include "plugin/gfx_property_lists.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "plugin/model_serialization.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "plugin/gfx_op_support.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_remote_context.hpp"
#include "transforms/pipeline.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::string make_compiled_runtime_properties(const ov::AnyMap& config) {
    const auto info = query_device_info_from_properties(config, /*log_fallback=*/false, "Plugin");
    std::ostringstream oss;
    oss << "backend=" << info.backend_name << ";device=" << info.device_name << ";id=" << info.device_id;
    return oss.str();
}

}  // namespace

Plugin::Plugin() {
    // Set device name exposed to Core
    m_device_name = "GFX";
    set_device_name(m_device_name);
    // Defaults for mutable properties expected by behavior tests
    m_config[ov::hint::num_requests.name()] = static_cast<uint32_t>(1);
    m_config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    m_config[ov::num_streams.name()] = ov::streams::Num(1);
    m_config[ov::inference_num_threads.name()] = static_cast<uint32_t>(1);
    m_config[ov::log::level.name()] = ov::log::Level::INFO;
    m_config[ov::hint::inference_precision.name()] = ov::element::f32;
    m_config[ov::internal::threads_per_stream.name()] = static_cast<uint32_t>(1);
    m_config[ov::internal::exclusive_async_requests.name()] = false;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) const {
    return compile_model_impl(model, properties, {});
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
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

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model_impl(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    OPENVINO_ASSERT(model, "Model is null");

    if (is_hetero_subgraph(model)) {
        OPENVINO_THROW("GFX plugin does not support HETERO subgraphs yet");
    }

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

    ov::AnyMap compile_properties = m_config;
    for (const auto& kv : properties) {
        compile_properties[kv.first] = kv.second;
    }
    const auto resolved = resolve_backend_for_properties(compile_properties, /*log_fallback=*/true, "Plugin");
    const auto backend_kind = resolved.backend;
    GFX_LOG_INFO("Plugin", "Selected GFX backend: " << resolved.backend_name);

    if (!model_supported_by_backend(transformed, backend_kind)) {
        auto summary = collect_unsupported(transformed, backend_kind);
        std::ostringstream oss;
        oss << "GFX: model contains unsupported ops for MLIR/GFX execution.";
        if (!summary.type_counts.empty()) {
            oss << " Types: ";
            size_t shown = 0;
            for (const auto& kv : summary.type_counts) {
                if (shown++)
                    oss << ", ";
                oss << kv.first << " x" << kv.second;
            }
        }
        if (!summary.node_names.empty()) {
            oss << ". Nodes: ";
            for (size_t i = 0; i < summary.node_names.size(); ++i) {
                if (i)
                    oss << ", ";
                oss << summary.node_names[i];
            }
        }
        OPENVINO_THROW(oss.str());
    }

    return std::make_shared<CompiledModel>(transformed,
                                           shared_from_this(),
                                           model,
                                           compile_properties,
                                           context);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                         const ov::AnyMap& properties) const {
    auto ov_model = read_model_from_stream(get_core_checked(), model);
    return import_model_impl(ov_model, properties, {});
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    auto ov_model = read_model_from_stream(get_core_checked(), model);
    return import_model_impl(ov_model, properties, context);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::AnyMap& properties) const {
    auto ov_model = read_model_from_buffer(get_core_checked(), model);
    return import_model_impl(ov_model, properties, {});
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    auto ov_model = read_model_from_buffer(get_core_checked(), model);
    return import_model_impl(ov_model, properties, context);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model_impl(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    if (context) {
        return compile_model(model, properties, context);
    }
    return compile_model(model, properties);
}

std::shared_ptr<ov::ICore> Plugin::get_core_checked() const {
    auto core = get_core();
    OPENVINO_ASSERT(core, "GFX: core is null");
    return core;
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OPENVINO_ASSERT(model, "Model is null");
    ov::AnyMap merged = m_config;
    for (const auto& kv : properties) {
        merged[kv.first] = kv.second;
    }
    const auto request = get_backend_request(merged);
    if (request.explicit_request && !backend_supported(request.kind)) {
        GFX_LOG_WARN("Plugin",
                     "query_model: requested backend '" << request.requested << "' is not supported");
        return {};
    }
    const auto backend_kind =
        resolve_backend_kind_from_properties(merged, /*log_fallback=*/false, "Plugin");
    if (!backend_supported(backend_kind)) {
        GFX_LOG_WARN("Plugin",
                     "query_model: backend '" << backend_to_string(backend_kind) << "' is not supported");
        return {};
    }
    ov::SupportedOpsMap res;
    // Use the same transformation pipeline as compile_model to keep support checks consistent.
    const auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);
    if (!model_supported_by_backend(transformed, backend_kind)) {
        // No partial fallback to CPU/HETERO: all-or-nothing support.
        return res;
    }
    for (const auto& node : model->get_ordered_ops()) {
        res.emplace(node->get_friendly_name(), get_device_name());
    }
    return res;
}

bool Plugin::is_hetero_subgraph(const std::shared_ptr<const ov::Model>& model) const {
    const auto fname = model->get_friendly_name();
    if (fname.find("Subgraph") != std::string::npos || fname.find("HETERO") != std::string::npos) {
        return true;
    }
    // Inspect rt_info keys for hetero markers
    const auto& rt = model->get_rt_info();
    for (const auto& kv : rt) {
        const auto& key = kv.first;
        if (key.find("HETERO") != std::string::npos || key.find("Subgraph") != std::string::npos) {
            return true;
        }
    }
    return false;
}

void Plugin::set_property(const ov::AnyMap& properties) {
    for (const auto& kv : properties) {
        if (apply_profiling_property(kv.first,
                                     kv.second,
                                     m_enable_profiling,
                                     m_profiling_level,
                                     m_profiling_level_set,
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
            } catch (const std::exception& e) {
                OPENVINO_THROW("Unsupported device id");
            }
        } else if (kv.first == kGfxBackendProperty) {
            ov::AnyMap tmp{{kGfxBackendProperty, kv.second}};
            const auto backend = resolve_backend_name_from_properties(tmp, /*log_fallback=*/true, "Plugin");
            m_config[kv.first] = backend;
        } else if (kv.first == ov::hint::inference_precision.name()) {
            m_config[kv.first] = kv.second.as<ov::element::Type>();
        } else if (kv.first == ov::internal::threads_per_stream.name()) {
            m_config[kv.first] = kv.second.as<uint32_t>();
        } else if (kv.first == ov::hint::num_requests.name() || kv.first == ov::hint::execution_mode.name() ||
                   kv.first == ov::num_streams.name() || kv.first == ov::inference_num_threads.name() ||
                   kv.first == ov::log::level.name() || kv.first == ov::internal::exclusive_async_requests.name()) {
            // Accepted but currently not acted upon; keep to satisfy behavior API expectations.
            m_config[kv.first] = kv.second;
        } else {
            OPENVINO_THROW("Unsupported property: ", kv.first);
        }
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    ov::AnyMap merged = m_config;
    for (const auto& kv : arguments) {
        merged[kv.first] = kv.second;
    }
    const auto device_info = [&]() {
        return query_device_info_from_properties(merged, /*log_fallback=*/false, "Plugin");
    };

    if (ov::supported_properties == name) {
        return gfx_plugin_supported_properties();
    } else if (ov::internal::supported_properties == name) {
        // Advertise internal properties this plugin understands (minimal set for dev flow)
        return decltype(ov::internal::supported_properties)::value_type(gfx_internal_supported_properties());
    } else if (ov::internal::caching_properties == name) {
        return decltype(ov::internal::caching_properties)::value_type(gfx_caching_properties());
    } else if (ov::internal::compiled_model_runtime_properties == name) {
        return make_compiled_runtime_properties(merged);
    } else if (ov::internal::compiled_model_runtime_properties_supported == name) {
        auto it = arguments.find(ov::internal::compiled_model_runtime_properties.name());
        if (it == arguments.end()) {
            return false;
        }
        const std::string expected = it->second.as<std::string>();
        return expected == make_compiled_runtime_properties(merged);
    } else if (ov::available_devices == name) {
        const auto info = device_info();
        if (info.available_devices.empty()) {
            return decltype(ov::available_devices)::value_type{{get_device_name()}};
        }
        return decltype(ov::available_devices)::value_type{info.available_devices.begin(),
                                                           info.available_devices.end()};
    } else if (ov::device::full_name == name) {
        const auto info = device_info();
        return decltype(ov::device::full_name)::value_type{info.full_name.empty() ? "GFX" : info.full_name};
    } else if (ov::device::architecture == name) {
        return decltype(ov::device::architecture)::value_type{"GFX"};
    } else if (ov::device::type == name) {
        const auto info = device_info();
        return decltype(ov::device::type)::value_type{info.device_type};
    } else if (ov::device::capabilities == name) {
        const auto info = device_info();
        return decltype(ov::device::capabilities)::value_type{info.capabilities.begin(),
                                                              info.capabilities.end()};
    } else if (ov::device::id == name) {
        const auto info = device_info();
        return decltype(ov::device::id)::value_type{info.device_id};
    } else if (ov::hint::performance_mode == name) {
        return m_performance_mode;
    } else if (ov::enable_profiling == name) {
        return m_enable_profiling;
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
        return resolve_backend_name_from_properties(merged, /*log_fallback=*/false, "Plugin");
    } else if (ov::internal::cache_header_alignment == name) {
        // Align cache header to 64 bytes to match Template expectations and CacheHeaderAlignmentTests.
        return decltype(ov::internal::cache_header_alignment)::value_type{64u};
    } else if (ov::range_for_async_infer_requests == name) {
        // min, max, step
        return decltype(ov::range_for_async_infer_requests)::value_type{1, 1, 1};
    } else if (ov::hint::inference_precision == name) {
        if (auto it = merged.find(ov::hint::inference_precision.name()); it != merged.end()) {
            return it->second.as<ov::element::Type>();
        }
        return ov::element::f32;
    }

    // Check user-provided properties preserved in m_config
    if (auto it = m_config.find(name); it != m_config.end()) {
        return it->second;
    }

    OPENVINO_THROW("Unsupported property: ", name);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& remote_properties) const {
    ov::AnyMap merged = m_config;
    for (const auto& kv : remote_properties) {
        merged[kv.first] = kv.second;
    }
    return make_gfx_remote_context(get_device_name(), merged);
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    return create_context(remote_properties);
}

}  // namespace gfx_plugin
}  // namespace ov

// Plugin entry point
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_gfx_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::gfx_plugin::Plugin, version)
