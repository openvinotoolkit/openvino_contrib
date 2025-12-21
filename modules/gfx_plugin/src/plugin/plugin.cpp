// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <algorithm>
#include <cctype>
#include <istream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "runtime/metal_op_factory.hpp"
#include "runtime/metal_memory.hpp"
#include "runtime/metal_logger.hpp"
#include "remote_stub.hpp"
#include "transforms/pipeline.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

constexpr const char* kBackendProperty = "GFX_BACKEND";

ProfilingLevel parse_profiling_level(const ov::Any& value) {
    if (value.is<int>()) {
        const int v = value.as<int>();
        if (v <= 0)
            return ProfilingLevel::Off;
        if (v == 1)
            return ProfilingLevel::Standard;
        return ProfilingLevel::Detailed;
    }
    if (value.is<unsigned int>()) {
        const unsigned int v = value.as<unsigned int>();
        if (v == 0)
            return ProfilingLevel::Off;
        if (v == 1)
            return ProfilingLevel::Standard;
        return ProfilingLevel::Detailed;
    }
    if (value.is<bool>()) {
        return value.as<bool>() ? ProfilingLevel::Standard : ProfilingLevel::Off;
    }
    if (value.is<std::string>()) {
        auto s = value.as<std::string>();
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "0" || s == "off" || s == "false") {
            return ProfilingLevel::Off;
        }
        if (s == "1" || s == "standard" || s == "on" || s == "true") {
            return ProfilingLevel::Standard;
        }
        if (s == "2" || s == "detailed" || s == "detail") {
            return ProfilingLevel::Detailed;
        }
    }
    OPENVINO_THROW("Unsupported profiling level type/value");
}

// Helper that enumerates ops currently supported by the GFX plugin; keep in sync with MLIR/MSL path.
// used by query_model (and can be reused to validate compile_model paths).
bool is_supported_node(const std::shared_ptr<const ov::Node>& node) {
    if (ov::as_type_ptr<const ov::op::v0::Parameter>(node) ||
        ov::as_type_ptr<const ov::op::v0::Constant>(node) ||
        ov::as_type_ptr<const ov::op::v0::Result>(node)) {
        return true;
    }
    try {
        auto probe = MetalOpFactory::create(node, /*device*/ nullptr, /*queue*/ nullptr);
        if (!probe && metal_log_debug_enabled()) {
            GFX_LOG_DEBUG("Plugin", "Unsupported node: " << node->get_friendly_name()
                                                           << " (" << node->get_type_name() << ")");
        }
        return probe != nullptr;
    } catch (const std::exception& e) {
        if (metal_log_debug_enabled()) {
            GFX_LOG_DEBUG("Plugin", "Exception probing node " << node->get_friendly_name()
                                                                << " (" << node->get_type_name() << "): " << e.what());
        }
        return false;
    } catch (...) {
        if (metal_log_debug_enabled()) {
            GFX_LOG_DEBUG("Plugin", "Unknown exception probing node " << node->get_friendly_name()
                                                                         << " (" << node->get_type_name() << ")");
        }
        return false;
    }
}

struct UnsupportedSummary {
    std::vector<std::string> node_names;
    std::vector<std::pair<std::string, size_t>> type_counts;
};

UnsupportedSummary collect_unsupported(const std::shared_ptr<const ov::Model>& model) {
    UnsupportedSummary summary;
    std::unordered_map<std::string, size_t> counts;
    for (const auto& node : model->get_ordered_ops()) {
        if (is_supported_node(node))
            continue;
        const std::string type = node->get_type_name();
        counts[type] += 1;
        if (summary.node_names.size() < 8) {
            summary.node_names.emplace_back(node->get_friendly_name() + " (" + type + ")");
        }
    }
    summary.type_counts.reserve(counts.size());
    for (const auto& kv : counts) {
        summary.type_counts.emplace_back(kv.first, kv.second);
    }
    return summary;
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
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) const {
    OPENVINO_ASSERT(model, "Model is null");

    if (is_hetero_subgraph(model)) {
        OPENVINO_THROW("GFX plugin does not support HETERO subgraphs yet");
    }

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

    if (auto it = properties.find(kBackendProperty); it != properties.end()) {
        const auto backend_name = ov::util::to_lower(it->second.as<std::string>());
        if (backend_name != "mlir") {
            OPENVINO_THROW("Only MLIR backend is supported; received: ", backend_name);
        }
    }

    ov::AnyMap compile_properties = m_config;
    for (const auto& kv : properties) {
        compile_properties[kv.first] = kv.second;
    }

    if (!model_supported_by_metal(transformed, compile_properties)) {
        auto summary = collect_unsupported(transformed);
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

    return std::make_shared<CompiledModel>(transformed, shared_from_this(), model, compile_properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    if (!context) {
        return compile_model(model, properties);
    }
    auto metal_ctx = std::dynamic_pointer_cast<GfxRemoteContext>(context._ptr);
    OPENVINO_ASSERT(metal_ctx, "GFX: remote context type mismatch");
    ov::AnyMap merged = properties;
    merged[ov::device::id.name()] = metal_ctx->device_id();
    OPENVINO_ASSERT(model, "Model is null");

    if (is_hetero_subgraph(model)) {
        OPENVINO_THROW("GFX plugin does not support HETERO subgraphs yet");
    }

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);

    if (auto it = merged.find(kBackendProperty); it != merged.end()) {
        const auto backend_name = ov::util::to_lower(it->second.as<std::string>());
        if (backend_name != "mlir") {
            OPENVINO_THROW("Only MLIR backend is supported; received: ", backend_name);
        }
    }

    ov::AnyMap compile_properties = m_config;
    for (const auto& kv : merged) {
        compile_properties[kv.first] = kv.second;
    }

    if (!model_supported_by_metal(transformed, compile_properties)) {
        auto summary = collect_unsupported(transformed);
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

    return std::make_shared<CompiledModel>(transformed, shared_from_this(), model, compile_properties, context);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& /*model*/,
                                                         const ov::AnyMap& /*properties*/) const {
    OPENVINO_THROW("GFX plugin import_model(stream) is not implemented yet");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& /*model*/,
                                                         const ov::SoPtr<ov::IRemoteContext>& /*context*/,
                                                         const ov::AnyMap& /*properties*/) const {
    OPENVINO_THROW("GFX plugin import_model(stream, context) is not implemented yet");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::AnyMap& properties) const {
    ov::SharedStreamBuffer buffer{model.data(), model.get_byte_size()};
    std::istream stream{&buffer};
    return import_model(stream, properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    ov::SharedStreamBuffer buffer{model.data(), model.get_byte_size()};
    std::istream stream{&buffer};
    return import_model(stream, context, properties);
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& /*properties*/) const {
    OPENVINO_ASSERT(model, "Model is null");
    ov::SupportedOpsMap res;
    if (!model_supported_by_metal(model, {})) {
        // No partial fallback to CPU/HETERO: all-or-nothing support.
        return res;
    }
    for (const auto& node : model->get_ordered_ops()) {
        res.emplace(node->get_friendly_name(), get_device_name());
    }
    return res;
}

bool Plugin::model_supported_by_metal(const std::shared_ptr<const ov::Model>& model,
                                      const ov::AnyMap& /*properties*/) const {
    for (const auto& node : model->get_ordered_ops()) {
        if (!is_supported_node(node))
            return false;
    }
    return true;
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
        } else if (kv.first == ov::enable_profiling.name()) {
            m_enable_profiling = kv.second.as<bool>();
            m_config[kv.first] = kv.second;
        } else if (kv.first == "PERF_COUNT") {  // legacy spelling accepted by benchmark_app
            m_enable_profiling = kv.second.as<bool>();
            m_config[ov::enable_profiling.name()] = m_enable_profiling;
            m_config[kv.first] = kv.second;
        } else if (kv.first == kGfxProfilingLevelProperty) {
            m_profiling_level = parse_profiling_level(kv.second);
            m_profiling_level_set = true;
            m_config[kv.first] = kv.second;
        } else if (kv.first == kBackendProperty) {
            m_config[kv.first] = kv.second;
        } else if (kv.first == ov::hint::inference_precision.name()) {
            m_config[kv.first] = kv.second.as<ov::element::Type>();
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

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& /*arguments*/) const {
    const auto ro_props = [&]() {
        return std::vector<ov::PropertyName>{ov::available_devices,
                                             ov::supported_properties,
                                             ov::internal::supported_properties,
                                             ov::device::full_name,
                                             ov::device::architecture,
                                             ov::device::type,
                                             ov::device::capabilities,
                                             ov::execution_devices,
                                             ov::range_for_async_infer_requests};
    };
    const auto rw_props = [&]() {
        return std::vector<ov::PropertyName>{ov::device::id,
                                             ov::enable_profiling,
                                             ov::PropertyName{kGfxProfilingLevelProperty,
                                                              ov::PropertyMutability::RW},
                                             ov::hint::performance_mode,
                                             ov::hint::num_requests,
                                             ov::hint::execution_mode,
                                             ov::hint::inference_precision,
                                             ov::num_streams,
                                             ov::inference_num_threads,
                                             ov::log::level};
    };

    if (ov::supported_properties == name) {
        auto ro = ro_props();
        auto rw = rw_props();
        rw.push_back(ov::PropertyName{kBackendProperty, ov::PropertyMutability::RW});
        // Accept legacy PERF_COUNT spelling as RW alias of enable_profiling
        rw.push_back(ov::PropertyName{"PERF_COUNT", ov::PropertyMutability::RW});
        std::vector<ov::PropertyName> supported;
        supported.reserve(ro.size() + rw.size());
        supported.insert(supported.end(), ro.begin(), ro.end());
        supported.insert(supported.end(), rw.begin(), rw.end());
        return supported;
    } else if (ov::internal::supported_properties == name) {
        // Advertise internal properties this plugin understands (minimal set for dev flow)
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::threads_per_stream.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::cache_header_alignment.name(), ov::PropertyMutability::RO},
        };
    } else if (ov::available_devices == name) {
        auto names = metal_get_device_names();
        if (names.empty()) {
            return decltype(ov::available_devices)::value_type{{get_device_name()}};
        }
        return decltype(ov::available_devices)::value_type{names.begin(), names.end()};
    } else if (ov::device::full_name == name) {
        auto names = metal_get_device_names();
        if (!names.empty())
            return decltype(ov::device::full_name)::value_type{"GFX (" + names.front() + ")"};
        return decltype(ov::device::full_name)::value_type{"GFX"};
    } else if (ov::device::architecture == name) {
        return decltype(ov::device::architecture)::value_type{"GFX"};
    } else if (ov::device::type == name) {
        return decltype(ov::device::type)::value_type{ov::device::Type::INTEGRATED};
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{ov::device::capability::FP32,
                                                              ov::device::capability::EXPORT_IMPORT,
                                                              ov::device::capability::FP16};
    } else if (ov::device::id == name) {
        // Default single device id = 0
        return decltype(ov::device::id)::value_type{"0"};
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
    } else if (name == kBackendProperty) {
        if (auto it = m_config.find(kBackendProperty); it != m_config.end()) {
            return it->second;
        }
        return std::string("MLIR");
    } else if (ov::internal::cache_header_alignment == name) {
        // Align cache header to 64 bytes to match Template expectations and CacheHeaderAlignmentTests.
        return decltype(ov::internal::cache_header_alignment)::value_type{64u};
    } else if (ov::range_for_async_infer_requests == name) {
        // min, max, step
        return decltype(ov::range_for_async_infer_requests)::value_type{1, 1, 1};
    } else if (ov::hint::inference_precision == name) {
        if (auto it = m_config.find(ov::hint::inference_precision.name()); it != m_config.end()) {
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
    int device_id = 0;
    if (auto it = remote_properties.find(ov::device::id.name()); it != remote_properties.end()) {
        try {
            if (it->second.is<std::string>()) {
                device_id = std::stoi(it->second.as<std::string>());
            } else {
                device_id = it->second.as<int>();
            }
        } catch (...) {
            device_id = 0;
        }
    }
    auto handle = metal_get_device_by_id(device_id);
    OPENVINO_ASSERT(handle, "GFX: failed to resolve device for remote context");
    return ov::SoPtr<ov::IRemoteContext>{
        std::make_shared<GfxRemoteContext>(get_device_name(), device_id, handle, remote_properties),
        nullptr};
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    return create_context(remote_properties);
}

}  // namespace gfx_plugin
}  // namespace ov

// Plugin entry point
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_gfx_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::gfx_plugin::Plugin, version)
