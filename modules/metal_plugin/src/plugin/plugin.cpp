// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <istream>
#include <memory>
#include <vector>

#include "compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/gelu.hpp"
#if __has_include("openvino/op/layer_norm.hpp")
#include "openvino/op/layer_norm.hpp"
#endif
#include "openvino/op/result.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "runtime/backend.hpp"
#include "transforms/pipeline.hpp"

namespace ov {
namespace metal_plugin {
namespace {

constexpr const char* kBackendProperty = "METAL_BACKEND";

// Helper that enumerates ops currently supported by the MPSGraph-backed METAL plugin;
// used by query_model (and can be reused to validate compile_model paths).
bool is_supported_node(const std::shared_ptr<const ov::Node>& node) {
    return ov::as_type_ptr<const ov::op::v0::Parameter>(node) ||
           ov::as_type_ptr<const ov::op::v0::Constant>(node) ||
           ov::as_type_ptr<const ov::op::v0::Result>(node) ||
           ov::as_type_ptr<const ov::op::v0::Relu>(node) ||
           ov::as_type_ptr<const ov::op::v0::Tanh>(node) ||
           ov::as_type_ptr<const ov::op::v0::Sigmoid>(node) ||
           ov::as_type_ptr<const ov::op::v0::Elu>(node) ||
           ov::as_type_ptr<const ov::op::v0::PRelu>(node) ||
           ov::as_type_ptr<const ov::op::v1::Add>(node) ||
           ov::as_type_ptr<const ov::op::v0::MatMul>(node) ||
           ov::as_type_ptr<const ov::op::v1::Convolution>(node) ||
           ov::as_type_ptr<const ov::op::v1::MaxPool>(node) ||
           ov::as_type_ptr<const ov::op::v1::AvgPool>(node) ||
           ov::as_type_ptr<const ov::op::v1::Softmax>(node) ||
           ov::as_type_ptr<const ov::op::v8::Softmax>(node) ||
           ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node) ||
           ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node) ||
           ov::as_type_ptr<const ov::op::v7::Gelu>(node) ||
           ov::as_type_ptr<const ov::op::v0::Gelu>(node)
#if __has_include("openvino/op/layer_norm.hpp")
           || ov::as_type_ptr<const ov::op::v12::LayerNorm>(node)
#endif
        ;
}

}  // namespace

Plugin::Plugin() {
    // Set device name exposed to Core
    m_device_name = "METAL";
    set_device_name(m_device_name);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) const {
    OPENVINO_ASSERT(model, "Model is null");

    auto transformed = ov::metal_plugin::transforms::run_pipeline(model);

    if (auto it = properties.find(kBackendProperty); it != properties.end()) {
        const auto backend_name = ov::util::to_lower(it->second.as<std::string>());
        if (backend_name != "mlir") {
            OPENVINO_THROW("Only MLIR backend is supported; received: ", backend_name);
        }
    }

    return std::make_shared<CompiledModel>(transformed, shared_from_this());
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& /*context*/) const {
    return compile_model(model, properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& /*model*/,
                                                         const ov::AnyMap& /*properties*/) const {
    OPENVINO_THROW("METAL plugin import_model(stream) is not implemented yet");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& /*model*/,
                                                         const ov::SoPtr<ov::IRemoteContext>& /*context*/,
                                                         const ov::AnyMap& /*properties*/) const {
    OPENVINO_THROW("METAL plugin import_model(stream, context) is not implemented yet");
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
    for (const auto& node : model->get_ordered_ops()) {
        if (is_supported_node(node)) {
            res.emplace(node->get_friendly_name(), get_device_name());
        }
    }
    return res;
}

void Plugin::set_property(const ov::AnyMap& properties) {
    for (const auto& kv : properties) {
        if (kv.first == ov::hint::performance_mode.name()) {
            m_performance_mode = kv.second.as<ov::hint::PerformanceMode>();
            m_config[kv.first] = kv.second;
        } else if (kv.first == kBackendProperty) {
            m_config[kv.first] = kv.second;
        } else {
            m_config[kv.first] = kv.second;  // store unknown for now
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
                                             ov::hint::performance_mode,
                                             ov::hint::num_requests,
                                             ov::hint::execution_mode,
                                             ov::num_streams,
                                             ov::inference_num_threads,
                                             ov::log::level};
    };

    if (ov::supported_properties == name) {
        auto ro = ro_props();
        auto rw = rw_props();
        rw.push_back(ov::PropertyName{kBackendProperty, ov::PropertyMutability::RW});
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
        // TODO: enumerate actual Metal devices/GPUs
        return decltype(ov::available_devices)::value_type{{get_device_name()}};
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{"METAL (Apple GPU)"};
    } else if (ov::device::architecture == name) {
        return decltype(ov::device::architecture)::value_type{"METAL"};
    } else if (ov::device::type == name) {
        return decltype(ov::device::type)::value_type{ov::device::Type::INTEGRATED};
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{ov::device::capability::FP32,
                                                              ov::device::capability::EXPORT_IMPORT,
                                                              ov::device::capability::FP16};
    } else if (ov::hint::performance_mode == name) {
        return m_performance_mode;
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_device_name()};
    } else if (name == kBackendProperty) {
    if (auto it = m_config.find(kBackendProperty); it != m_config.end()) {
        return it->second;
    }
    return std::string("MLIR");
    } else if (ov::range_for_async_infer_requests == name) {
        // min, max, step
        return decltype(ov::range_for_async_infer_requests)::value_type{1, 1, 1};
    }

    // Check user-provided properties preserved in m_config
    if (auto it = m_config.find(name); it != m_config.end()) {
        return it->second;
    }

    OPENVINO_THROW("Unsupported property: ", name);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& /*remote_properties*/) const {
    OPENVINO_THROW("METAL remote context is not implemented yet");
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& /*remote_properties*/) const {
    OPENVINO_THROW("METAL default remote context is not implemented yet");
}

}  // namespace metal_plugin
}  // namespace ov

// Plugin entry point
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_metal_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::metal_plugin::Plugin, version)
