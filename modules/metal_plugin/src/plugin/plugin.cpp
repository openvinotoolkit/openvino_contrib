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
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/group_conv.hpp"
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
#include "remote_stub.hpp"
#include "transforms/pipeline.hpp"

namespace ov {
namespace metal_plugin {
namespace {

constexpr const char* kBackendProperty = "METAL_BACKEND";

bool is_fp_supported(const ov::element::Type& t) {
    return t == ov::element::f32 || t == ov::element::f16;
}

bool shape_is_static(const ov::PartialShape& ps) {
    if (!ps.rank().is_static())
        return false;
    for (const auto& d : ps) {
        if (!d.is_static())
            return false;
    }
    return true;
}

bool softmax_shape_supported(const ov::PartialShape& ps, int64_t axis) {
    if (!shape_is_static(ps))
        return false;
    const auto rank = ps.rank().get_length();
    if (rank < 2 || rank > 5)
        return false;

    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank)
        return false;

    if (rank == 4 || rank == 5) {
        // Allow channel or last axis (common cases), plus batch axis for simplicity
        if (!(axis == 1 || axis == rank - 1 || axis == 0))
            return false;
    }
    return true;
}

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
           ov::as_type_ptr<const ov::op::v1::Reshape>(node) ||
           ov::as_type_ptr<const ov::op::v0::Concat>(node) ||
           ov::as_type_ptr<const ov::op::v3::ShapeOf>(node) ||
           ov::as_type_ptr<const ov::op::v1::Gather>(node) ||
           ov::as_type_ptr<const ov::op::v0::Convert>(node) ||
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
        OPENVINO_THROW("METAL plugin does not support HETERO subgraphs yet");
    }

    if (!model_supported_by_metal(model, properties)) {
        OPENVINO_THROW("METAL: model is not fully supported");
    }

    auto transformed = ov::metal_plugin::transforms::run_pipeline(model);

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

    return std::make_shared<CompiledModel>(transformed, shared_from_this(), model, compile_properties);
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

bool Plugin::model_supported_by_metal(const std::shared_ptr<const ov::Model>& model,
                                      const ov::AnyMap& /*properties*/) const {
    // Conservative acceptance: allow only the ops / dtypes / shapes we can lower today.
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Parameter>(node.get()) || ov::is_type<ov::op::v0::Result>(node.get()) ||
            ov::is_type<ov::op::v0::Constant>(node.get())) {
            continue;
        }

        const auto& out_type = node->get_output_element_type(0);

        auto check_fp = [&](const ov::element::Type& t) {
            return is_fp_supported(t);
        };

        if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
            if (!shape_is_static(node->get_input_partial_shape(0)) ||
                !shape_is_static(node->get_input_partial_shape(1)))
                return false;
            const auto r0 = node->get_input_partial_shape(0).rank().get_length();
            const auto r1 = node->get_input_partial_shape(1).rank().get_length();
            if (r0 < 2 || r0 > 4 || r1 < 2 || r1 > 4)
                return false;
            if (!check_fp(out_type) || !check_fp(node->get_input_element_type(0)) ||
                !check_fp(node->get_input_element_type(1)))
                return false;
            continue;
        }
        if (ov::as_type_ptr<const ov::op::v1::Convolution>(node) ||
            ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
            // Conv2D policy: NCHW, static, groups==1 for now
            auto in_ps = node->get_input_partial_shape(0);
            auto w_ps  = node->get_input_partial_shape(1);
            if (!shape_is_static(in_ps) || !shape_is_static(w_ps))
                return false;
            if (in_ps.rank().get_length() != 4 || (w_ps.rank().get_length() != 4 && w_ps.rank().get_length() != 5))
                return false;
            if (!check_fp(out_type) || !check_fp(node->get_input_element_type(0)) || !check_fp(node->get_input_element_type(1)))
                return false;
            // groups==1 only
            size_t groups = 1;
            if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
                // group count is the first dimension of weights for GroupConv
                groups = node->get_input_partial_shape(1)[0].get_length();
            }
            if (groups != 1)
                return false;
            continue;
        }
        if (ov::as_type_ptr<const ov::op::v1::Add>(node) || ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
            if (!shape_is_static(node->get_input_partial_shape(0)) ||
                !shape_is_static(node->get_input_partial_shape(1)))
                return false;
            if (!check_fp(out_type) || !check_fp(node->get_input_element_type(0)) ||
                !check_fp(node->get_input_element_type(1)))
                return false;
            continue;
        }
        if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
            if (!softmax_shape_supported(node->get_input_partial_shape(0), s1->get_axis()))
                return false;
            if (!check_fp(out_type))
                return false;
            continue;
        }
        if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
            if (!softmax_shape_supported(node->get_input_partial_shape(0), s8->get_axis()))
                return false;
            if (!check_fp(out_type))
                return false;
            continue;
        }
        if (ov::as_type_ptr<const ov::op::v1::MaxPool>(node) || ov::as_type_ptr<const ov::op::v1::AvgPool>(node)) {
            auto in_ps = node->get_input_partial_shape(0);
            if (!shape_is_static(in_ps))
                return false;
            if (in_ps.rank().get_length() != 4)
                return false;
            if (!check_fp(out_type))
                return false;
            continue;
        }
        if (ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node) ||
            ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node)) {
            if (!shape_is_static(node->get_input_partial_shape(0)))
                return false;
            if (!check_fp(out_type) || !check_fp(node->get_input_element_type(0)))
                return false;
            continue;
        }
        if (ov::as_type_ptr<const ov::op::v0::Relu>(node) || ov::as_type_ptr<const ov::op::v0::Tanh>(node) ||
            ov::as_type_ptr<const ov::op::v0::Sigmoid>(node) || ov::as_type_ptr<const ov::op::v0::Elu>(node) ||
            ov::as_type_ptr<const ov::op::v0::PRelu>(node) ||
            ov::as_type_ptr<const ov::op::v0::Gelu>(node) || ov::as_type_ptr<const ov::op::v7::Gelu>(node)) {
            if (!shape_is_static(node->get_input_partial_shape(0)))
                return false;
            if (!check_fp(out_type) || !check_fp(node->get_input_element_type(0)))
                return false;
            continue;
        }
        if (ov::as_type_ptr<const ov::op::v1::Reshape>(node) || ov::as_type_ptr<const ov::op::v1::Transpose>(node) ||
            ov::as_type_ptr<const ov::op::v0::Squeeze>(node) || ov::as_type_ptr<const ov::op::v0::Unsqueeze>(node) ||
            ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node) || ov::as_type_ptr<const ov::op::v0::Concat>(node) ||
            ov::as_type_ptr<const ov::op::v3::ShapeOf>(node) || ov::as_type_ptr<const ov::op::v1::Gather>(node) ||
            ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
            // Layout / shape helper ops are accepted as long as shapes are static.
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                if (!shape_is_static(node->get_input_partial_shape(i)))
                    return false;
            }
            continue;
        }

        // Unknown op
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
            // Accept numeric IDs (0) and empty; reject arbitrary strings
            try {
                // allow both string and integral form
                auto id_any = kv.second;
                if (id_any.is<std::string>()) {
                    auto s = id_any.as<std::string>();
                    // allow empty or "0"
                    if (!s.empty() && s != "0") {
                        OPENVINO_THROW("Unsupported device id: ", s);
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
    } else if (ov::device::id == name) {
        // Default single device id = 0
        return decltype(ov::device::id)::value_type{"0"};
    } else if (ov::hint::performance_mode == name) {
        return m_performance_mode;
    } else if (ov::enable_profiling == name) {
        return m_enable_profiling;
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
    return ov::SoPtr<ov::IRemoteContext>{std::make_shared<MetalRemoteContext>(get_device_name(), remote_properties), nullptr};
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    return create_context(remote_properties);
}

}  // namespace metal_plugin
}  // namespace ov

// Plugin entry point
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_metal_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::metal_plugin::Plugin, version)
