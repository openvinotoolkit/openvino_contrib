// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <fmt/format.h>

#include "cuda/props.hpp"
#include "cuda_compiled_model.hpp"
#include "cuda_infer_request.hpp"
#include "cuda_itt.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_plugin.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

using namespace ov::nvidia_gpu;

Plugin::Plugin() {
    set_device_name("NVIDIA");
    for (int i = 0; i < CUDA::Device::count(); ++i) {
        CUDA::Device device{i};
        const size_t num_concurrent_streams = max_concurrent_streams(device);
        device_thread_pool_[std::to_string(i)] = std::make_shared<CudaThreadPool>(device, num_concurrent_streams);
        configs_.insert({std::to_string(i),
            Configuration({ov::device::id(i),
                            ov::hint::inference_precision(isHalfSupported(device) ? ov::element::f16 : ov::element::f32)})});
    }
}

Plugin::~Plugin() {
}

std::shared_ptr<ov::threading::ITaskExecutor> Plugin::get_stream_executor(const Configuration& config) const {
    auto device_id = std::to_string(config.get_device_id());
    OPENVINO_ASSERT(device_thread_pool_.count(device_id), "Couldn't find config for NVIDIA with id ", device_id);
    return device_thread_pool_.at(device_id);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    return compile_model(model, properties, {});
}

Configuration Plugin::get_full_config(const ov::AnyMap& properties, const bool throw_on_unsupported) const {
    std::string device_id = default_device_id;
    if (properties.find(ov::device::id.name()) != properties.end()) {
        device_id = properties.find(ov::device::id.name())->second.as<std::string>();
        auto pos = device_id.find_last_of("NVIDIA.");
        if (pos != std::string::npos) {
            device_id = device_id.substr(pos + 1);
        }
    }
    OPENVINO_ASSERT(configs_.find(device_id) != configs_.end(), "Couldn't find config for NVIDIA with id ", device_id);
    return Configuration{properties, configs_.at(device_id), throw_on_unsupported};
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "Plugin::compile_model");

    auto full_config = get_full_config(properties);
    CUDA::Device device{full_config.get_device_id()};

    // Create stream executor for given device
    auto wait_executor = get_stream_executor(full_config);
    auto compiled_model = std::make_shared<CompiledModel>(model->clone(),
                                                          full_config,
                                                          wait_executor,
                                                          shared_from_this(),
                                                          false);
    return compiled_model;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model, const ov::AnyMap& properties) const {
    ov::SharedStreamBuffer buffer{reinterpret_cast<char*>(model.data()), model.get_byte_size()};
    std::istream stream{&buffer};
    return import_model(stream, properties);
};

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    ov::SharedStreamBuffer buffer{reinterpret_cast<char*>(model.data()), model.get_byte_size()};
    std::istream stream{&buffer};
    return import_model(stream, context, properties);
};

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model_stream,
                                                         const ov::AnyMap& properties) const {
    return import_model(model_stream, {}, properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model_stream,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "ov::nvidia_gpu::import_model");

    // Read XML content
    std::string xml_string;
    std::uint64_t data_size = 0;
    model_stream.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    xml_string.resize(data_size);
    model_stream.read(const_cast<char*>(xml_string.c_str()), data_size);

    // Read blob content
    ov::Tensor weights;
    model_stream.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    if (0 != data_size) {
        weights = ov::Tensor(ov::element::from<char>(), ov::Shape{static_cast<ov::Shape::size_type>(data_size)});
        model_stream.read(weights.data<char>(), data_size);
    }

    auto model = get_core()->read_model(xml_string, weights);

    // check ov::loaded_from_cache property and erase it due to not needed any more.
    auto _properties = properties;
    const auto& it = _properties.find(ov::loaded_from_cache.name());
    bool loaded_from_cache = false;
    if (it != _properties.end()) {
        loaded_from_cache = it->second.as<bool>();
        _properties.erase(it);
    }

    auto full_config = get_full_config(_properties);
    auto wait_executor = get_stream_executor(full_config);
    auto compiled_model= std::make_shared<CompiledModel>(model,
                                                         full_config,
                                                         wait_executor,
                                                         shared_from_this(),
                                                         loaded_from_cache);
    return compiled_model;
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "ov::nvidia_gpu::query_model");

    auto full_config = get_full_config(properties, false);

    auto supported = ov::get_supported_nodes(model,
    [&](std::shared_ptr<ov::Model>& model) {
            transformer_.transform(CUDA::Device{full_config.get_device_id()}, model, full_config);
        },
    [&](const std::shared_ptr<ov::Node>& op) {
        return is_operation_supported(op, full_config);
    });

    ov::SupportedOpsMap res;
    for (auto&& op_name : supported) {
        res.emplace(op_name, get_device_name() + "." + std::to_string(full_config.get_device_id()));
    }
    return res;
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(
    const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(
    const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

bool Plugin::is_operation_supported(const std::shared_ptr<ov::Node>& node, const Configuration& config) const {
    bool is_op_supported = false;
    if (node->is_dynamic()) {
        return false;
    }
    if (OperationRegistry::getInstance().hasOperation(node)) {
        const TensorID dummyTensorID{0};
        const CreationContext context{CUDA::Device{config.get_device_id()}, false};
        const std::vector<TensorID> inIds(node->get_input_size(), dummyTensorID);
        const std::vector<TensorID> outIds(node->get_output_size(), dummyTensorID);
        try {
            OperationRegistry::getInstance().createOperation(context, node, inIds, outIds);
            is_op_supported = true;
        } catch (...) {
        }
    }
    return is_op_supported;
}

void Plugin::set_property(const ov::AnyMap& properties) {
    auto has_property_value = [&properties](const std::string& name) {
        return properties.find(name) != properties.end();
    };
    auto get_property_value = [&properties](const std::string& name) {
        return properties.at(name).as<std::string>();
    };
    auto update_config = [this, &properties](const ov::AnyMap& config_properties, Configuration& config) {
        config = Configuration{config_properties, config};
    };
    if (has_property_value(ov::internal::config_device_id.name())) {
        std::string device_id = get_property_value(ov::internal::config_device_id.name());
        auto properties_for_device = properties;
        properties_for_device.erase(ov::internal::config_device_id.name());
        update_config(properties_for_device, configs_.at(device_id));
    } else {
        if (has_property_value(ov::device::id.name())) {
            default_device_id = get_property_value(ov::device::id.name());
            update_config(properties, configs_.at(default_device_id));
        } else {
            for (auto& conf : configs_) {
                update_config(properties, conf.second);
            }
        }
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& properties) const {
    auto full_config = get_full_config(properties);

    if (ov::supported_properties == name) {
        return decltype(ov::supported_properties)::value_type{Configuration::get_supported_properties()};
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{Configuration::get_supported_internal_properties()};
    } else if (ov::internal::caching_properties == name) {
        return decltype(ov::internal::caching_properties)::value_type{Configuration::get_caching_properties()};
    } else if (ov::available_devices == name) {
        std::vector<std::string> available_devices = {};
        for (size_t i = 0; i < CUDA::Device::count(); ++i) {
            available_devices.push_back(std::to_string(i));
        }
        return decltype(ov::available_devices)::value_type{available_devices};
    } else if (ov::device::uuid == name) {
        CUDA::Device device{full_config.get_device_id()};
        const auto& props = device.props();
        ov::device::UUID uuid = {};
        std::copy(std::begin(props.uuid.bytes), std::end(props.uuid.bytes), std::begin(uuid.uuid));
        return decltype(ov::device::uuid)::value_type{uuid};
    } else if (ov::device::full_name == name) {
        CUDA::Device device{full_config.get_device_id()};
        const auto& props = device.props();
        const std::string name = props.name;
        return decltype(ov::device::full_name)::value_type{name};
    } else if (ov::device::architecture == name) {
        CUDA::Device device{full_config.get_device_id()};
        const auto& props = device.props();
        std::stringstream ss;
        ss << "NVIDIA: ";
        ss << "v" << props.major;
        ss << "." << props.minor;
        return decltype(ov::device::architecture)::value_type{ss.str()};
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{{
            ov::device::capability::EXPORT_IMPORT,
            ov::device::capability::FP32,
            ov::device::capability::FP16}};
    } else if (ov::range_for_streams == name) {
        return decltype(ov::range_for_streams)::value_type{1, Configuration::reasonable_limit_of_streams};
    } else if (ov::range_for_async_infer_requests == name) {
        return decltype(ov::range_for_async_infer_requests)::value_type{1, 1, 1};
    } else {
        return full_config.get(name);
    }
}
