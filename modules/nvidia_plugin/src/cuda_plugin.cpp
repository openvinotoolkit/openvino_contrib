// Copyright (C) 20182023 Intel Corporation
// SPDXLicenseIdentifier: Apache2.0
//
#include <fmt/format.h>

#include "ie_metric_helpers.hpp"

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cuda/props.hpp"
#include "cuda_compiled_model.hpp"
#include "cuda_infer_request.hpp"
#include "cuda_itt.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_plugin.hpp"
#include "nvidia/nvidia_config.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "threading/ie_executor_manager.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformer/nodes/concat_optimized.hpp"
#include "transformer/nodes/fully_connected.hpp"
#include "transformer/nodes/fused_convolution.hpp"
#include "transformer/nodes/fused_convolution_backprop_data.hpp"
#include "transformer/nodes/lstm_sequence_optimized.hpp"

using namespace ov::nvidia_gpu;

Plugin::Plugin() {
    set_device_name("NVIDIA");
    for (size_t i = 0; i < CUDA::Device::count(); ++i) {
        CUDA::Device device{i};
        const size_t num_concurrent_streams = max_concurrent_streams(device);
        device_thread_pool_[std::to_string(i)] = std::make_shared<CudaThreadPool>(device, num_concurrent_streams);
    }
}

Plugin::~Plugin() {
}

std::shared_ptr<ov::threading::ITaskExecutor> Plugin::get_stream_executor(const Configuration& config) const {
    auto device_id = std::to_string(config.get_device_id());
    OPENVINO_ASSERT(device_thread_pool_.count(device_id), "Device id is out of range!");
    return device_thread_pool_.at(device_id);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    return compile_model(model, properties, {});
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::RemoteContext& context) const {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "Plugin::compile_model");

    auto full_config = Configuration{properties, config_};
    CUDA::Device device{full_config.get_device_id()};

    // Create stream executor for given device
    auto wait_executor = get_stream_executor(full_config);
    auto compiled_model = std::make_shared<CompiledModel>(model->clone(),
                                                          full_config,
                                                          wait_executor,
                                                          shared_from_this());
    return compiled_model;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model_stream,
                                                         const ov::AnyMap& properties) const {
    return import_model(model_stream, {}, properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model_stream,
                                                         const ov::RemoteContext& context,
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

    // Register operation itself, required to be read from IR
    const std::vector<ov::Extension::Ptr> extensions = {
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::ConcatOptimized>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::FullyConnected>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::FusedConvBackpropData>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::FusedConvolution>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::FusedGroupConvolution>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::LSTMSequenceOptimized>>()};

    ov::Core core;
    core.add_extension(extensions);
    auto model = core.read_model(xml_string, weights);

    Configuration full_config{properties, config_};
    auto wait_executor = get_stream_executor(full_config);
    auto compiled_model= std::make_shared<CompiledModel>(model,
                                                         full_config,
                                                         wait_executor,
                                                         shared_from_this(),
                                                         true);
    return compiled_model;
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                        const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "ov::nvidia_gpu::query_model");

    Configuration full_config{properties, config_, false};

    auto supported = ov::get_supported_nodes(model,
    [&](std::shared_ptr<ov::Model>& model) {
            transformer_.transform(CUDA::Device{full_config.get_device_id()}, model, full_config);
        },
    [&](const std::shared_ptr<ngraph::Node>& op) {
        return is_operation_supported(op);
    });

    ov::SupportedOpsMap res;
    for (auto&& op_name : supported) {
        res.emplace(op_name, get_device_name() + "." + std::to_string(full_config.get_device_id()));
    }
    return res;
}

std::shared_ptr<ov::IRemoteContext> Plugin::create_context(
    const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteContext> Plugin::get_default_context(
    const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

bool Plugin::is_operation_supported(const std::shared_ptr<ov::Node>& node) const {
    bool is_op_supported = false;
    if (OperationRegistry::getInstance().hasOperation(node)) {
        const TensorID dummyTensorID{0};
        const CreationContext context{CUDA::Device{config_.get_device_id()}, false};
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

void Plugin::set_property(const ov::AnyMap& properties) { config_ = Configuration{properties, config_}; }

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& properties) const {
    using namespace InferenceEngine::CUDAMetrics;

    Configuration full_config{properties, config_};

    if (ov::supported_properties == name) {
        return decltype(ov::supported_properties)::value_type{Configuration::get_supported_properties()};
    } else if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        std::vector<std::string> supportedMetrics = {METRIC_KEY(AVAILABLE_DEVICES),
                                                     METRIC_KEY(SUPPORTED_METRICS),
                                                     METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                     ov::device::uuid.name(),
                                                     METRIC_KEY(FULL_DEVICE_NAME),
                                                     METRIC_KEY(IMPORT_EXPORT_SUPPORT),
                                                     METRIC_KEY(DEVICE_ARCHITECTURE),
                                                     METRIC_KEY(OPTIMIZATION_CAPABILITIES),
                                                     METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)};
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, supportedMetrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {
            CONFIG_KEY(DEVICE_ID), CONFIG_KEY(PERF_COUNT), NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS)};
        auto streamExecutorConfigKeys = InferenceEngine::IStreamsExecutor::Config{}.SupportedKeys();
        for (auto&& configKey : streamExecutorConfigKeys) {
            if (configKey != InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS) {
                configKeys.emplace_back(configKey);
            }
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (ov::caching_properties == name) {
        return decltype(ov::caching_properties)::value_type{Configuration::get_caching_properties()};
    } else if (ov::available_devices == name) {
        std::vector<std::string> availableDevices = {};
        for (size_t i = 0; i < CUDA::Device::count(); ++i) {
            availableDevices.push_back(fmt::format("{}.{}", get_device_name(), i));
        }
        return decltype(ov::available_devices)::value_type{availableDevices};
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
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
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
    } else if (METRIC_KEY(OPTIMIZATION_CAPABILITIES) == name) {
        std::vector<std::string> capabilities = {METRIC_VALUE(FP32)};
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
     } else if (ov::range_for_streams == name) {
        return decltype(ov::range_for_streams)::value_type{1, Configuration::reasonable_limit_of_streams};
    } else if (ov::range_for_async_infer_requests == name) {
        return decltype(ov::range_for_async_infer_requests)::value_type{1, 1, 1};
    } else {
        return full_config.get(name);
    }
}
