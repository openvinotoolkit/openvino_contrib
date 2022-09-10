// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
// ^^ must come before ie_plugin_config.hpp, which is included by
// hetero_plugin_config.hpp
#include <fmt/format.h>

#include <cuda/props.hpp>
#include <hetero/hetero_plugin_config.hpp>
#include <ie_algorithm.hpp>
#include <ie_ngraph_utils.hpp>
#include <ie_plugin_config.hpp>
#include <ngraph/opsets/opset.hpp>
#include <openvino/op/util/op_types.hpp>
#include <threading/ie_executor_manager.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>

#include "nvidia/cuda_config.hpp"
#include "cuda_executable_network.hpp"
#include "cuda_infer_request.hpp"
#include "cuda_itt.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_plugin.hpp"
#include "openvino/runtime/properties.hpp"
using namespace ov::nvidia_gpu;

Plugin::Plugin() { _pluginName = "NVIDIA"; }

Plugin::~Plugin() {
    // Plugin should remove executors from executor cache to avoid threads number growth in the whole application
    executorManager()->clear("CudaCPUPreprocessExecutor");
    // NOTE: Uncomment this if Inference Engine Executor cache is used to create callback executor
    executorManager()->clear("CudaCallbackExecutor");
}

InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                                                                            const ConfigMap& config) {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "Plugin::LoadExeNetworkImpl");

    auto cfg = Configuration{config, _cfg};
    InferenceEngine::InputsDataMap networkInputs = network.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputs = network.getOutputsInfo();

    // TODO: check with precisions supported by Cuda device

    for (auto networkOutput : networkOutputs) {
        auto output_precision = networkOutput.second->getPrecision();

        if (output_precision != InferenceEngine::Precision::FP32 &&
            output_precision != InferenceEngine::Precision::FP16 &&
            output_precision != InferenceEngine::Precision::I32 &&
            output_precision != InferenceEngine::Precision::I16 && output_precision != InferenceEngine::Precision::U8 &&
            output_precision != InferenceEngine::Precision::I8 &&
            output_precision != InferenceEngine::Precision::BOOL) {
            throwIEException(
                fmt::format("Output format {} is not supported yet. Supported "
                            "formats are: FP32, FP16, I32, I16, I8, U8 and BOOL.",
                            output_precision));
        }
    }

    for (auto networkInput : networkInputs) {
        auto input_precision = networkInput.second->getTensorDesc().getPrecision();

        if (input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::FP16 && input_precision != InferenceEngine::Precision::I32 &&
            input_precision != InferenceEngine::Precision::I16 && input_precision != InferenceEngine::Precision::U8 &&
            input_precision != InferenceEngine::Precision::I8 && input_precision != InferenceEngine::Precision::BOOL) {
            throwIEException(
                fmt::format("Input format {} is not supported yet. Supported "
                            "formats are: FP32, FP16, I32, I16, I8, U8 and BOOL.",
                            input_precision));
        }
    }

    // Create stream executor for given device
    auto waitExecutor = GetStreamExecutor(cfg);
    return std::make_shared<ExecutableNetwork>(
        network, cfg, waitExecutor, std::static_pointer_cast<Plugin>(shared_from_this()));
}

InferenceEngine::ITaskExecutor::Ptr Plugin::GetStreamExecutor(const Configuration& cfg) {
    // TODO: get available integer value instead of chain of conversions
    std::string deviceId = cfg.Get(CONFIG_KEY(DEVICE_ID));
    CUDA::Device device{std::stoi(deviceId)};
    const size_t numConcurrentStreams = maxConcurrentStreams(device);
    {
        std::lock_guard<std::mutex> lock{mtx_};
        auto& p = device_thread_pool_[deviceId];
        if (!p) p = std::make_shared<CudaThreadPool>(device, numConcurrentStreams);
        return p;
    }
}

InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::ImportNetwork(
    std::istream& model, const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "ov::nvidia_gpu::ImportNetworkImpl");

    Configuration cfg{config, _cfg};
    auto waitExecutor = GetStreamExecutor(cfg);
    auto exec = std::make_shared<ExecutableNetwork>(
        model, std::move(cfg), std::move(waitExecutor), std::static_pointer_cast<Plugin>(shared_from_this()));
    SetExeNetworkInfo(exec, exec->export_function_);
    return exec;
}

InferenceEngine::QueryNetworkResult Plugin::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                         const ConfigMap& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "ov::nvidia_gpu::QueryNetwork");

    InferenceEngine::QueryNetworkResult res;
    Configuration cfg{config, _cfg, false};

    auto function = network.getFunction();
    if (function == nullptr) {
        throwIEException("CUDA Plugin supports only ngraph cnn network representation");
    }

    // 1. First of all we should store initial input operation set
    std::unordered_set<std::string> originalOps;
    std::map<std::string, ov::NodeTypeInfo> friendlyNameToType;
    for (auto&& node : function->get_ops()) {
        originalOps.emplace(node->get_friendly_name());
        friendlyNameToType[node->get_friendly_name()] = node->get_type_info();
    }

    // 2. It is needed to apply all transformations as it is done in LoadExeNetworkImpl
    auto transformedFunction = transformer_.transform(
        CUDA::Device{cfg.deviceId}, network.getFunction(), network.getInputsInfo(), network.getOutputsInfo(), cfg);

    // 3. The same input node can be transformed into supported and unsupported backend node
    // So we need store as supported either unsupported node sets
    std::unordered_set<std::string> supported;
    std::unordered_set<std::string> unsupported;
    for (auto&& node : transformedFunction->get_ops()) {
        const bool isOpSupported = isOperationSupported(node);
        // Extract transformation history from transformed node as list of nodes
        for (auto&& fusedLayerName : ngraph::getFusedNamesVector(node)) {
            // Filter just nodes from original operation set
            if (InferenceEngine::details::contains(originalOps, fusedLayerName)) {
                if (isOpSupported) {
                    supported.emplace(fusedLayerName);
                } else {
                    unsupported.emplace(fusedLayerName);
                }
            }
        }
    }

    // 4. The result set should contain just nodes from supported set
    for (auto&& unsupportedNode : unsupported) {
        supported.erase(unsupportedNode);
    }

    for (auto&& node : function->get_ops()) {
        // 5. If some housekeeping nodes were not added - add them.
        if (InferenceEngine::details::contains(supported, node->get_friendly_name())) {
            for (auto&& inputNodeOutput : node->input_values()) {
                if (ov::op::util::is_constant(inputNodeOutput.get_node()) ||
                    ov::op::util::is_parameter(inputNodeOutput.get_node())) {
                    supported.emplace(inputNodeOutput.get_node()->get_friendly_name());
                }
            }
            for (auto&& outputs : node->outputs()) {
                for (auto&& outputNodeInput : outputs.get_target_inputs()) {
                    if (ov::op::util::is_output(outputNodeInput.get_node())) {
                        supported.emplace(outputNodeInput.get_node()->get_friendly_name());
                    }
                }
            }
        }

        // 6. Eliminate subgraphs that consist of housekeeping nodes only
        if (ov::op::util::is_constant(node) || ov::op::util::is_parameter(node)) {
            if (!InferenceEngine::details::contains(
                    supported, node->output(0).get_target_inputs().begin()->get_node()->get_friendly_name())) {
                supported.erase(node->get_friendly_name());
            }
        } else if (ov::op::util::is_output(node)) {
            auto name = node->input_values().begin()->get_node()->get_friendly_name();
            if (!InferenceEngine::details::contains(supported, name)) {
                supported.erase(node->get_friendly_name());
            }
        }
    }

    // 7. Produce the result
    for (auto&& layerName : supported) {
        res.supportedLayersMap.emplace(layerName, GetName());
    }

    return res;
}

bool Plugin::isOperationSupported(const std::shared_ptr<ov::Node>& node) const {
    bool isOpSupported = false;
    if (OperationRegistry::getInstance().hasOperation(node)) {
        const TensorID dummyTensorID{0};
        const CreationContext context{CUDA::Device{_cfg.deviceId}, false};
        const std::vector<TensorID> inIds(node->get_input_size(), dummyTensorID);
        const std::vector<TensorID> outIds(node->get_output_size(), dummyTensorID);
        try {
            OperationRegistry::getInstance().createOperation(context, node, inIds, outIds);
            isOpSupported = true;
        } catch (...) {
        }
    }
    return isOpSupported;
}

void Plugin::SetConfig(const ConfigMap& config) { _cfg = Configuration{config, _cfg}; }

InferenceEngine::Parameter Plugin::GetConfig(
    const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& /*options*/) const {
    return _cfg.Get(name);
}

InferenceEngine::Parameter Plugin::GetMetric(const std::string& name,
                                             const std::map<std::string, InferenceEngine::Parameter>& options) const {
    using namespace InferenceEngine::CUDAMetrics;
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        std::vector<std::string> supportedMetrics = {METRIC_KEY(AVAILABLE_DEVICES),
                                                     METRIC_KEY(SUPPORTED_METRICS),
                                                     METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                     METRIC_KEY(FULL_DEVICE_NAME),
                                                     METRIC_KEY(IMPORT_EXPORT_SUPPORT),
                                                     METRIC_KEY(DEVICE_ARCHITECTURE),
                                                     METRIC_KEY(OPTIMIZATION_CAPABILITIES),
                                                     METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)};
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, supportedMetrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {
            CONFIG_KEY(DEVICE_ID), CONFIG_KEY(PERF_COUNT), CUDA_CONFIG_KEY(THROUGHPUT_STREAMS)};
        auto streamExecutorConfigKeys = InferenceEngine::IStreamsExecutor::Config{}.SupportedKeys();
        for (auto&& configKey : streamExecutorConfigKeys) {
            if (configKey != InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS) {
                configKeys.emplace_back(configKey);
            }
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
        // TODO: Uncomment when will be required 'SUPPORTED_PROPERTIES' and check all tests
        //    } else if (ov::supported_properties == name) {
        //        using properties_type = decltype(ov::supported_properties)::value_type;
        //        properties_type supportedMetrics = {
        //            ov::supported_properties.name(),
        //            METRIC_KEY(AVAILABLE_DEVICES),
        //            METRIC_KEY(SUPPORTED_METRICS),
        //            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        //            METRIC_KEY(FULL_DEVICE_NAME),
        //            METRIC_KEY(IMPORT_EXPORT_SUPPORT),
        //            METRIC_KEY(DEVICE_ARCHITECTURE),
        //            METRIC_KEY(OPTIMIZATION_CAPABILITIES),
        //            METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)
        //        };
        //        properties_type configKeys = {
        //            CONFIG_KEY(DEVICE_ID),
        //            CONFIG_KEY(PERF_COUNT),
        //            CUDA_CONFIG_KEY(THROUGHPUT_STREAMS)
        //        };
        //        auto streamExecutorConfigKeys = InferenceEngine::IStreamsExecutor::Config{}.SupportedKeys();
        //        for (auto&& configKey : streamExecutorConfigKeys) {
        //            if (configKey != InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS) {
        //                configKeys.emplace_back(configKey);
        //            }
        //        }
        //        properties_type all_properties;
        //        all_properties.insert(all_properties.end(), supportedMetrics.begin(), supportedMetrics.end());
        //        all_properties.insert(all_properties.end(), configKeys.begin(), configKeys.end());
        //        return all_properties;
    } else if (METRIC_KEY(AVAILABLE_DEVICES) == name) {
        // TODO: fill list of available devices
        std::vector<std::string> availableDevices = {""};
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (METRIC_KEY(FULL_DEVICE_NAME) == name) {
        std::string name = getCudaAttribute<Plugin::cuda_attribute::name, std::string>();
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, name);
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (METRIC_KEY(DEVICE_ARCHITECTURE) == name) {
        // TODO: return device architecture for device specified by DEVICE_ID config
        std::string arch = "CUDA";
        IE_SET_METRIC_RETURN(DEVICE_ARCHITECTURE, arch);
    } else if (METRIC_KEY(OPTIMIZATION_CAPABILITIES) == name) {
        // TODO: fill actual list of supported capabilities: e.g. Cuda device supports only FP32
        std::vector<std::string> capabilities = {METRIC_VALUE(FP32) /*, TEMPLATE_METRIC_VALUE(HARDWARE_CONVOLUTION)*/};
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS) == name) {
        // TODO: fill with actual values
        using uint = unsigned int;
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, std::make_tuple(uint{1}, uint{1}, uint{1}));
    } else {
        IE_THROW(NotFound) << "Unsupported device metric: " << name;
    }
}
