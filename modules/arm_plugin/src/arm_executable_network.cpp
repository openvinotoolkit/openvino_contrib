// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <atomic>
#include <set>
#include <utility>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <threading/ie_executor_manager.hpp>
#include <ngraph/function.hpp>

#include "arm_plugin.hpp"
#include "arm_executable_network.hpp"
#include "arm_converter/arm_converter.hpp"

using namespace InferenceEngine;
using namespace ArmPlugin;
using namespace InferenceEngine::PluginConfigParams;

ArmPlugin::ExecutableNetwork::ExecutableNetwork(const std::shared_ptr<const ngraph::Function>&  function,
                                                const Configuration&                            cfg,
                                                const ArmPlugin::Plugin::Ptr&                   plugin):
    ExecutableNetworkThreadSafeDefault{nullptr, nullptr},
    _function{function},
    _cfg{cfg},
    _plugin{plugin} {
    InitExecutor();
}

void ArmPlugin::ExecutableNetwork::InitExecutor() {
    if (_cfg._exclusiveAsyncRequests) {
        _taskExecutor = ExecutorManager::getInstance()->getExecutor("CPU");
    } else {
        auto streamsExecutorConfig = InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(_cfg._streamsExecutorConfig);
        streamsExecutorConfig._name = "CPUStreamsExecutor";
        _taskExecutor = ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
    }
    _executor = _taskExecutor.get();
}

InferenceEngine::InferRequestInternal::Ptr
ArmPlugin::ExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                     InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<ArmInferRequest>(networkInputs,
                                             networkOutputs,
                                             std::static_pointer_cast<ExecutableNetwork>(shared_from_this()));
}

IInferRequest::Ptr ArmPlugin::ExecutableNetwork::CreateInferRequest() {
    return CreateAsyncInferRequestFromSync<InferenceEngine::AsyncInferRequestThreadSafeDefault>();
}

InferenceEngine::Parameter ArmPlugin::ExecutableNetwork::GetConfig(const std::string& name) const {
    if (name == CONFIG_KEY(PERF_COUNT)) {
        return _cfg.Get(name);
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork config key: " << name;
    }
}

InferenceEngine::Parameter ArmPlugin::ExecutableNetwork::GetMetric(const std::string& name) const {
    // TODO: return more supported values for metrics
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {
            CONFIG_KEY(PERF_COUNT),
            CONFIG_KEY(CPU_THROUGHPUT_STREAMS) };
        auto streamExecutorConfigKeys = IStreamsExecutor::Config{}.SupportedKeys();
        for (auto&& configKey : streamExecutorConfigKeys) {
            configKeys.emplace_back(configKey);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (METRIC_KEY(NETWORK_NAME) == name) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _function->get_friendly_name());
    } else if (METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) == name) {
        unsigned int value = _cfg._streamsExecutorConfig._streams;
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
    }
}
