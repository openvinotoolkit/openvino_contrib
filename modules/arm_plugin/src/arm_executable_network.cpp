// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <set>
#include <utility>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <ie_ngraph_utils.hpp>
#include <threading/ie_executor_manager.hpp>
#include <ngraph/function.hpp>
#include <ie_icore.hpp>

#include <openvino/runtime/properties.hpp>

#include "arm_plugin.hpp"
#include "arm_executable_network.hpp"
#include "arm_converter/arm_converter.hpp"

using namespace InferenceEngine;
using namespace ArmPlugin;
using namespace InferenceEngine::PluginConfigParams;

ArmPlugin::ExecutableNetwork::ExecutableNetwork(const std::shared_ptr<const ov::Model>&  model,
                                                const Configuration&                     cfg,
                                                const ArmPlugin::Plugin::Ptr&            plugin):
    ExecutableNetworkThreadSafeDefault{nullptr, nullptr},
    _model{model},
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
        streamsExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::NONE;
        _taskExecutor = ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
    }
    _executor = _taskExecutor.get();
}

InferenceEngine::IInferRequestInternal::Ptr
ArmPlugin::ExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                     InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<ArmInferRequest>(networkInputs,
                                             networkOutputs,
                                             std::static_pointer_cast<ExecutableNetwork>(shared_from_this()));
}
InferenceEngine::IInferRequestInternal::Ptr
ArmPlugin::ExecutableNetwork::CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                     const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    if (!this->_plugin || !this->_plugin->GetCore() || !this->_plugin->GetCore()->isNewAPI())
        return nullptr;
    return std::make_shared<ArmInferRequest>(inputs,
                                             outputs,
                                             std::static_pointer_cast<ExecutableNetwork>(shared_from_this()));
}

InferenceEngine::Parameter ArmPlugin::ExecutableNetwork::GetConfig(const std::string& name) const {
    return _cfg.Get(name);
}

InferenceEngine::Parameter ArmPlugin::ExecutableNetwork::GetMetric(const std::string& name) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{
            ov::model_name.name(),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            ov::supported_properties.name(),
            ov::inference_num_threads.name(),
            ov::streams::num.name(),
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys;
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (ov::supported_properties == name) {
        return decltype(ov::supported_properties)::value_type{
            {ov::model_name.name(), ov::PropertyMutability::RO},
            {ov::supported_properties.name(), ov::PropertyMutability::RO},
            {ov::optimal_number_of_infer_requests.name(), ov::PropertyMutability::RO},
            {ov::streams::num.name(), ov::PropertyMutability::RO},
            {ov::inference_num_threads.name(), ov::PropertyMutability::RO}};
    } else if (ov::model_name == name) {
        return decltype(ov::model_name)::value_type{_model->get_friendly_name()};
    } else if (ov::optimal_number_of_infer_requests == name) {
        return decltype(ov::optimal_number_of_infer_requests)::value_type(
            _cfg._streamsExecutorConfig._streams);
    } else if (ov::inference_num_threads == name) {
        return decltype(ov::inference_num_threads)::value_type(
            _cfg._streamsExecutorConfig._threads);
    } else if (ov::streams::num == name) {
        return decltype(ov::streams::num)::value_type{
            _cfg._streamsExecutorConfig._streams};
    }  else {
        IE_THROW() << "Unsupported ExecutableNetwork metric: " << name;
    }
}

std::shared_ptr<ov::Model> ArmPlugin::ExecutableNetwork::GetExecGraphInfo() {
    for (auto&& node : _model->get_ops()) {
        auto& rtInfo = node->get_rt_info();
        rtInfo.emplace("layerType", node->get_type_name());
        rtInfo.emplace("runtimePrecision", InferenceEngine::details::convertPrecision(node->output(0).get_element_type()).name());
    }
    return std::const_pointer_cast<ov::Model>(_model);
}
