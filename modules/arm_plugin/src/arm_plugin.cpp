// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <utility>
#include <memory>
#include <vector>
#include <sstream>
#include <regex>
#include <string>
#include <map>
#include <thread>

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <inference_engine.hpp>
#include <file_utils.h>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <threading/ie_executor_manager.hpp>
#include <ie_input_info.hpp>
#include <ie_layouts.h>
#include <hetero/hetero_plugin_config.hpp>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>

#include <ie_parallel.hpp>
#include "arm_ie_scheduler.hpp"
#include "arm_compute/runtime/CPP/CPPScheduler.h"

#include "arm_plugin.hpp"
#include "arm_executable_network.hpp"
#include "arm_converter/arm_converter.hpp"
#include "transformations/arm_optimizations.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace InferenceEngine::PluginConfigParams;
using namespace ArmPlugin;

Plugin::Plugin() {
    _pluginName = "CPU";
#if IE_THREAD == IE_THREAD_SEQ
    arm_compute::Scheduler::get();  // Init default AC scheduler list
    arm_compute::Scheduler::set(arm_compute::Scheduler::Type::CPP);
#else
    arm_compute::Scheduler::set(std::make_shared<IEScheduler>());
#endif
}

Plugin::~Plugin() {
    arm_compute::Scheduler::set(arm_compute::Scheduler::Type::ST);
    ExecutorManager::getInstance()->clear("CPUStreamsExecutor");
}

static std::shared_ptr<ngraph::Function> Transform(const std::shared_ptr<const ngraph::Function>& function) {
    auto transformedFunction = ngraph::clone_function(*function);
    ngraph::pass::Manager passManager;
    passManager.register_pass<pass::ArmOptimizations>();
    passManager.run_passes(transformedFunction);
    return transformedFunction;
}

InferenceEngine::ExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                                                                           const ConfigMap& config) {
    auto cfg = Configuration{config, _cfg};
    InferenceEngine::InputsDataMap networkInputs = network.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputs = network.getOutputsInfo();

    auto function = network.getFunction();
    if (function == nullptr) {
         THROW_IE_EXCEPTION << "Arm Plugin supports only ngraph cnn network representation";
    }
    return std::make_shared<ExecutableNetwork>(Transform(function), cfg, std::static_pointer_cast<Plugin>(shared_from_this()));
}

QueryNetworkResult Plugin::QueryNetwork(const CNNNetwork& network, const ConfigMap& config) const {
    QueryNetworkResult res;
    Configuration cfg{config, _cfg, false};
    auto function = network.getFunction();
    if (function == nullptr) {
         THROW_IE_EXCEPTION << "Arm Plugin supports only ngraph cnn network representation";
    }
    std::unordered_set<std::string> originalOps;
    for (auto&& node : function->get_ops()) {
        originalOps.emplace(node->get_friendly_name());
    }
    auto transformedFunction = Transform(function);
    std::unordered_set<std::string> supported;
    std::unordered_set<std::string> unsupported;
    Converter converter{transformedFunction};
    for (auto&& node : transformedFunction->get_ops()) {
        auto itConversion = converter._conversions.find(node->get_type_info());
        bool nodeIsSupported = false;
        if (itConversion != converter._conversions.end()) {
            if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node) || ngraph::op::is_output(node)) {
                nodeIsSupported = true;
            } else {
                Converter::Conversion::Ptr layer;
                try {
                    layer = converter._conversions.at(node->get_type_info())(*node);
                } catch(...) {
                    nodeIsSupported = false;
                }
                if (layer != nullptr) {
                    nodeIsSupported = static_cast<bool>(layer->Validate());
                }
            }
        }
        for (auto&& fusedLayerName : ngraph::getFusedNamesVector(node)) {
            if (contains(originalOps, fusedLayerName)) {
                if (nodeIsSupported) {
                    supported.emplace(fusedLayerName);
                } else {
                    unsupported.emplace(fusedLayerName);
                }
            }
        }
    }
    for (auto&& unsupportedNode : unsupported) {
        supported.erase(unsupportedNode);
    }
    for (auto&& node : function->get_ops()) {
        if (contains(supported, node->get_friendly_name())) {
            for (auto&& inputNodeOutput : node->input_values()) {
                if (ngraph::op::is_constant(inputNodeOutput.get_node()) || ngraph::op::is_parameter(inputNodeOutput.get_node())) {
                    supported.emplace(inputNodeOutput.get_node()->get_friendly_name());
                }
            }
            for (auto&& outputs : node->outputs()) {
                for (auto&& outputNodeInput : outputs.get_target_inputs()) {
                    if (ngraph::op::is_output(outputNodeInput.get_node())) {
                        supported.emplace(outputNodeInput.get_node()->get_friendly_name());
                    }
                }
            }
        }
    }
    for (auto&& node : function->get_ops()) {
        if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node)) {
            if (!contains(supported, node->output(0).get_target_inputs().begin()->get_node()->get_friendly_name())) {
                supported.erase(node->get_friendly_name());
            }
        } else if (ngraph::op::is_output(node)) {
            if (!contains(supported, node->input_values().begin()->get_node()->get_friendly_name())) {
                supported.erase(node->get_friendly_name());
            }
        }
    }
    for (auto&& layerName : supported) {
        res.supportedLayersMap.emplace(layerName, GetName());
    }

    return res;
}

void Plugin::SetConfig(const ConfigMap &config) {
    _cfg = Configuration{config, _cfg};
}

InferenceEngine::Parameter Plugin::GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& /*options*/) const {
    return _cfg.Get(name);
}

InferenceEngine::Parameter Plugin::GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        std::vector<std::string> supportedMetrics = {
            METRIC_KEY(AVAILABLE_DEVICES),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(FULL_DEVICE_NAME),
            METRIC_KEY(OPTIMIZATION_CAPABILITIES) };
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, supportedMetrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {
            CONFIG_KEY(PERF_COUNT) };
        auto streamExecutorConfigKeys = IStreamsExecutor::Config{}.SupportedKeys();
        for (auto&& configKey : streamExecutorConfigKeys) {
            configKeys.emplace_back(configKey);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (METRIC_KEY(AVAILABLE_DEVICES) == name) {
        std::vector<std::string> availableDevices = { "NEON" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (METRIC_KEY(FULL_DEVICE_NAME) == name) {
        std::string name = "arm_compute::NEON";
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, name);
    } else if (METRIC_KEY(OPTIMIZATION_CAPABILITIES) == name) {
        std::vector<std::string> capabilities = { METRIC_VALUE(FP32), METRIC_VALUE(FP16) };
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else  {
        THROW_IE_EXCEPTION << "Unsupported device metric: " << name;
    }
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "armPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)
