// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include <ie_common.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "arm_config.hpp"
#include "arm_infer_request.hpp"

namespace ArmPlugin {

class Plugin;

struct ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
    ExecutableNetwork(const std::shared_ptr<const ngraph::Function>&  func,
                      const Configuration&           cfg,
                      const std::shared_ptr<Plugin>& plugin);

    InferenceEngine::IInferRequestInternal::Ptr
    CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                           InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::Parameter GetMetric(const std::string& name) const override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    InferenceEngine::CNNNetwork GetExecGraphInfo() override;

    void RegistorTranslations();
    void Translate(std::shared_ptr<const ngraph::Function> function);
    void InitExecutor();

    std::shared_ptr<const ngraph::Function>                 _function;
    Configuration                                           _cfg;
    std::shared_ptr<Plugin>                                 _plugin;
    std::atomic_int                                         _requestId = {0};
    InferenceEngine::ITaskExecutor*                         _executor = nullptr;
};
}  // namespace ArmPlugin
