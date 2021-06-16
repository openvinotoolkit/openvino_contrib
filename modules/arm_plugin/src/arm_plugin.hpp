// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

#include <cpp/ie_cnn_network.h>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

#include "arm_executable_network.hpp"
#include "arm_config.hpp"

namespace ArmPlugin {
struct Plugin : public InferenceEngine::IInferencePlugin {
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin();

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::QueryNetworkResult
    QueryNetwork(const InferenceEngine::CNNNetwork& network,
                 const std::map<std::string, std::string>& config) const override;
    InferenceEngine::IExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                       const std::map<std::string, std::string>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    std::shared_ptr<ngraph::Function> Transform(const std::shared_ptr<const ngraph::Function>& function,
                                                const Configuration& config) const;

    Configuration _cfg;
};
}  // namespace ArmPlugin
