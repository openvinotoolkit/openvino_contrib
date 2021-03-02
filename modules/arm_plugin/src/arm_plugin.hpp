// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include <description_buffer.hpp>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

#include "arm_executable_network.hpp"
#include "arm_config.hpp"

namespace ArmPlugin {
struct Plugin : public InferenceEngine::InferencePluginInternal {
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin();

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::QueryNetworkResult
    QueryNetwork(const InferenceEngine::CNNNetwork& network,
                 const std::map<std::string, std::string>& config) const override;
    InferenceEngine::ExecutableNetworkInternal::Ptr
    LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                       const std::map<std::string, std::string>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const override;

    Configuration       _cfg;
};
}  // namespace ArmPlugin
