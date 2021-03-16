// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include <map>
#include <unordered_map>

#include <ie_parameter.hpp>
#include <threading/ie_istreams_executor.hpp>

namespace ArmPlugin {
using ConfigMap = std::map<std::string, std::string>;
struct Configuration {
    Configuration();
    Configuration(const Configuration&)             = default;
    Configuration(Configuration&&)                  = default;
    Configuration& operator=(const Configuration&)  = default;
    Configuration& operator=(Configuration&&)       = default;

    explicit Configuration(const ConfigMap& config,
                           const Configuration& defaultCfg = {},
                           const bool throwOnUnsupported = true);

    InferenceEngine::Parameter Get(const std::string& name) const;

    // Plugin configuration parameters

    bool _exclusiveAsyncRequests = false;
    bool _perfCount              = true;
    InferenceEngine::IStreamsExecutor::Config _streamsExecutorConfig;
};
}  //  namespace ArmPlugin
