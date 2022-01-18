// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>

#include <ie_parameter.hpp>
#include <threading/ie_istreams_executor.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>

namespace InferenceEngine {
namespace PluginConfigInternalParams {
DECLARE_CONFIG_KEY(USE_REF_IMPL);
DECLARE_CONFIG_KEY(DUMP_GRAPH);
}  // namespace PluginConfigInternalParams
}  // namespace InferenceEngine

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
    bool _ref                    = true;
    bool _lpt                    = true;
    bool _dump                   = false;
    mutable InferenceEngine::IStreamsExecutor::Config _streamsExecutorConfig;
};
}  //  namespace ArmPlugin
