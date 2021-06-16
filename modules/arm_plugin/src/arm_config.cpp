// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <string>
#include <vector>
#include <thread>

#include <ie_plugin_config.hpp>

#include "arm_config.hpp"

using namespace ArmPlugin;

Configuration::Configuration() {
    _streamsExecutorConfig._streams = 1;
    _streamsExecutorConfig._threadsPerStream = std::thread::hardware_concurrency();
    _streamsExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::NONE;
}

Configuration::Configuration(const ConfigMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;
    auto streamExecutorConfigKeys = _streamsExecutorConfig.SupportedKeys();
    for (auto&& c : config) {
        const auto& key = c.first;
        const auto& value = c.second;
        if ((streamExecutorConfigKeys.end() !=
             std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key))) {
            _streamsExecutorConfig.SetConfig(key, value);
            _streamsExecutorConfig._threadBindingType = InferenceEngine::IStreamsExecutor::NONE;
        } else if (CONFIG_KEY(PERF_COUNT) == key) {
            _perfCount = (CONFIG_VALUE(YES) == value);
        } else if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) == key) {
            _exclusiveAsyncRequests = (CONFIG_VALUE(YES) == value);
        } else if (CONFIG_KEY_INTERNAL(LP_TRANSFORMS_MODE) == key) {
            _lpt = (CONFIG_VALUE(YES) == value);
        } else if (throwOnUnsupported) {
            IE_THROW(NotFound) << ": " << key;
        }
    }
    if (_exclusiveAsyncRequests)
        _streamsExecutorConfig._streams = 1;
}

InferenceEngine::Parameter Configuration::Get(const std::string& name) const {
    auto streamExecutorConfigKeys = _streamsExecutorConfig.SupportedKeys();
    if ((streamExecutorConfigKeys.end() !=
             std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), name))) {
        return _streamsExecutorConfig.GetConfig(name);
    } else if (name == CONFIG_KEY(PERF_COUNT)) {
        return {_perfCount ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO)};
    } else if (name == CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)) {
        return {_exclusiveAsyncRequests};
    } else if (name == CONFIG_KEY_INTERNAL(LP_TRANSFORMS_MODE)) {
        return {_lpt};
    } else {
        IE_THROW(NotFound) << ": " << name;
    }
}
