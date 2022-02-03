// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <string>
#include <vector>
#include <thread>

#include <ie_plugin_config.hpp>
#include <openvino/runtime/properties.hpp>

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
        } else if (ov::enable_profiling == key) {
            _perfCount = (CONFIG_VALUE(YES) == value);
        } else if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) == key) {
            _exclusiveAsyncRequests = (CONFIG_VALUE(YES) == value);
        } else if (CONFIG_KEY_INTERNAL(USE_REF_IMPL) == key) {
            _ref = (CONFIG_VALUE(YES) == value);
        } else if (CONFIG_KEY_INTERNAL(LP_TRANSFORMS_MODE) == key) {
            _lpt = (CONFIG_VALUE(YES) == value);
        } else if (CONFIG_KEY_INTERNAL(DUMP_GRAPH) == key) {
            _dump = (CONFIG_VALUE(YES) == value);
        }  else if (throwOnUnsupported) {
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
    } else if (ov::enable_profiling == name) {
        return _perfCount ? InferenceEngine::PluginConfigParams::YES : InferenceEngine::PluginConfigParams::NO;
    } else if (name == CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)) {
        return {_exclusiveAsyncRequests};
    } else if (name == CONFIG_KEY_INTERNAL(USE_REF_IMPL)) {
        return {_ref};
    } else if (name == CONFIG_KEY_INTERNAL(LP_TRANSFORMS_MODE)) {
        return {_lpt};
    } else if (name == CONFIG_KEY_INTERNAL(DUMP_GRAPH)) {
        return {_dump};
    }  else {
        IE_THROW(NotFound) << ": " << name;
    }
}
