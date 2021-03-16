// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <string>
#include <vector>
#include <algorithm>
#include <thread>

#include <ie_plugin_config.hpp>
#include <file_utils.h>
#include <cpp_interfaces/exception2status.hpp>

#include "arm_config.hpp"

using namespace ArmPlugin;

Configuration::Configuration() {
    _streamsExecutorConfig._streams = 1;
    _streamsExecutorConfig._threadsPerStream = std::thread::hardware_concurrency();
}

Configuration::Configuration(const ConfigMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;
    auto streamExecutorConfigKeys = _streamsExecutorConfig.SupportedKeys();
    for (auto&& c : config) {
        const auto& key = c.first;
        const auto& value = c.second;

        if (CONFIG_KEY(CPU_THROUGHPUT_STREAMS) == key) {
            _streamsExecutorConfig.SetConfig(CONFIG_KEY(CPU_THROUGHPUT_STREAMS), value);
        } else if (streamExecutorConfigKeys.end() !=
            std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            _streamsExecutorConfig.SetConfig(key, value);
        } else if (CONFIG_KEY(PERF_COUNT) == key) {
            _perfCount = (CONFIG_VALUE(YES) == value);
        } else if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) == key) {
            _exclusiveAsyncRequests = (CONFIG_VALUE(YES) == value);
        } else if (throwOnUnsupported) {
            THROW_IE_EXCEPTION << NOT_FOUND_str << ": " << key;
        }
    }
    if (_exclusiveAsyncRequests)
        _streamsExecutorConfig._streams = 1;
}

InferenceEngine::Parameter Configuration::Get(const std::string& name) const {
    if (name == CONFIG_KEY(PERF_COUNT)) {
        return {_perfCount ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO)};
    } else if (name == CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)) {
        return {_exclusiveAsyncRequests};
    } else if (name == CONFIG_KEY(CPU_THROUGHPUT_STREAMS)) {
        return {std::to_string(_streamsExecutorConfig._streams)};
    } else {
        THROW_IE_EXCEPTION << NOT_FOUND_str << ": " << name;
    }
}
