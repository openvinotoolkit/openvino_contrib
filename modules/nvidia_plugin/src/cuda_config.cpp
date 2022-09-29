// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_config.hpp"

#include <fmt/format.h>

#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <error.hpp>

using namespace ov::nvidia_gpu;

Configuration::Configuration() {}

Configuration::Configuration(const ConfigMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;
    // If plugin needs to use InferenceEngine::StreamsExecutor it should be able to process its configuration
    auto streamExecutorConfigKeys = streams_executor_config_.SupportedKeys();
    for (auto&& c : config) {
        const auto& key = c.first;
        const auto& value = c.second;

        if (NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS) == key) {
            if (value != NVIDIA_CONFIG_VALUE(THROUGHPUT_AUTO)) {
                try {
                    std::stoi(value);
                } catch (...) {
                    throwIEException(
                        fmt::format("NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS) = {} "
                                    "is not a number !!",
                                    value));
                }
            }
            cuda_throughput_streams_ = value;
        } else if (CONFIG_KEY(CPU_THROUGHPUT_STREAMS) == key) {
            streams_executor_config_.SetConfig(CONFIG_KEY(CPU_THROUGHPUT_STREAMS), value);
        } else if (streamExecutorConfigKeys.end() !=
                   std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            streams_executor_config_.SetConfig(key, value);
        } else if (CONFIG_KEY(DEVICE_ID) == key) {
            deviceId = std::stoi(value);
            if (deviceId > 0) {
                throwIEException(fmt::format("Device ID {} is not supported", deviceId));
            }
        } else if (NVIDIA_CONFIG_KEY(OPERATION_BENCHMARK) == key) {
            if (value == NVIDIA_CONFIG_VALUE(YES)) {
                operation_benchmark = true;
            } else if (value == NVIDIA_CONFIG_VALUE(NO)) {
                operation_benchmark = false;
            } else {
                throwIEException(fmt::format("operation benchmark option value {} is not supported", value));
            }
        } else if (NVIDIA_CONFIG_KEY(DISABLE_TENSORITERATOR_TRANSFORM) == key) {
            if (value == NVIDIA_CONFIG_VALUE(YES)) {
                disabled_tensoriterator_transform = true;
            } else if (value == NVIDIA_CONFIG_VALUE(NO)) {
                disabled_tensoriterator_transform = false;
            } else {
                throwIEException(fmt::format("disabled_transformations option value {} is not supported", value));
            }
        } else if (CONFIG_KEY(PERF_COUNT) == key) {
            perfCount = (CONFIG_VALUE(YES) == value);
        } else if (ov::hint::performance_mode == key) {
            std::stringstream strm{value};
            strm >> performance_mode;
        } else if (throwOnUnsupported) {
            throwNotFound(key);
        }
    }
}

InferenceEngine::Parameter Configuration::Get(const std::string& name) const {
    auto streamExecutorConfigKeys = streams_executor_config_.SupportedKeys();
    if ((streamExecutorConfigKeys.end() !=
         std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), name))) {
        return streams_executor_config_.GetConfig(name);
    } else if (name == CONFIG_KEY(DEVICE_ID)) {
        return {std::to_string(deviceId)};
    } else if (name == CONFIG_KEY(PERF_COUNT)) {
        return {perfCount};
    } else if (name == NVIDIA_CONFIG_KEY(OPERATION_BENCHMARK)) {
        return {std::string(operation_benchmark ? NVIDIA_CONFIG_VALUE(YES) : NVIDIA_CONFIG_VALUE(NO))};
    } else if (name == NVIDIA_CONFIG_KEY(DISABLE_TENSORITERATOR_TRANSFORM)) {
        return {std::string(disabled_tensoriterator_transform ? NVIDIA_CONFIG_VALUE(YES) : NVIDIA_CONFIG_VALUE(NO))};
    } else if (name == NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS)) {
        return {cuda_throughput_streams_};
    } else if (name == CONFIG_KEY(CPU_THROUGHPUT_STREAMS)) {
        return {std::to_string(streams_executor_config_._streams)};
    } else if (name == CONFIG_KEY(CPU_BIND_THREAD)) {
        return const_cast<InferenceEngine::IStreamsExecutor::Config&>(streams_executor_config_).GetConfig(name);
    } else if (name == CONFIG_KEY(CPU_THREADS_NUM)) {
        return {std::to_string(streams_executor_config_._threads)};
    } else if (name == CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)) {
        return {std::to_string(streams_executor_config_._threadsPerStream)};
    } else if (name == ov::num_streams) {
        return {std::to_string(streams_executor_config_._streams)};
    } else if (name == ov::inference_num_threads) {
        return {std::to_string(streams_executor_config_._threads)};
        // TODO: Refactoring
        //    } else if (name == ov::affinity) {
        //        return {std::to_string(streams_executor_config_._threadPreferredCoreType)};
    } else if (name == ov::hint::performance_mode) {
        return performance_mode;
    } else {
        throwNotFound(name);
    }
}
