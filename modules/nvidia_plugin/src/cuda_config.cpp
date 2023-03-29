// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_config.hpp"

#include <fmt/format.h>

#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <error.hpp>
#include <regex>

#include "nvidia/properties.hpp"

using namespace ov::nvidia_gpu;

Configuration::Configuration() {}

std::vector<ov::PropertyName> Configuration::get_ro_properties() {
    static const std::vector<ov::PropertyName> ro_properties = {
        // Metrics
        ov::PropertyName{ov::available_devices.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::caching_properties.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::range_for_streams.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::range_for_async_infer_requests.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::device::architecture.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::device::capabilities.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::device::uuid.name(), ov::PropertyMutability::RO}
    };
    return ro_properties;
}

std::vector<ov::PropertyName> Configuration::get_rw_properties() {
    static const std::vector<ov::PropertyName> rw_properties = {
        // Configs
        ov::PropertyName{ov::device::id.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::hint::inference_precision.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::num_streams.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::hint::num_requests.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::hint::performance_mode.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::enable_profiling.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::nvidia_gpu::operation_benchmark.name(), ov::PropertyMutability::RW},
    };
    return rw_properties;
}

bool Configuration::is_rw_property(const std::string& name) {
    auto supported_rw_properties = get_rw_properties();
    return (std::find(supported_rw_properties.begin(),
                        supported_rw_properties.end(), name) != supported_rw_properties.end());
};

std::vector<ov::PropertyName> Configuration::get_supported_properties() {
    const std::vector<ov::PropertyName> ro_properties = get_ro_properties();
    const std::vector<ov::PropertyName> rw_properties = get_rw_properties();
    std::vector<ov::PropertyName> supported_properties(std::begin(ro_properties), std::end(ro_properties));
    supported_properties.insert(std::end(supported_properties), std::begin(rw_properties), std::end(rw_properties));
    return supported_properties;
}

std::vector<ov::PropertyName> Configuration::get_caching_properties() {
    static const std::vector<ov::PropertyName> caching_properties = {
        ov::PropertyName{ov::device::architecture.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::hint::inference_precision.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), ov::PropertyMutability::RW}
    };
    return caching_properties;
}

void Configuration::update_device_id(const ConfigMap& config) {
    auto it = config.find(ov::device::id.name());
    if (it == config.end())
        it = config.find(CONFIG_KEY(DEVICE_ID));
    if (it != config.end()) {
        auto value = it->second;
        std::smatch match;
        std::regex re_device_id(R"((NVIDIA\.)?(\d+))");
        if (std::regex_match(value, match, re_device_id)) {
            const std::string device_id_prefix = match[1].str();
            const std::string device_id_value = match[2].str();
            if (!device_id_prefix.empty() && "NVIDIA." != device_id_prefix) {
                throwIEException(
                    fmt::format("Prefix for deviceId should be 'NVIDIA.' (user deviceId = {}). "
                                "For example: NVIDIA.0, NVIDIA.1 and etc.",
                                value));
            }
            deviceId = std::stoi(device_id_value);
            if (deviceId < 0) {
                throwIEException(fmt::format(
                    "Device ID {} is not supported. Index should be >= 0 (user index = {})", value, deviceId));
            }
        } else {
            throwIEException(
                fmt::format("Device ID {} is not supported. Supported deviceIds: 0, 1, 2, NVIDIA.0, NVIDIA.1, "
                            "NVIDIA.2 and etc.",
                            value));
        }
    }
}

ov::element::Type Configuration::get_inference_precision() const noexcept {
    return inference_precision;
    /*
    Uncomment this code to switch to f16 by default
    if (inference_precision != ov::element::undefined)
        return inference_precision;
    if (execution_mode == ov::hint::ExecutionMode::PERFORMANCE) {
        if (isHalfSupported(CUDA::Device(deviceId))) {
            return ov::element::f16;
        }
    }
    return ov::element::f32; */
}

uint32_t Configuration::get_optimal_number_of_streams() const noexcept {
    // Default number for latency mode
    uint32_t optimal_number_of_streams = 1;
    if (ov::hint::PerformanceMode::THROUGHPUT == performance_mode) {
        // If user is planning to use number of requests which is lower than reasonable range of streams
        // there is no sense to create more
        optimal_number_of_streams = (hint_num_requests > 0) ?
            std::min(hint_num_requests, reasonable_limit_of_streams)
            : reasonable_limit_of_streams;
    }
    if (num_streams > 0) {
        optimal_number_of_streams = num_streams;
    }
    return optimal_number_of_streams;
}

Configuration::Configuration(const ConfigMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;
    // If plugin needs to use InferenceEngine::StreamsExecutor it should be able to process its configuration
    auto streamExecutorConfigKeys = streams_executor_config_.SupportedKeys();
    // Update device id first
    update_device_id(config);
    for (auto&& c : config) {
        const auto& key = c.first;
        const auto& value = c.second;

        if (ov::num_streams == key) {
            num_streams = ov::util::from_string(value, ov::num_streams);
        } if (NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS) == key) {
            if (value != NVIDIA_CONFIG_VALUE(THROUGHPUT_AUTO)) {
                try {
                    num_streams = ov::streams::Num(std::stoi(value));
                } catch (...) {
                    throwIEException(
                        fmt::format("NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS) = {} "
                                    "is not a number !!",
                                    value));
                }
            } else {
                num_streams = ov::streams::AUTO;
            }
        } else if (ov::device::id == key || CONFIG_KEY(DEVICE_ID) == key) {
            // Device id is updated already
            continue;
        } else if (streamExecutorConfigKeys.end() !=
                   std::find(std::begin(streamExecutorConfigKeys), std::end(streamExecutorConfigKeys), key)) {
            streams_executor_config_.SetConfig(key, value);
        } else if (ov::nvidia_gpu::operation_benchmark == key || NVIDIA_CONFIG_KEY(OPERATION_BENCHMARK) == key) {
            operation_benchmark = ov::util::from_string(value, ov::nvidia_gpu::operation_benchmark);
        } else if (ov::enable_profiling == key) {
            is_profiling_enabled = ov::util::from_string(value, ov::enable_profiling);
        } else if (ov::hint::num_requests == key) {
            hint_num_requests = ov::util::from_string(value, ov::hint::num_requests);
        } else if (ov::hint::inference_precision == key) {
            auto element_type = ov::util::from_string(value, ov::hint::inference_precision);
            const std::set<ov::element::Type> supported_types = {
                ov::element::undefined, ov::element::f16, ov::element::f32,
            };
            if (supported_types.count(element_type) == 0) {
                throwIEException(fmt::format("Inference precision {} is not supported by plugin", value));
            }
            inference_precision = element_type;
        } else if (ov::hint::performance_mode == key) {
            performance_mode = ov::util::from_string(value, ov::hint::performance_mode);
        } else if (ov::hint::execution_mode == key) {
            execution_mode = ov::util::from_string(value, ov::hint::execution_mode);
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
    } else if (name == ov::device::id || name == CONFIG_KEY(DEVICE_ID)) {
        return {std::to_string(deviceId)};
    } else if (name == ov::enable_profiling || name == CONFIG_KEY(PERF_COUNT)) {
        return is_profiling_enabled;
    } else if (name == ov::nvidia_gpu::operation_benchmark || name == NVIDIA_CONFIG_KEY(OPERATION_BENCHMARK)) {
        return operation_benchmark;
    } else if (name == ov::num_streams) {
        return (num_streams == 0) ?
            ov::streams::Num(get_optimal_number_of_streams()) : num_streams;
    } else if (name == NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS)) {
        auto value = (num_streams == 0) ?
            ov::streams::Num(get_optimal_number_of_streams()) : num_streams;
        return (value ==  ov::streams::AUTO) ? NVIDIA_CONFIG_VALUE(THROUGHPUT_AUTO)
                                             : ov::util::to_string(value);
    } else if (name == ov::hint::num_requests) {
        return hint_num_requests;
    } else if (name == ov::hint::inference_precision) {
        return get_inference_precision();
    } else if (name == ov::hint::performance_mode) {
        return performance_mode;
    } else if (name == ov::hint::execution_mode) {
        return execution_mode;
    } else {
        throwNotFound(name);
    }
}
