// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_config.hpp"
#include "openvino/runtime/internal_properties.hpp"

#include <fmt/format.h>

#include <error.hpp>
#include <regex>

#include "nvidia/properties.hpp"

using namespace ov::nvidia_gpu;

Configuration::Configuration() {}

std::vector<ov::PropertyName> Configuration::get_ro_properties() {
    static const std::vector<ov::PropertyName> ro_properties = {
        // Metrics
        ov::PropertyName{ov::available_devices.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::range_for_streams.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::range_for_async_infer_requests.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::device::architecture.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::device::capabilities.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::device::uuid.name(), ov::PropertyMutability::RO},
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
        ov::PropertyName{ov::nvidia_gpu::use_cuda_graph.name(), ov::PropertyMutability::RW},
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

std::vector<ov::PropertyName> Configuration::get_supported_internal_properties() {
    static const std::vector<ov::PropertyName> supported_internal_properties = {
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::config_device_id.name(), ov::PropertyMutability::WO},
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW}};
    return supported_internal_properties;
}

std::vector<ov::PropertyName> Configuration::get_caching_properties() {
    static const std::vector<ov::PropertyName> caching_properties = {
        ov::PropertyName{ov::device::architecture.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::hint::inference_precision.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), ov::PropertyMutability::RW}
    };
    return caching_properties;
}

void Configuration::update_device_id(const ov::AnyMap& config) {
    auto it = config.find(ov::device::id.name());
    if (it != config.end()) {
        auto value = it->second.as<std::string>();
        std::smatch match;
        std::regex re_device_id(R"((NVIDIA\.)?(\d+))");
        if (std::regex_match(value, match, re_device_id)) {
            const std::string device_id_prefix = match[1].str();
            const std::string device_id_value = match[2].str();
            if (!device_id_prefix.empty() && "NVIDIA." != device_id_prefix) {
                throw_ov_exception(
                    fmt::format("Prefix for deviceId should be 'NVIDIA.' (user deviceId = {}). "
                                "For example: NVIDIA.0, NVIDIA.1 and etc.",
                                value));
            }
            device_id = std::stoi(device_id_value);
            if (device_id < 0) {
                throw_ov_exception(fmt::format(
                    "Device ID {} is not supported. Index should be >= 0 (user index = {})", value, device_id));
            }
        } else {
            throw_ov_exception(
                fmt::format("Device ID {} is not supported. Supported deviceIds: 0, 1, 2, NVIDIA.0, NVIDIA.1, "
                            "NVIDIA.2 and etc.",
                            value));
        }
    }
}

ov::element::Type Configuration::get_inference_precision() const noexcept {
    return inference_precision;
}

bool Configuration::auto_streams_detection_required() const noexcept {
    if (exclusive_async_requests)
        return false;
    return ((ov::hint::PerformanceMode::THROUGHPUT == performance_mode) && (num_streams <= 0)) ||
            (num_streams == ov::streams::AUTO);
}

uint32_t Configuration::get_optimal_number_of_streams() const noexcept {
    // Default number for latency mode
    uint32_t optimal_number_of_streams = 1;
    if (auto_streams_detection_required()) {
        // If user is planning to use number of requests which is lower than reasonable range of streams
        // there is no sense to create more
        optimal_number_of_streams = (hint_num_requests > 0) ?
            std::min(hint_num_requests, reasonable_limit_of_streams)
            : reasonable_limit_of_streams;
    } else if (num_streams > 0 && !exclusive_async_requests) {
        optimal_number_of_streams = num_streams;
    }
    return optimal_number_of_streams;
}

bool Configuration::is_stream_executor_property(const std::string& name) const {
    auto stream_executor_properties = streams_executor_config_.get_property(
        ov::supported_properties.name()).as<std::vector<std::string>>();
    return (stream_executor_properties.end() !=
        std::find(std::begin(stream_executor_properties), std::end(stream_executor_properties), name));
}

bool Configuration::is_exclusive_async_requests() const noexcept {
    return exclusive_async_requests;
}

Configuration::Configuration(const ov::AnyMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;
    // Update device id first
    update_device_id(config);
    for (auto&& c : config) {
        const auto& key = c.first;
        const auto& value = c.second;

        if (ov::num_streams == key) {
            num_streams = value.as<ov::streams::Num>();
        } else if (ov::device::id == key) {
            // Device id is updated already
            continue;
        } else if (is_stream_executor_property(key)) {
            streams_executor_config_.set_property(key, value);
        } else if (ov::nvidia_gpu::operation_benchmark == key) {
            operation_benchmark = value.as<bool>();
        } else if (ov::nvidia_gpu::use_cuda_graph == key) {
            use_cuda_graph = value.as<bool>();
        } else if (ov::enable_profiling == key) {
            is_profiling_enabled = value.as<bool>();
        } else if (ov::hint::num_requests == key) {
            hint_num_requests = value.as<uint32_t>();
        } else if (ov::hint::inference_precision == key) {
            auto element_type = value.as<ov::element::Type>();
            const std::set<ov::element::Type> supported_types = {
                ov::element::f16, ov::element::f32,
            };
            if (supported_types.count(element_type) == 0) {
                throw_ov_exception(fmt::format("Inference precision {} is not supported by plugin", value.as<std::string>()));
            }
            inference_precision = element_type;
        } else if (ov::hint::performance_mode == key) {
            performance_mode = value.as<ov::hint::PerformanceMode>();
        } else if (ov::hint::execution_mode == key) {
            execution_mode = value.as<ov::hint::ExecutionMode>();
        } else if (ov::internal::exclusive_async_requests == key) {
            exclusive_async_requests = value.as<bool>();
        } else if (throwOnUnsupported) {
            throw_ov_exception(key);
        }
    }
}

ov::Any Configuration::get(const std::string& name) const {
    if (is_stream_executor_property(name)) {
        return streams_executor_config_.get_property(name);
    } else if (name == ov::device::id) {
        return {std::to_string(device_id)};
    } else if (name == ov::enable_profiling) {
        return is_profiling_enabled;
    } else if (name == ov::nvidia_gpu::operation_benchmark) {
        return operation_benchmark;
    } else if (name == ov::nvidia_gpu::use_cuda_graph) {
        return use_cuda_graph;
    } else if (name == ov::num_streams) {
        return (num_streams == 0) ?
            ov::streams::Num(get_optimal_number_of_streams()) : num_streams;
    } else if (name == ov::hint::num_requests) {
        return hint_num_requests;
    } else if (name == ov::hint::inference_precision) {
        return get_inference_precision();
    } else if (name == ov::hint::performance_mode) {
        return performance_mode;
    } else if (name == ov::hint::execution_mode) {
        return execution_mode;
    } else if (name == ov::internal::exclusive_async_requests) {
        return exclusive_async_requests;
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
}
