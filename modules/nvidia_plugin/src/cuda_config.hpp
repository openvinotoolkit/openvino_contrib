// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_parameter.hpp>
#include <map>
#include <memory>
#include <nvidia/nvidia_config.hpp>
#include <openvino/runtime/properties.hpp>
#include <string>
#include <threading/ie_istreams_executor.hpp>

namespace ov {
namespace nvidia_gpu {

using ConfigMap = std::map<std::string, std::string>;

struct Configuration {
    using Ptr = std::shared_ptr<Configuration>;

    Configuration();
    Configuration(const Configuration&) = default;
    Configuration(Configuration&&) = default;
    Configuration& operator=(const Configuration&) = default;
    Configuration& operator=(Configuration&&) = default;

    explicit Configuration(const ConfigMap& config,
                           const Configuration& defaultCfg = {},
                           const bool throwOnUnsupported = true);

    InferenceEngine::Parameter Get(const std::string& name) const;

    static std::vector<ov::PropertyName> get_supported_properties();
    static std::vector<ov::PropertyName> get_ro_properties();
    static std::vector<ov::PropertyName> get_rw_properties();
    static std::vector<ov::PropertyName> get_caching_properties();
    static bool is_rw_property(const std::string& name);
    void update_device_id(const ConfigMap& config);
    ov::element::Type get_inference_precision() const noexcept;
    uint32_t get_optimal_number_of_streams() const noexcept;

    // Plugin configuration parameters
    static constexpr uint32_t reasonable_limit_of_streams = 10;
    int deviceId = 0;
    InferenceEngine::IStreamsExecutor::Config streams_executor_config_;
private:
    bool is_profiling_enabled = false;
    bool operation_benchmark = false;
    uint32_t hint_num_requests = 0;
    ov::streams::Num num_streams = 0;
    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::UNDEFINED;
    ov::hint::ExecutionMode execution_mode = ov::hint::ExecutionMode::UNDEFINED;
    ov::element::Type inference_precision = ov::element::undefined;
};

}  // namespace nvidia_gpu
}  // namespace ov
