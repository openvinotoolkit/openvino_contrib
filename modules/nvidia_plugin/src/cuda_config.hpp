// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace ov {
namespace nvidia_gpu {

struct Configuration {
    using Ptr = std::shared_ptr<Configuration>;

    Configuration();
    Configuration(const Configuration&) = default;
    Configuration(Configuration&&) = default;
    Configuration& operator=(const Configuration&) = default;
    Configuration& operator=(Configuration&&) = default;

    explicit Configuration(const ov::AnyMap& params,
                           const Configuration& defaultCfg = {},
                           const bool throwOnUnsupported = true);

    ov::Any get(const std::string& name) const;

    static std::vector<ov::PropertyName> get_supported_properties();
    static std::vector<ov::PropertyName> get_supported_internal_properties();
    static std::vector<ov::PropertyName> get_ro_properties();
    static std::vector<ov::PropertyName> get_rw_properties();
    static std::vector<ov::PropertyName> get_caching_properties();
    static bool is_rw_property(const std::string& name);
    bool is_stream_executor_property(const std::string& name) const;
    void update_device_id(const ov::AnyMap& config);
    int get_device_id() const { return device_id; };
    ov::element::Type get_inference_precision() const noexcept;
    uint32_t get_optimal_number_of_streams() const noexcept;
    bool auto_streams_detection_required() const noexcept;
    bool is_exclusive_async_requests() const noexcept;

    // Plugin configuration parameters
    static constexpr uint32_t reasonable_limit_of_streams = 10;
    ov::threading::IStreamsExecutor::Config streams_executor_config_;

private:
    int device_id = 0;
    bool is_profiling_enabled = false;
    bool operation_benchmark = false;
    bool use_cuda_graph = true;
    bool exclusive_async_requests = false;
    uint32_t hint_num_requests = 0;
    ov::streams::Num num_streams = 0;
    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::LATENCY;
    ov::hint::ExecutionMode execution_mode = ov::hint::ExecutionMode::PERFORMANCE;
    ov::element::Type inference_precision = ov::element::dynamic;
};

}  // namespace nvidia_gpu
}  // namespace ov
