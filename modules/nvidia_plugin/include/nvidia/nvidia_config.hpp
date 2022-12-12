// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for DLIA plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file nvidia_config.hpp
 */

#pragma once

#include <string>

#include "ie_plugin_config.hpp"

namespace InferenceEngine {

namespace CUDAMetrics {

/**
 * @def NVIDIA_METRIC_VALUE(name)
 * @brief Shortcut for defining Template metric values
 */
#define NVIDIA_METRIC_VALUE(name) InferenceEngine::CUDAMetrics::name
#define DECLARE_NVIDIA_METRIC_VALUE(name) static constexpr auto name = #name

// ! [public_header:metrics]
/**
 * @brief Defines whether current Template device instance supports hardware blocks for fast convolution computations.
 */
DECLARE_NVIDIA_METRIC_VALUE(HARDWARE_CONVOLUTION);
// ! [public_header:metrics]

}  // namespace CUDAMetrics

namespace CUDAConfigParams {

/**
 * @def NVIDIA_CONFIG_KEY(name)
 * @brief Shortcut for defining Template device configuration keys
 */
#define NVIDIA_CONFIG_KEY(name) InferenceEngine::CUDAConfigParams::_CONFIG_KEY(NVIDIA_##name)
#define NVIDIA_CONFIG_VALUE(name) InferenceEngine::CUDAConfigParams::NVIDIA_##name

#define DECLARE_NVIDIA_CONFIG_KEY(name) DECLARE_CONFIG_KEY(NVIDIA_##name)
#define DECLARE_NVIDIA_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(NVIDIA_##name)

DECLARE_NVIDIA_CONFIG_VALUE(YES);
DECLARE_NVIDIA_CONFIG_VALUE(NO);

/**
 * @brief Defines the number of throutput streams used by NVIDIA GPU plugin.
 */
DECLARE_NVIDIA_CONFIG_VALUE(THROUGHPUT_AUTO);
DECLARE_NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS);

/**
 * @brief Defines if optimization should be run for CUDA libraries ("NVIDIA_YES", "NVIDIA_NO" - default).
 */
DECLARE_NVIDIA_CONFIG_KEY(OPERATION_BENCHMARK);

}  // namespace CUDAConfigParams
}  // namespace InferenceEngine
