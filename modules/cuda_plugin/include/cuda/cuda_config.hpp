// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for DLIA plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file dlia_config.hpp
 */

#pragma once

#include <string>
#include "ie_plugin_config.hpp"

namespace InferenceEngine {

namespace CUDAMetrics {

/**
 * @def CUDA_METRIC_VALUE(name)
 * @brief Shortcut for defining Template metric values
 */
#define CUDA_METRIC_VALUE(name) InferenceEngine::CUDAMetrics::name
#define DECLARE_CUDA_METRIC_VALUE(name) static constexpr auto name = #name

// ! [public_header:metrics]
/**
 * @brief Defines whether current Template device instance supports hardware blocks for fast convolution computations.
 */
DECLARE_CUDA_METRIC_VALUE(HARDWARE_CONVOLUTION);
// ! [public_header:metrics]

}  // namespace CUDAMetrics

namespace CUDAConfigParams {

/**
 * @def CUDA_CONFIG_KEY(name)
 * @brief Shortcut for defining Template device configuration keys
 */
#define CUDA_CONFIG_KEY(name) InferenceEngine::CUDAConfigParams::_CONFIG_KEY(CUDA_##name)

#define DECLARE_CUDA_CONFIG_KEY(name) DECLARE_CONFIG_KEY(CUDA_##name)
#define DECLARE_CUDA_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(CUDA_##name)


/**
 * @brief Defines the number of throutput streams used by CUDA plugin.
 */
DECLARE_CUDA_CONFIG_KEY(THROUGHPUT_STREAMS);


}  // namespace CUDAConfigParams
}  // namespace InferenceEngine
