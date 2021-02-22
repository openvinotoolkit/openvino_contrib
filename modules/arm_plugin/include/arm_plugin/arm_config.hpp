// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for ARM plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file arm_config.hpp
 */

#pragma once

#include <string>
#include "ie_plugin_config.hpp"

namespace InferenceEngine {

namespace ArmConfigParams {

/**
 * @def ARM_CONFIG_KEY(name)
 * @brief Shortcut for defining Arm device configuration keys
 */
#define ARM_CONFIG_KEY(name) InferenceEngine::ArmConfigParams::_CONFIG_KEY(ARM_##name)

#define DECLARE_ARM_CONFIG_KEY(name) DECLARE_CONFIG_KEY(ARM_##name)
#define DECLARE_ARM_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(ARM_##name)

DECLARE_ARM_CONFIG_KEY(THROUGHPUT_STREAMS);

}  // namespace ArmConfigParams
}  // namespace InferenceEngine
