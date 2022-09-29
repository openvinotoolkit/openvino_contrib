// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov::nvidia_gpu::nodes {

/**
 * @brief Activation modes for fused convolutions.
 *
 * Mirrors the cuDNN cudnnActivationMode_t enum
 */
enum class ActivationMode { SIGMOID, RELU, TANH, CLIPPED_RELU, ELU, SWISH, NO_ACTIVATION };

}  // namespace ov::nvidia_gpu::nodes
