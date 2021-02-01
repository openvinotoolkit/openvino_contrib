// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ArmPlugin {
namespace opset {
enum class ActivationFunction {
    LOGISTIC,
    TANH,
    RELU,
    LU_BOUNDED_RELU,
    LEAKY_RELU,
    SOFT_RELU,
    ELU,
    ABS,
    SQRT,
    HARD_SWISH,
    IDENTITY,
};

struct ActivationInfo {
    ActivationFunction function = ActivationFunction::IDENTITY;
    float a = 0.0f;
    float b = 0.0f;
};
}  // namespace opset
}  // namespace ArmPlugin
