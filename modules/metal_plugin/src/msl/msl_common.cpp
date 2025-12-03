// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msl/msl_common.hpp"

namespace ov {
namespace metal_plugin {
namespace msl {

std::string activation_expr(ActivationKind kind, float alpha) {
    switch (kind) {
        case ActivationKind::Relu:
            return "max(x, 0.0f)";
        case ActivationKind::Sigmoid:
            return "1.0f / (1.0f + exp(-x))";
        case ActivationKind::Tanh:
            return "tanh(x)";
        case ActivationKind::Elu:
            return "(x > 0.0f ? x : " + std::to_string(alpha) + "f * (exp(x) - 1.0f))";
        case ActivationKind::Prelu:
            return "(x > 0.0f ? x : " + std::to_string(alpha) + "f * x)";
        case ActivationKind::Gelu:
            return "(0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x))))";
        case ActivationKind::Swish:
            return "(x * (1.0f / (1.0f + exp(-x))))";
    }
    return "x";
}

}  // namespace msl
}  // namespace metal_plugin
}  // namespace ov
