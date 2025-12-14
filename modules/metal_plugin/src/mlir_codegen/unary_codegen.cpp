// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace metal_plugin {
namespace {

std::string activation_expr(ActivationKind kind, float alpha) {
    switch (kind) {
        case ActivationKind::Relu: return "max(x, 0.0f)";
        case ActivationKind::Sigmoid: return "1.0f / (1.0f + exp(-x))";
        case ActivationKind::Tanh: return "tanh(x)";
        case ActivationKind::Elu: return "(x > 0.0f) ? x : (exp(x) - 1.0f) * " + std::to_string(alpha);
        case ActivationKind::Prelu: return "(x >= 0.0f) ? x : x * " + std::to_string(alpha);
        case ActivationKind::Gelu: return "0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)))";
        case ActivationKind::Swish: return "x / (1.0f + exp(-x))";
        case ActivationKind::Abs: return "fabs(x)";
        case ActivationKind::Sign: return "(x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f))";
        case ActivationKind::Clamp: return "clamp(x, " + std::to_string(alpha) + ", " + std::to_string(alpha) + ")";
        case ActivationKind::Identity:
        default: return "x";
    }
}

}  // namespace

std::string generate_msl_for_unary(const UnaryCodegenDesc& d) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "kernel void unary_kernel(\n";
    ss << "  device const float* in0 [[buffer(0)]],\n";
    ss << "  device float* out [[buffer(1)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    float x = in0[gid];\n";
    ss << "    out[gid] = " << activation_expr(d.activation, d.alpha) << ";\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
