// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {
namespace {

std::string activation_expr(ActivationKind kind, float alpha, double clamp_min, double clamp_max) {
    switch (kind) {
        case ActivationKind::Relu: return "max(x, 0.0f)";
        case ActivationKind::Sigmoid: return "1.0f / (1.0f + exp(-x))";
        case ActivationKind::Tanh: return "tanh(x)";
        case ActivationKind::Elu: return "(x > 0.0f) ? x : (exp(x) - 1.0f) * " + std::to_string(alpha);
        case ActivationKind::Prelu: return "(x >= 0.0f) ? x : x * " + std::to_string(alpha);
        case ActivationKind::Gelu: return "0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)))";
        case ActivationKind::Swish: return "x / (1.0f + exp(-x))";
        case ActivationKind::HSwish: return "x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        case ActivationKind::HSigmoid: return "clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        case ActivationKind::SoftPlus: return "log(1.0f + exp(x))";
        case ActivationKind::Mish: return "x * tanh(log(1.0f + exp(x)))";
        case ActivationKind::SoftSign: return "x / (1.0f + fabs(x))";
        case ActivationKind::Abs: return "fabs(x)";
        case ActivationKind::Sign: return "(x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f))";
        case ActivationKind::Clamp:
            return "clamp(x, " + std::to_string(static_cast<float>(clamp_min)) + "f, " +
                   std::to_string(static_cast<float>(clamp_max)) + "f)";
        case ActivationKind::Exp: return "exp(x)";
        case ActivationKind::Log: return "log(x)";
        case ActivationKind::Sqrt: return "sqrt(x)";
        case ActivationKind::Floor: return "floor(x)";
        case ActivationKind::Ceil: return "ceil(x)";
        case ActivationKind::Negative: return "-x";
        case ActivationKind::Sin: return "sin(x)";
        case ActivationKind::Cos: return "cos(x)";
        case ActivationKind::Tan: return "tan(x)";
        case ActivationKind::Erf: return "erf(x)";
        case ActivationKind::Asin: return "asin(x)";
        case ActivationKind::Acos: return "acos(x)";
        case ActivationKind::Atan: return "atan(x)";
        case ActivationKind::Asinh: return "asinh(x)";
        case ActivationKind::Acosh: return "acosh(x)";
        case ActivationKind::Atanh: return "atanh(x)";
        case ActivationKind::Sinh: return "sinh(x)";
        case ActivationKind::Cosh: return "cosh(x)";
        case ActivationKind::RoundEven: return "rint(x)";
        case ActivationKind::RoundAway: return "round(x)";
        case ActivationKind::Identity:
        default: return "x";
    }
}

}  // namespace

std::string generate_msl_for_unary(const UnaryCodegenDesc& d, mlir::ModuleOp module) {
    std::ostringstream ss;
    std::string scalar = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar = msl_type_from_mlir(ft.getInput(0));
        }
    } else {
        scalar = (d.element_type == ov::element::f16) ? "half" : "float";
    }
    const bool is_int_scalar = (scalar != "float" && scalar != "half");
    const bool is_bool = (scalar == "bool");
    const bool is_unsigned = (scalar == "uchar" || scalar == "ushort" || scalar == "uint" ||
                              scalar == "ulong" || scalar == "bool");
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "kernel void unary_kernel(\n";
    ss << "  device const " << scalar << "* in0 [[buffer(0)]],\n";
    ss << "  device " << scalar << "* out [[buffer(1)]],\n";
    ss << "  constant uint& NUM_ELEMS [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    if (is_int_scalar &&
        (d.activation == ActivationKind::Sign ||
         d.activation == ActivationKind::Abs ||
         d.activation == ActivationKind::LogicalNot)) {
        ss << "    " << scalar << " x = in0[gid];\n";
        if (d.activation == ActivationKind::LogicalNot) {
            if (is_bool) {
                ss << "    out[gid] = !x;\n";
            } else {
                ss << "    out[gid] = (x == 0) ? (" << scalar << ")1 : (" << scalar << ")0;\n";
            }
        } else if (d.activation == ActivationKind::Abs) {
            if (is_unsigned) {
                ss << "    out[gid] = x;\n";
            } else {
                ss << "    out[gid] = (x < 0) ? (" << scalar << ")(-x) : x;\n";
            }
        } else if (d.activation == ActivationKind::Sign) {
            if (is_unsigned) {
                ss << "    out[gid] = (x > 0) ? (" << scalar << ")1 : (" << scalar << ")0;\n";
            } else {
                ss << "    out[gid] = (x > 0) ? (" << scalar << ")1 : ((x < 0) ? (" << scalar << ")(-1) : (" << scalar << ")0);\n";
            }
        }
    } else {
        ss << "    float x = static_cast<float>(in0[gid]);\n";
        ss << "    out[gid] = " << activation_expr(d.activation, d.alpha, d.clamp_min, d.clamp_max) << ";\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
