// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

#include "mlir/msl_codegen_apple_msl_common.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool uses_msl_erf_approximation(ActivationKind kind,
                                bool gelu_tanh_approximation) {
  return kind == ActivationKind::Erf ||
         (kind == ActivationKind::Gelu && !gelu_tanh_approximation);
}

std::string msl_erf_approximation_function() {
  return R"(inline float gfx_msl_erf_approx(float x) {
    const float sign_value = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.5f * ax);
    const float tau = t * precise::exp(
        -ax * ax - 1.26551223f +
        t * (1.00002368f +
        t * (0.37409196f +
        t * (0.09678418f +
        t * (-0.18628806f +
        t * (0.27886807f +
        t * (-1.13520398f +
        t * (1.48851587f +
        t * (-0.82215223f +
        t * 0.17087277f)))))))));
    return sign_value * (1.0f - tau);
}
)";
}

std::string activation_expr(ActivationKind kind, float alpha, double clamp_min,
                            double clamp_max, bool gelu_tanh_approximation,
                            const std::string &swish_beta_expr) {
  switch (kind) {
  case ActivationKind::Relu:
    return "max(x, 0.0f)";
  case ActivationKind::Sigmoid:
    return "1.0f / (1.0f + precise::exp(-x))";
  case ActivationKind::Tanh:
    return msl_stable_tanh_expr("x");
  case ActivationKind::Elu:
    return "(x > 0.0f) ? x : (exp(x) - 1.0f) * " + std::to_string(alpha);
  case ActivationKind::Prelu:
    return "(x >= 0.0f) ? x : x * " + std::to_string(alpha);
  case ActivationKind::Gelu:
    return gelu_tanh_approximation
               ? msl_stable_gelu_tanh_expr("x")
               : "0.5f * x * (1.0f + gfx_msl_erf_approx(x * 0.70710678118f))";
  case ActivationKind::Swish:
    return "x / (1.0f + precise::exp(-(" + swish_beta_expr + " * x)))";
  case ActivationKind::HSwish:
    return "x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
  case ActivationKind::HSigmoid:
    return "clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
  case ActivationKind::SoftPlus:
    return "log(1.0f + exp(x))";
  case ActivationKind::Mish:
    return "x * " + msl_stable_tanh_expr("log(1.0f + exp(x))");
  case ActivationKind::SoftSign:
    return "x / (1.0f + fabs(x))";
  case ActivationKind::Abs:
    return "fabs(x)";
  case ActivationKind::Sign:
    return "(x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f))";
  case ActivationKind::Clamp:
    return "clamp(x, " + std::to_string(static_cast<float>(clamp_min)) + "f, " +
           std::to_string(static_cast<float>(clamp_max)) + "f)";
  case ActivationKind::Exp:
    return "exp(x)";
  case ActivationKind::Log:
    return "log(x)";
  case ActivationKind::Sqrt:
    return "sqrt(x)";
  case ActivationKind::Floor:
    return "floor(x)";
  case ActivationKind::Ceil:
    return "ceil(x)";
  case ActivationKind::Negative:
    return "-x";
  case ActivationKind::Sin:
    return "sin(x)";
  case ActivationKind::Cos:
    return "cos(x)";
  case ActivationKind::Tan:
    return "tan(x)";
  case ActivationKind::Erf:
    return "gfx_msl_erf_approx(x)";
  case ActivationKind::Asin:
    return "asin(x)";
  case ActivationKind::Acos:
    return "acos(x)";
  case ActivationKind::Atan:
    return "atan(x)";
  case ActivationKind::Asinh:
    return "asinh(x)";
  case ActivationKind::Acosh:
    return "acosh(x)";
  case ActivationKind::Atanh:
    return "atanh(x)";
  case ActivationKind::Sinh:
    return "sinh(x)";
  case ActivationKind::Cosh:
    return "cosh(x)";
  case ActivationKind::RoundEven:
    return "rint(x)";
  case ActivationKind::RoundAway:
    return "round(x)";
  case ActivationKind::Identity:
  default:
    return "x";
  }
}

} // namespace

std::string generate_msl_for_unary(const UnaryCodegenDesc &d,
                                   mlir::ModuleOp module) {
  std::ostringstream ss;
  std::string scalar = (d.element_type == ov::element::f16) ? "half" : "float";
  if (module) {
    auto func = get_entry_func(module);
    if (func) {
      auto ft = func.getFunctionType();
      if (ft.getNumInputs() >= 1) {
        scalar = msl_type_from_mlir(ft.getInput(0));
      }
    }
  }
  const bool is_bool = (scalar == "bool");
  if (is_bool) {
    scalar = "uchar";
  }
  const bool is_int_scalar = (scalar != "float" && scalar != "half");
  const bool is_unsigned = (scalar == "uchar" || scalar == "ushort" ||
                            scalar == "uint" || scalar == "ulong" || is_bool);
  const bool has_runtime_swish_beta =
      d.activation == ActivationKind::Swish && d.swish_beta_runtime_input;
  const uint32_t out_buffer_index = has_runtime_swish_beta ? 2u : 1u;
  const uint32_t count_buffer_index = has_runtime_swish_beta ? 3u : 2u;
  ss << "#include <metal_stdlib>\n";
  ss << "using namespace metal;\n";
  if (uses_msl_erf_approximation(d.activation, d.gelu_tanh_approximation)) {
    ss << msl_erf_approximation_function();
  }
  ss << "kernel void " << d.entry_point << "(\n";
  ss << "  device const " << scalar << "* in0 [[buffer(0)]],\n";
  if (has_runtime_swish_beta) {
    ss << "  device const " << scalar << "* beta [[buffer(1)]],\n";
  }
  ss << "  device " << scalar << "* out [[buffer(" << out_buffer_index
     << ")]],\n";
  ss << "  constant uint& NUM_ELEMS [[buffer(" << count_buffer_index
     << ")]],\n";
  ss << "  uint gid [[thread_position_in_grid]]) {\n";
  ss << "    if (gid >= NUM_ELEMS) return;\n";
  if (is_int_scalar && (d.activation == ActivationKind::Sign ||
                        d.activation == ActivationKind::Abs ||
                        d.activation == ActivationKind::LogicalNot ||
                        d.activation == ActivationKind::Floor ||
                        d.activation == ActivationKind::Ceil ||
                        d.activation == ActivationKind::RoundEven ||
                        d.activation == ActivationKind::RoundAway)) {
    ss << "    " << scalar << " x = in0[gid];\n";
    if (d.activation == ActivationKind::LogicalNot) {
      ss << "    out[gid] = (x == 0) ? (" << scalar << ")1 : (" << scalar
         << ")0;\n";
    } else if (d.activation == ActivationKind::Floor ||
               d.activation == ActivationKind::Ceil ||
               d.activation == ActivationKind::RoundEven ||
               d.activation == ActivationKind::RoundAway) {
      ss << "    out[gid] = x;\n";
    } else if (d.activation == ActivationKind::Abs) {
      if (is_unsigned) {
        ss << "    out[gid] = x;\n";
      } else {
        ss << "    out[gid] = (x < 0) ? (" << scalar << ")(-x) : x;\n";
      }
    } else if (d.activation == ActivationKind::Sign) {
      if (is_unsigned) {
        ss << "    out[gid] = (x > 0) ? (" << scalar << ")1 : (" << scalar
           << ")0;\n";
      } else {
        ss << "    out[gid] = (x > 0) ? (" << scalar << ")1 : ((x < 0) ? ("
           << scalar << ")(-1) : (" << scalar << ")0);\n";
      }
    }
  } else {
    ss << "    float x = static_cast<float>(in0[gid]);\n";
    if (has_runtime_swish_beta) {
      ss << "    float beta_value = static_cast<float>(beta[0]);\n";
    }
    const std::string swish_beta_expr =
        has_runtime_swish_beta ? "beta_value" : std::to_string(d.alpha) + "f";
    ss << "    out[gid] = "
       << activation_expr(d.activation, d.alpha, d.clamp_min, d.clamp_max,
                          d.gelu_tanh_approximation, swish_beta_expr)
       << ";\n";
  }
  ss << "}\n";
  return ss.str();
}

} // namespace gfx_plugin
} // namespace ov
