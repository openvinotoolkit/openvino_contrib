// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_matmul_mpsrt.hpp"

#include <sstream>

#include "mlir/codegen_common.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

ov::element::Type resolve_matmul_epilogue_type(const ov::element::Type& type,
                                               const ov::element::Type& fallback) {
    if (type != ov::element::dynamic) {
        return type;
    }
    return fallback == ov::element::dynamic ? ov::element::f32 : fallback;
}

std::string matmul_epilogue_activation_expr(ActivationKind activation) {
    switch (activation) {
        case ActivationKind::Relu:
            return "max(x, 0.0f)";
        case ActivationKind::Sigmoid:
            return "1.0f / (1.0f + exp(-x))";
        case ActivationKind::Tanh:
            return msl_stable_tanh_expr("x");
        case ActivationKind::Gelu:
            return msl_stable_gelu_tanh_expr("x");
        case ActivationKind::Swish:
            return "x / (1.0f + precise::exp(-x))";
        case ActivationKind::HSwish:
            return "x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        case ActivationKind::HSigmoid:
            return "clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        case ActivationKind::Abs:
            return "fabs(x)";
        case ActivationKind::Sign:
            return "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)";
        case ActivationKind::Identity:
        default:
            return "x";
    }
}

}  // namespace

std::string generate_msl_for_matmul_mpsrt_epilogue(const MatMulCodegenDesc& desc) {
    const ov::element::Type output_type = resolve_matmul_epilogue_type(desc.output_type,
                                                                       desc.element_type);
    const ov::element::Type bias_type = resolve_matmul_epilogue_type(desc.bias_type,
                                                                     output_type);
    const std::string scalar_out = msl_type_from_element(output_type);
    const std::string scalar_bias = msl_type_from_element(bias_type);

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "constant uint BATCH = " << desc.batch << ";\n";
    ss << "constant uint M = " << desc.M << ";\n";
    ss << "constant uint N = " << desc.N << ";\n";
    if (desc.has_bias) {
        ss << "constant uint BIAS_B = " << desc.bias_dims[0] << ";\n";
        ss << "constant uint BIAS_M = " << desc.bias_dims[1] << ";\n";
        ss << "constant uint BIAS_N = " << desc.bias_dims[2] << ";\n";
    }
    ss << "kernel void eltwise_fused_buffer(\n";
    ss << "  device const " << scalar_out << "* gemm [[buffer(0)]],\n";
    if (desc.has_bias) {
        ss << "  device const " << scalar_bias << "* bias [[buffer(1)]],\n";
        ss << "  device " << scalar_out << "* output [[buffer(2)]],\n";
    } else {
        ss << "  device " << scalar_out << "* output [[buffer(1)]],\n";
    }
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    const uint total = BATCH * M * N;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    const uint batch = gid / (M * N);\n";
    ss << "    const uint idx = gid - batch * M * N;\n";
    ss << "    const uint row = idx / N;\n";
    ss << "    const uint col = idx - row * N;\n";
    ss << "    float x = static_cast<float>(gemm[gid]);\n";
    if (desc.has_bias) {
        ss << "    const uint bb = (BIAS_B == 1) ? 0 : batch;\n";
        ss << "    const uint bm = (BIAS_M == 1) ? 0 : row;\n";
        ss << "    const uint bn = (BIAS_N == 1) ? 0 : col;\n";
        ss << "    const uint bias_idx = (bb * BIAS_M + bm) * BIAS_N + bn;\n";
        ss << "    x += static_cast<float>(bias[bias_idx]);\n";
    }
    if (desc.has_activation) {
        ss << "    x = " << matmul_epilogue_activation_expr(desc.activation) << ";\n";
    }
    ss << "    output[gid] = static_cast<" << scalar_out << ">(x);\n";
    ss << "}\n";
    return ss.str();
}

GfxMatMulMpsrtLoweringResult lower_matmul_module_to_mpsrt_plan(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const MatMulCodegenDesc& desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b) {
    GfxMatMulMpsrtLoweringResult result{};
    result.lowering = annotate_module_with_matmul_mpsrt_plan(module, plan, desc, shape_a, shape_b);
    if (result.lowering == GfxMatMulMpsrtLoweringKind::None) {
        return result;
    }

    switch (result.lowering) {
        case GfxMatMulMpsrtLoweringKind::MpsGemm:
            result.mpsrt_plan = make_mpsrt_kernel_source_plan_from_module(module);
            break;
        case GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue:
            result.mpsrt_plan =
                make_mpsrt_kernel_source_plan_from_msl_source(module,
                                                              generate_msl_for_matmul_mpsrt_epilogue(desc));
            break;
        case GfxMatMulMpsrtLoweringKind::None:
            break;
    }
    if (!result.mpsrt_plan.valid()) {
        result.lowering = GfxMatMulMpsrtLoweringKind::None;
        return result;
    }
    return result;
}

}  // namespace gfx_plugin
}  // namespace ov
