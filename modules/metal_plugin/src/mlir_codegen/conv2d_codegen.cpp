// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

// Simple text generator: emits a single-kernel Conv2D with optional bias, batchnorm and activation.
// Generation is parameterized by Conv2DCodegenDesc; MLIR module is not required (we don't pattern-match it).
std::string generate_msl_for_conv2d(const Conv2DCodegenDesc& d, mlir::ModuleOp /*module*/) {
    OPENVINO_ASSERT(d.N && d.C_in && d.H && d.W && d.C_out && d.kH && d.kW, "Conv2D desc missing dims");
    const bool promote_fp16 = true;
    uint32_t outH = d.outH;
    uint32_t outW = d.outW;
    const bool use_half = !promote_fp16 && d.element_type == ov::element::f16;
    const char* scalar = use_half ? "half" : "float";
    if (outH == 0) {
        int64_t eff_kh = static_cast<int64_t>(d.dilationH) * (static_cast<int64_t>(d.kH) - 1) + 1;
        outH = static_cast<uint32_t>((static_cast<int64_t>(d.H) + d.padTop + d.padBottom - eff_kh) / d.strideH + 1);
    }
    if (outW == 0) {
        int64_t eff_kw = static_cast<int64_t>(d.dilationW) * (static_cast<int64_t>(d.kW) - 1) + 1;
        outW = static_cast<uint32_t>((static_cast<int64_t>(d.W) + d.padLeft + d.padRight - eff_kw) / d.strideW + 1);
    }

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "enum Activation : uint {\n";
    ss << "  ActIdentity = 0,\n";
    ss << "  ActRelu = 1,\n";
    ss << "  ActSigmoid = 2,\n";
    ss << "  ActTanh = 3,\n";
    ss << "  ActElu = 4,\n";
    ss << "  ActPrelu = 5,\n";
    ss << "  ActGelu = 6,\n";
    ss << "  ActSwish = 7,\n";
    ss << "  ActAbs = 8,\n";
    ss << "  ActSign = 9,\n";
    ss << "  ActClamp = 10,\n";
    ss << "};\n";
    ss << "struct ConvParams {\n";
    ss << "  uint N, C_in, H, W;\n";
    ss << "  uint C_out;\n";
    ss << "  uint groups;\n";
    ss << "  uint C_in_pg;\n";
    ss << "  uint C_out_pg;\n";
    ss << "  uint kH, kW;\n";
    ss << "  uint strideH, strideW;\n";
    ss << "  uint dilationH, dilationW;\n";
    ss << "  uint padTop, padLeft;\n";
    ss << "  uint padBottom, padRight;\n";
    ss << "  uint outH, outW;\n";
    ss << "  uint has_bias;\n";
    ss << "  uint has_bn;\n";
    ss << "  uint activation;\n";
    ss << "  float alpha;\n";
    ss << "  float epsilon;\n";
    ss << "  float clamp_min;\n";
    ss << "  float clamp_max;\n";
    ss << "};\n";

    ss << "kernel void conv2d_kernel(\n";
    ss << "  device const " << scalar << "* in0   [[buffer(0)]],\n";
    ss << "  device const " << scalar << "* w     [[buffer(1)]],\n";
    ss << "  device const " << scalar << "* bias  [[buffer(2)]],\n";
    ss << "  device const " << scalar << "* gamma [[buffer(3)]],\n";
    ss << "  device const " << scalar << "* beta  [[buffer(4)]],\n";
    ss << "  device const " << scalar << "* mean  [[buffer(5)]],\n";
    ss << "  device const " << scalar << "* var   [[buffer(6)]],\n";
    ss << "  device " << scalar << "* out         [[buffer(7)]],\n";
    ss << "  constant ConvParams& p    [[buffer(8)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint total = p.N * p.outH * p.outW * p.C_out;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    uint tmp = gid;\n";
    ss << "    uint ow = tmp % p.outW; tmp /= p.outW;\n";
    ss << "    uint oh = tmp % p.outH; tmp /= p.outH;\n";
    ss << "    uint co = tmp % p.C_out; tmp /= p.C_out;\n";
    ss << "    uint n = tmp;\n";
    ss << "    uint g = (p.groups == 0 || p.groups == 1) ? 0 : co / p.C_out_pg;\n";
    ss << "    uint co_g = (p.groups == 0 || p.groups == 1) ? co : co - g * p.C_out_pg;\n";
    ss << "    int in_h0 = int(oh) * int(p.strideH) - int(p.padTop);\n";
    ss << "    int in_w0 = int(ow) * int(p.strideW) - int(p.padLeft);\n";
    ss << "    float acc = 0.0f;\n";
    ss << "    uint cin_pg = (p.groups == 0 || p.groups == 1) ? p.C_in : p.C_in_pg;\n";
    ss << "    for (uint ci = 0; ci < cin_pg; ++ci) {\n";
    ss << "        for (uint kh = 0; kh < p.kH; ++kh) {\n";
    ss << "            int ih = in_h0 + int(kh) * int(p.dilationH);\n";
    ss << "            if (ih < 0 || ih >= int(p.H)) continue;\n";
    ss << "            for (uint kw = 0; kw < p.kW; ++kw) {\n";
    ss << "                int iw = in_w0 + int(kw) * int(p.dilationW);\n";
    ss << "                if (iw < 0 || iw >= int(p.W)) continue;\n";
    ss << "                uint ci_global = (p.groups == 0 || p.groups == 1) ? ci : g * p.C_in_pg + ci;\n";
    ss << "                uint in_idx = ((n * p.C_in + ci_global) * p.H + uint(ih)) * p.W + uint(iw);\n";
    ss << "                uint w_idx = (((g * p.C_out_pg + co_g) * p.C_in_pg + ci) * p.kH + kh) * p.kW + kw;\n";
    ss << "                acc = fma((float)in0[in_idx], (float)w[w_idx], acc);\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "    }\n";
    ss << "    if (p.has_bias) {\n";
    ss << "        acc += (float)bias[co];\n";
    ss << "    }\n";
    ss << "    if (p.has_bn) {\n";
    ss << "        float g_scale = (float)gamma[co];\n";
    ss << "        float b_shift = (float)beta[co];\n";
    ss << "        float m = (float)mean[co];\n";
    ss << "        float v = (float)var[co];\n";
    ss << "        float inv_std = rsqrt(v + p.epsilon);\n";
    ss << "        acc = g_scale * (acc - m) * inv_std + b_shift;\n";
    ss << "    }\n";
    ss << "    switch (p.activation) {\n";
    ss << "      case ActRelu: acc = max(acc, 0.0f); break;\n";
    ss << "      case ActSigmoid: acc = 1.0f / (1.0f + exp(-acc)); break;\n";
    ss << "      case ActTanh: acc = tanh(acc); break;\n";
    ss << "      case ActElu: acc = (acc > 0.0f) ? acc : (exp(acc) - 1.0f) * p.alpha; break;\n";
    ss << "      case ActPrelu: acc = (acc >= 0.0f) ? acc : acc * p.alpha; break;\n";
    ss << "      case ActGelu: acc = 0.5f * acc * (1.0f + tanh(0.79788456f * (acc + 0.044715f * acc * acc * acc))); break;\n";
    ss << "      case ActSwish: acc = acc / (1.0f + exp(-acc)); break;\n";
    ss << "      case ActAbs: acc = fabs(acc); break;\n";
    ss << "      case ActSign: acc = (acc > 0.0f) ? 1.0f : (acc < 0.0f ? -1.0f : 0.0f); break;\n";
    ss << "      case ActClamp: acc = clamp(acc, p.clamp_min, p.clamp_max); break;\n";
    ss << "      case ActIdentity:\n";
    ss << "      default: break;\n";
    ss << "    }\n";
    ss << "    uint out_idx = ((n * p.C_out + co) * p.outH + oh) * p.outW + ow;\n";
    if (use_half) {
        ss << "    out[out_idx] = static_cast<" << scalar << ">(acc);\n";
    } else {
        ss << "    out[out_idx] = acc;\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
