// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msl/conv_msl.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_conv2d(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Conv2D, "MSL generator supports only Conv2D here");
    OPENVINO_ASSERT(op.input0 && op.input1 && op.output, "Invalid KernelOp wiring for Conv2D");
    std::ostringstream ss;
    ss << "using namespace metal;\n";
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
    ss << "};\n";
    ss << "kernel void conv2d_kernel(\n";
    ss << "  device const float* in0 [[buffer(0)]],\n";
    ss << "  device const float* w [[buffer(1)]],\n";
    ss << "  device float* out [[buffer(2)]],\n";
    ss << "  constant ConvParams& p [[buffer(3)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint total = p.N * p.outH * p.outW * p.C_out;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    uint tmp = gid;\n";
    ss << "    uint co = tmp % p.C_out; tmp /= p.C_out;\n";
    ss << "    uint ow = tmp % p.outW; tmp /= p.outW;\n";
    ss << "    uint oh = tmp % p.outH; tmp /= p.outH;\n";
    ss << "    uint n = tmp;\n";
    ss << "    uint g = p.groups == 0 ? 0 : co / p.C_out_pg;\n";
    ss << "    uint co_g = p.groups == 0 ? co : co - g * p.C_out_pg;\n";
    ss << "    int in_h0 = int(oh) * int(p.strideH) - int(p.padTop);\n";
    ss << "    int in_w0 = int(ow) * int(p.strideW) - int(p.padLeft);\n";
    ss << "    float acc = 0.0f;\n";
    ss << "    uint cin_pg = (p.groups == 0 ? p.C_in : p.C_in_pg);\n";
    ss << "    for (uint ci = 0; ci < cin_pg; ++ci) {\n";
    ss << "        for (uint kh = 0; kh < p.kH; ++kh) {\n";
    ss << "            int ih = in_h0 + int(kh) * int(p.dilationH);\n";
    ss << "            if (ih < 0 || ih >= int(p.H)) continue;\n";
    ss << "            for (uint kw = 0; kw < p.kW; ++kw) {\n";
    ss << "                int iw = in_w0 + int(kw) * int(p.dilationW);\n";
    ss << "                if (iw < 0 || iw >= int(p.W)) continue;\n";
    ss << "                uint ci_global = (p.groups == 0 ? ci : g * p.C_in_pg + ci);\n";
    ss << "                uint in_idx = ((n * p.C_in + ci_global) * p.H + uint(ih)) * p.W + uint(iw);\n";
    ss << "                uint w_idx = (((g * p.C_out_pg + co_g) * p.C_in_pg + ci) * p.kH + kh) * p.kW + kw;\n";
    ss << "                acc += in0[in_idx] * w[w_idx];\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "    }\n";
    ss << "    uint out_idx = ((n * p.C_out + co) * p.outH + oh) * p.outW + ow;\n";
    ss << "    out[out_idx] = acc;\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
