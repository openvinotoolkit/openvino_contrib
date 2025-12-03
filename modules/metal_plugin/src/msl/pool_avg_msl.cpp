// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msl/pool_avg_msl.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_avgpool2d(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::AvgPool2D, "MSL generator supports only AvgPool2D here");
    OPENVINO_ASSERT(op.input0 && op.output, "Invalid KernelOp wiring for AvgPool2D");
    std::ostringstream ss;
    ss << "using namespace metal;\n";
    ss << "struct PoolParams {\n";
    ss << "  uint N, H, W, C;\n";
    ss << "  uint outH, outW;\n";
    ss << "  uint kH, kW;\n";
    ss << "  uint strideH, strideW;\n";
    ss << "  uint padTop, padLeft;\n";
    ss << "  uint exclude_pad;\n";
    ss << "};\n";
    ss << "kernel void avgpool2d_kernel(\n";
    ss << "  device const float* in0 [[buffer(0)]],\n";
    ss << "  device float* out [[buffer(1)]],\n";
    ss << "  constant PoolParams& p [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint total = p.N * p.outH * p.outW * p.C;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    uint tmp = gid;\n";
    ss << "    uint c = tmp % p.C; tmp /= p.C;\n";
    ss << "    uint ow = tmp % p.outW; tmp /= p.outW;\n";
    ss << "    uint oh = tmp % p.outH; tmp /= p.outH;\n";
    ss << "    uint n = tmp;\n";
    ss << "    int in_h0 = int(oh) * int(p.strideH) - int(p.padTop);\n";
    ss << "    int in_w0 = int(ow) * int(p.strideW) - int(p.padLeft);\n";
    ss << "    float sum = 0.0f;\n";
    ss << "    uint count = 0;\n";
    ss << "    for (uint kh = 0; kh < p.kH; ++kh) {\n";
    ss << "        int ih = in_h0 + int(kh);\n";
    ss << "        if (ih < 0 || ih >= int(p.H)) continue;\n";
    ss << "        for (uint kw = 0; kw < p.kW; ++kw) {\n";
    ss << "            int iw = in_w0 + int(kw);\n";
    ss << "            if (iw < 0 || iw >= int(p.W)) continue;\n";
    ss << "            uint in_idx = ((n * p.C + c) * p.H + uint(ih)) * p.W + uint(iw);\n";
    ss << "            sum += in0[in_idx];\n";
    ss << "            ++count;\n";
    ss << "        }\n";
    ss << "    }\n";
    ss << "    float denom = p.exclude_pad ? float(count) : float(p.kH * p.kW);\n";
    ss << "    float val = (denom == 0.0f) ? 0.0f : (sum / denom);\n";
    ss << "    uint out_idx = ((n * p.C + c) * p.outH + oh) * p.outW + ow;\n";
    ss << "    out[out_idx] = val;\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
