// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_codegen/conv3d_msl.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_conv3d(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Conv3D, "generate_msl_for_conv3d: wrong op kind");
    const auto& d = op.conv3d;
    OPENVINO_ASSERT(d.N && d.C_in && d.D && d.H && d.W && d.C_out && d.kernelD && d.kernelH && d.kernelW, "Conv3D desc incomplete");

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "struct Conv3DParams {\n";
    ss << "  uint N, C_in, D, H, W;\n";
    ss << "  uint C_out;\n";
    ss << "  uint kD, kH, kW;\n";
    ss << "  uint strideD, strideH, strideW;\n";
    ss << "  uint dilationD, dilationH, dilationW;\n";
    ss << "  uint padFront, padTop, padLeft, padBack, padBottom, padRight;\n";
    ss << "  uint outD, outH, outW;\n";
    ss << "};\n";
    ss << "kernel void conv3d_kernel(\n";
    ss << "  device const float* input  [[buffer(0)]],\n";
    ss << "  device const float* weight [[buffer(1)]],\n";
    ss << "  device float*       output [[buffer(2)]],\n";
    ss << "  constant Conv3DParams& p   [[buffer(3)]],\n";
    ss << "  uint gid_x [[thread_position_in_grid.x]],\n";
    ss << "  uint gid_y [[thread_position_in_grid.y]]) {\n";
    ss << "  uint n = gid_y;\n";
    ss << "  uint oc = gid_x;\n";
    ss << "  if (n >= p.N || oc >= p.C_out) return;\n";
    ss << "  for (uint od = 0; od < p.outD; ++od) {\n";
    ss << "    for (uint oh = 0; oh < p.outH; ++oh) {\n";
    ss << "      for (uint ow = 0; ow < p.outW; ++ow) {\n";
    ss << "        float acc = 0.0f;\n";
    ss << "        for (uint ic = 0; ic < p.C_in; ++ic) {\n";
    ss << "          for (uint kd = 0; kd < p.kD; ++kd) {\n";
    ss << "            int id = int(od * p.strideD) - int(p.padFront) + int(kd * p.dilationD);\n";
    ss << "            if (id < 0 || id >= int(p.D)) continue;\n";
    ss << "            for (uint kh = 0; kh < p.kH; ++kh) {\n";
    ss << "              int ih = int(oh * p.strideH) - int(p.padTop) + int(kh * p.dilationH);\n";
    ss << "              if (ih < 0 || ih >= int(p.H)) continue;\n";
    ss << "              for (uint kw = 0; kw < p.kW; ++kw) {\n";
    ss << "                int iw = int(ow * p.strideW) - int(p.padLeft) + int(kw * p.dilationW);\n";
    ss << "                if (iw < 0 || iw >= int(p.W)) continue;\n";
    ss << "                uint in_idx = (((n * p.C_in + ic) * p.D + uint(id)) * p.H + uint(ih)) * p.W + uint(iw);\n";
    ss << "                uint w_idx  = ((((oc * p.C_in) + ic) * p.kD + kd) * p.kH + kh) * p.kW + kw;\n";
    ss << "                acc += input[in_idx] * weight[w_idx];\n";
    ss << "              }\n";
    ss << "            }\n";
    ss << "          }\n";
    ss << "        }\n";
    ss << "        uint out_idx = (((n * p.C_out + oc) * p.outD + od) * p.outH + oh) * p.outW + ow;\n";
    ss << "        output[out_idx] = acc;\n";
    ss << "      }\n";
    ss << "    }\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov

