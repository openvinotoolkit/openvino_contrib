// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msl/batchnorm_msl.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_batchnorm2d(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::BatchNorm2D, "MSL generator supports only BatchNorm2D here");
    OPENVINO_ASSERT(op.input0 && op.output, "Invalid KernelOp wiring for BatchNorm2D");
    std::ostringstream ss;
    ss << "using namespace metal;\n";
    ss << "struct BNParams {\n";
    ss << "  uint N, C, H, W;\n";
    ss << "};\n";
    ss << "kernel void batchnorm2d_kernel(\n";
    ss << "  device const float* in0 [[buffer(0)]],\n";
    ss << "  device const float* params [[buffer(1)]],\n";
    ss << "  device float* out [[buffer(2)]],\n";
    ss << "  constant BNParams& p [[buffer(3)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint total = p.N * p.C * p.H * p.W;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    uint tmp = gid;\n";
    ss << "    uint w = tmp % p.W; tmp /= p.W;\n";
    ss << "    uint h = tmp % p.H; tmp /= p.H;\n";
    ss << "    uint c = tmp % p.C; tmp /= p.C;\n";
    ss << "    uint n = tmp;\n";
    ss << "    float gamma = params[c];\n";
    ss << "    float beta  = params[p.C + c];\n";
    ss << "    float mean  = params[2 * p.C + c];\n";
    ss << "    float var   = params[3 * p.C + c];\n";
    ss << "    float eps   = params[4 * p.C];\n";
    ss << "    uint in_idx = ((n * p.C + c) * p.H + h) * p.W + w;\n";
    ss << "    float x = in0[in_idx];\n";
    ss << "    float y = gamma * (x - mean) / sqrt(var + eps) + beta;\n";
    ss << "    out[in_idx] = y;\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov

