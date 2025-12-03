// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msl/softmax_msl.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_softmax(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Softmax, "MSL generator supports only Softmax here");
    OPENVINO_ASSERT(op.input0 && op.output, "Invalid KernelOp wiring for Softmax");
    const bool is_fp16 = op.element_type == static_cast<uint32_t>(ov::element::Type_t::f16);

    std::ostringstream ss;
    ss << "using namespace metal;\n";
    ss << "struct SoftmaxParams {\n";
    ss << "  uint rows;\n";
    ss << "  uint cols;\n";
    ss << "  uint inner;\n";
    ss << "};\n";
    ss << "kernel void softmax_kernel(\n";
    if (is_fp16) {
        ss << "  device const half* in0 [[buffer(0)]],\n";
        ss << "  device half* out [[buffer(1)]],\n";
    } else {
        ss << "  device const float* in0 [[buffer(0)]],\n";
        ss << "  device float* out [[buffer(1)]],\n";
    }
    ss << "  constant SoftmaxParams& p [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint row = gid;\n";
    ss << "    if (row >= p.rows) return;\n";
    ss << "    float max_val = -INFINITY;\n";
    ss << "    uint outer = row / p.inner;\n";
    ss << "    uint inner = row - outer * p.inner;\n";
    ss << "    uint base = outer * p.cols * p.inner + inner;\n";
    if (is_fp16) {
        ss << "    device const half* row_in = in0 + base;\n";
    } else {
        ss << "    device const float* row_in = in0 + base;\n";
    }
    ss << "    for (uint c = 0; c < p.cols; ++c) {\n";
    ss << "        float v = static_cast<float>(row_in[c * p.inner]);\n";
    ss << "        if (v > max_val) max_val = v;\n";
    ss << "    }\n";
    ss << "    float sum = 0.0f;\n";
    if (is_fp16) {
        ss << "    device half* row_out = out + base;\n";
    } else {
        ss << "    device float* row_out = out + base;\n";
    }
    ss << "    for (uint c = 0; c < p.cols; ++c) {\n";
    ss << "        float e = exp(static_cast<float>(row_in[c * p.inner]) - max_val);\n";
    if (is_fp16) {
        ss << "        row_out[c * p.inner] = static_cast<half>(e);\n";
    } else {
        ss << "        row_out[c * p.inner] = e;\n";
    }
    ss << "        sum += e;\n";
    ss << "    }\n";
    ss << "    float inv_sum = 1.0f / sum;\n";
    ss << "    for (uint c = 0; c < p.cols; ++c) {\n";
    if (is_fp16) {
        ss << "        row_out[c * p.inner] = static_cast<half>((float)row_out[c * p.inner] * inv_sum);\n";
    } else {
        ss << "        row_out[c * p.inner] *= inv_sum;\n";
    }
    ss << "    }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
