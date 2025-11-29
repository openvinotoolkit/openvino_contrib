// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_codegen/msl_generator.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_elementwise_add(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::ElementwiseAdd, "MSL generator supports only ElementwiseAdd");
    OPENVINO_ASSERT(op.input0 && op.input1 && op.output, "Invalid KernelOp wiring for Add");

    std::ostringstream ss;
    ss << "using namespace metal;\n";
    ss << "kernel void add_kernel(\n";
    ss << "  device const float* in0 [[buffer(0)]],\n";
    ss << "  device const float* in1 [[buffer(1)]],\n";
    ss << "  device float* out [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    out[gid] = in0[gid] + in1[gid];\n";
    ss << "}\n";
    return ss.str();
}

std::string generate_msl_for_matmul(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::MatMul, "MSL generator supports only MatMul here");
    OPENVINO_ASSERT(op.input0 && op.input1 && op.output, "Invalid KernelOp wiring for MatMul");
    OPENVINO_ASSERT(op.M > 0 && op.N > 0 && op.K > 0, "MatMul dims must be positive");
    std::ostringstream ss;
    ss << "using namespace metal;\n";
    ss << "constant uint M = " << op.M << ";\n";
    ss << "constant uint N = " << op.N << ";\n";
    ss << "constant uint K = " << op.K << ";\n";
    ss << "kernel void matmul_kernel(\n";
    ss << "  device const float* A [[buffer(0)]],\n";
    ss << "  device const float* B [[buffer(1)]],\n";
    ss << "  device float* C [[buffer(2)]],\n";
    ss << "  uint2 gid [[thread_position_in_grid]]) {\n";
    ss << "    uint row = gid.y;\n";
    ss << "    uint col = gid.x;\n";
    ss << "    if (row < M && col < N) {\n";
    ss << "        float acc = 0.0f;\n";
    ss << "        for (uint k = 0; k < K; ++k) {\n";
    ss << "            acc += A[row * K + k] * B[k * N + col];\n";
    ss << "        }\n";
    ss << "        C[row * N + col] = acc;\n";
    ss << "    }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
