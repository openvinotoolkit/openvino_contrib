// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msl/matmul_msl.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_matmul(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::MatMul, "MSL generator supports only MatMul here");
    OPENVINO_ASSERT(op.input0 && op.input1 && op.output, "Invalid KernelOp wiring for MatMul");
    OPENVINO_ASSERT(op.M > 0 && op.N > 0 && op.K > 0, "MatMul dims must be positive");
    OPENVINO_ASSERT(op.batch > 0, "Batch dimension must be positive");
    std::ostringstream ss;
    ss << "using namespace metal;\n";
    ss << "constant uint M = " << op.M << ";\n";
    ss << "constant uint N = " << op.N << ";\n";
    ss << "constant uint K = " << op.K << ";\n";
    ss << "constant uint BATCH = " << op.batch << ";\n";
    ss << "constant uint BATCH_A = " << op.batch_a << ";\n";
    ss << "constant uint BATCH_B = " << op.batch_b << ";\n";
    ss << "constant bool B_IS_NK = " << (op.b_is_nk_layout ? "true" : "false") << ";\n";
    ss << "constant bool A_TRANSPOSE = " << (op.a_transpose ? "true" : "false") << ";\n";
    ss << "kernel void matmul_kernel(\n";
    ss << "  device const float* A [[buffer(0)]],\n";
    ss << "  device const float* B [[buffer(1)]],\n";
    ss << "  device float* C [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint total = BATCH * M * N;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    uint batch = gid / (M * N);\n";
    ss << "    uint idx = gid - batch * M * N;\n";
    ss << "    uint row = idx / N;\n";
    ss << "    uint col = idx - row * N;\n";
    ss << "    if (row < M && col < N) {\n";
    ss << "        uint batch_a = (BATCH_A == 1) ? 0 : batch;\n";
    ss << "        uint batch_b = (BATCH_B == 1) ? 0 : batch;\n";
    ss << "        device const float* Ap = A + batch_a * M * K;\n";
    ss << "        device const float* Bp = B + batch_b * K * N;\n";
    ss << "        float acc = 0.0f;\n";
    ss << "        for (uint k = 0; k < K; ++k) {\n";
    ss << "            float a = A_TRANSPOSE ? Ap[k * M + row] : Ap[row * K + k];\n";
    ss << "            float b = B_IS_NK ? Bp[col * K + k] : Bp[k * N + col];\n";
    ss << "            acc += a * b;\n";
    ss << "        }\n";
    ss << "        C[(batch * M + row) * N + col] = acc;\n";
    ss << "    }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
