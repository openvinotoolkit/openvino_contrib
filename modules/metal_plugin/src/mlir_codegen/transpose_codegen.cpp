// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_transpose(const TransposeCodegenDesc& d, mlir::ModuleOp) {
    const uint32_t rank = static_cast<uint32_t>(d.out_shape.size());
    std::string scalar_ty = "float";
    if (d.use_half) scalar_ty = "half";
    else if (d.use_int) scalar_ty = "int";
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "kernel void transpose_kernel(\n";
    ss << "  device const " << scalar_ty << "* A [[buffer(0)]],\n";
    ss << "  device " << scalar_ty << "* C [[buffer(1)]],\n";
    ss << "  constant uint& NUM_ELEMS [[buffer(2)]],\n";
    ss << "  constant uint& RANK [[buffer(3)]],\n";
    ss << "  constant uint* out_shape [[buffer(4)]],\n";
    ss << "  constant uint* perm [[buffer(5)]],\n";
    ss << "  constant uint* in_stride [[buffer(6)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    ss << "    uint idx = gid;\n";
    ss << "    uint off_in = 0;\n";
    ss << "    for (int d = (int)RANK - 1; d >= 0; --d) {\n";
    ss << "        uint coord = idx % out_shape[d];\n";
    ss << "        idx /= out_shape[d];\n";
    ss << "        uint p = perm[d];\n";
    ss << "        off_in += coord * in_stride[p];\n";
    ss << "    }\n";
    ss << "    C[gid] = A[off_in];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
