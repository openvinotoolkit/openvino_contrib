// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace metal_plugin {

namespace {
std::string scalar_type(ov::element::Type et) {
    switch (et) {
        case ov::element::f16: return "half";
        case ov::element::f32: return "float";
        case ov::element::i32: return "int";
        case ov::element::i64: return "long";
        default: return "float";
    }
}
}  // namespace

// Uses ConvertCodegenDesc just to carry dst_type (dtype of slice tensors).
std::string generate_msl_for_slice_generic(const ConvertCodegenDesc& d, mlir::ModuleOp /*module*/) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "using scalar_t = " << scalar_type(d.dst_type) << ";\n";
    ss << "kernel void slice_kernel(\n";
    ss << "  device const scalar_t* A [[buffer(0)]],\n";
    ss << "  device scalar_t* C [[buffer(1)]],\n";
    ss << "  constant uint& TOTAL [[buffer(2)]],\n";
    ss << "  constant uint& RANK [[buffer(3)]],\n";
    ss << "  constant uint* out_shape [[buffer(4)]],\n";
    ss << "  constant uint* in_stride [[buffer(5)]],\n";
    ss << "  constant int* starts [[buffer(6)]],\n";
    ss << "  constant uint* steps [[buffer(7)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= TOTAL) return;\n";
    ss << "    uint idx = gid;\n";
    ss << "    uint in_off = 0;\n";
    ss << "    for (int d = (int)RANK - 1; d >= 0; --d) {\n";
    ss << "        uint coord = idx % out_shape[d];\n";
    ss << "        idx /= out_shape[d];\n";
    ss << "        in_off += (uint)((int)starts[d] + (int)(coord * steps[d])) * in_stride[d];\n";
    ss << "    }\n";
    ss << "    C[gid] = A[in_off];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov

