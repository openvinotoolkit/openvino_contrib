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

std::string generate_msl_for_concat(const ConcatCodegenDesc& d, mlir::ModuleOp /*module*/) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_type(d.element_type) << ";\n";
    ss << "struct ConcatParams {\n";
    ss << "  uint outer;\n";
    ss << "  uint inner;\n";
    ss << "  uint axis_offset;\n";
    ss << "  uint axis_len;\n";
    ss << "};\n";
    ss << "kernel void concat_kernel(\n";
    ss << "  device const scalar_t* src [[buffer(0)]],\n";
    ss << "  device scalar_t* dst [[buffer(1)]],\n";
    ss << "  constant ConcatParams& p [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  uint total = p.outer * p.axis_len * p.inner;\n";
    ss << "  if (gid >= total) return;\n";
    ss << "  uint tmp = gid;\n";
    ss << "  uint outer = tmp / (p.axis_len * p.inner);\n";
    ss << "  tmp -= outer * p.axis_len * p.inner;\n";
    ss << "  uint axis = tmp / p.inner;\n";
    ss << "  uint inner = tmp - axis * p.inner;\n";
    ss << "  uint dst_idx = ((outer * (p.axis_len + p.axis_offset) + (p.axis_offset + axis)) * p.inner) + inner;\n";
    ss << "  uint src_idx = ((outer * p.axis_len + axis) * p.inner) + inner;\n";
    ss << "  dst[dst_idx] = src[src_idx];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
