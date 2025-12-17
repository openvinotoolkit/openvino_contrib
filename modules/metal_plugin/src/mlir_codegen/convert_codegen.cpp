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
        case ov::element::u8:  return "uchar";
        case ov::element::i8:  return "char";
        default: return "float";
    }
}
}  // namespace

std::string generate_msl_for_convert(const ConvertCodegenDesc& d, mlir::ModuleOp /*module*/) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "using src_t = " << scalar_type(d.src_type) << ";\n";
    ss << "using dst_t = " << scalar_type(d.dst_type) << ";\n";
    ss << "kernel void convert_kernel(\n";
    ss << "  device const src_t* src [[buffer(0)]],\n";
    ss << "  device dst_t* dst [[buffer(1)]],\n";
    ss << "  constant uint& NUM_ELEMS [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    ss << "    dst[gid] = static_cast<dst_t>(src[gid]);\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
