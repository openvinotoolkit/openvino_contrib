// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_convert(const ConvertCodegenDesc& d, mlir::ModuleOp module) {
    std::ostringstream ss;
    std::string src_t = "float";
    std::string dst_t = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            src_t = msl_type_from_mlir(ft.getInput(0));
        }
        if (ft.getNumResults() >= 1) {
            dst_t = msl_type_from_mlir(ft.getResult(0));
        }
    } else {
        // Fallback to descriptor types if MLIR module is not provided.
        switch (d.src_type) {
            case ov::element::f16: src_t = "half"; break;
            case ov::element::f32: src_t = "float"; break;
            case ov::element::i32: src_t = "int"; break;
            case ov::element::i64: src_t = "long"; break;
            case ov::element::u8:  src_t = "uchar"; break;
            case ov::element::i8:  src_t = "char"; break;
            default: break;
        }
        switch (d.dst_type) {
            case ov::element::f16: dst_t = "half"; break;
            case ov::element::f32: dst_t = "float"; break;
            case ov::element::i32: dst_t = "int"; break;
            case ov::element::i64: dst_t = "long"; break;
            case ov::element::u8:  dst_t = "uchar"; break;
            case ov::element::i8:  dst_t = "char"; break;
            default: break;
        }
    }
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "using src_t = " << src_t << ";\n";
    ss << "using dst_t = " << dst_t << ";\n";
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

}  // namespace gfx_plugin
}  // namespace ov
