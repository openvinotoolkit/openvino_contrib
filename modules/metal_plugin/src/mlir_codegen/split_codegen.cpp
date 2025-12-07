// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_split(const SplitCodegenDesc& d, mlir::ModuleOp /*module*/) {
    auto scalar = [](ov::element::Type et) {
        switch (et) {
            case ov::element::f16: return "half";
            case ov::element::f32: return "float";
            case ov::element::i32: return "int";
            case ov::element::i64: return "long";
            default: return "float";
        }
    };
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar(d.element_type) << ";\n";
    ss << "struct SplitParams { uint axis_offset; uint split_size; uint inner; uint outer; uint axis_len; };\n";
    ss << "kernel void split_kernel(device const scalar_t* src [[buffer(0)]],\n";
    ss << "                           device scalar_t* dst [[buffer(1)]],\n";
    ss << "                           constant SplitParams& p [[buffer(2)]],\n";
    ss << "                           uint gid [[thread_position_in_grid]]) {\n";
    ss << "  uint total = p.outer * p.split_size * p.inner;\n";
    ss << "  if (gid >= total) return;\n";
    ss << "  uint tmp = gid;\n";
    ss << "  uint outer_idx = tmp / (p.split_size * p.inner);\n";
    ss << "  tmp -= outer_idx * p.split_size * p.inner;\n";
    ss << "  uint axis_idx = tmp / p.inner;\n";
    ss << "  uint inner_idx = tmp - axis_idx * p.inner;\n";
    ss << "  uint src_idx = (outer_idx * (p.axis_len * p.inner)) + ((p.axis_offset + axis_idx) * p.inner) + inner_idx;\n";
    ss << "  dst[gid] = src[src_idx];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
