// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_shapeof(const ShapeOfCodegenDesc& d, mlir::ModuleOp module) {
    std::ostringstream ss;
    std::string scalar_t = "int";
    if (d.element_type != ov::element::dynamic) {
        scalar_t = msl_type_from_element(d.element_type);
    } else if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumResults() >= 1) {
            scalar_t = msl_type_from_mlir(ft.getResult(0));
        }
    } else {
        switch (d.element_type) {
            case ov::element::i32: scalar_t = "int"; break;
            case ov::element::i64: scalar_t = "long"; break;
            default: break;
        }
    }
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "kernel void shapeof_kernel(\n";
    ss << "  device const char* src [[buffer(0)]],\n";
    ss << "  device scalar_t* out [[buffer(1)]],\n";
    ss << "  constant uint& RANK [[buffer(2)]],\n";
    ss << "  constant scalar_t* shape [[buffer(3)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  (void)src;\n";
    ss << "  if (gid >= RANK) return;\n";
    ss << "  out[gid] = shape[gid];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
