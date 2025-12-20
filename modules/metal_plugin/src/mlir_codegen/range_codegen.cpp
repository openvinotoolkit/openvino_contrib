// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_range(const RangeCodegenDesc& d, mlir::ModuleOp module) {
    std::string scalar_t = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumResults() >= 1) {
            scalar_t = msl_type_from_mlir(ft.getResult(0));
        }
    } else {
        switch (d.element_type) {
            case ov::element::f16: scalar_t = "half"; break;
            case ov::element::f32: scalar_t = "float"; break;
            case ov::element::i32: scalar_t = "int"; break;
            case ov::element::i64: scalar_t = "long"; break;
            default: scalar_t = "float"; break;
        }
    }

    const bool is_half = (scalar_t == "half");
    const bool is_int = (scalar_t == "int" || scalar_t == "long");

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "kernel void range_kernel(device scalar_t* O [[buffer(0)]],\n";
    ss << "                        constant uint& N [[buffer(1)]],\n";
    ss << "                        constant scalar_t& START [[buffer(2)]],\n";
    ss << "                        constant scalar_t& STEP [[buffer(3)]],\n";
    ss << "                        uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= N) return;\n";
    if (is_int) {
        ss << "    O[gid] = START + static_cast<scalar_t>(gid) * STEP;\n";
    } else if (is_half) {
        ss << "    float v = static_cast<float>(START) + static_cast<float>(gid) * static_cast<float>(STEP);\n";
        ss << "    O[gid] = static_cast<scalar_t>(v);\n";
    } else {
        ss << "    O[gid] = START + static_cast<scalar_t>(gid) * STEP;\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov

