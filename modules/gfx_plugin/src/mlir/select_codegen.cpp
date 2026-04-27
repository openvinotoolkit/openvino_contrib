// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_select(mlir::ModuleOp module, ov::element::Type et) {
    std::string scalar_t = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 2) {
            scalar_t = msl_type_from_mlir(ft.getInput(1));
        }
    } else {
        switch (et) {
            case ov::element::f16: scalar_t = "half"; break;
            case ov::element::f32: scalar_t = "float"; break;
            case ov::element::i32: scalar_t = "int"; break;
            case ov::element::i64: scalar_t = "long"; break;
            case ov::element::boolean: scalar_t = "bool"; break;
            default: break;
        }
    }
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "kernel void select_kernel(device const uchar* C [[buffer(0)]],\n";
    ss << "                           device const scalar_t* A [[buffer(1)]],\n";
    ss << "                           device const scalar_t* B [[buffer(2)]],\n";
    ss << "                           device scalar_t* O [[buffer(3)]],\n";
    ss << "                           constant uint& NUM_ELEMS [[buffer(4)]],\n";
    ss << "                           constant uint& RANK [[buffer(5)]],\n";
    ss << "                           constant int* OUT_DIMS [[buffer(6)]],\n";
    ss << "                           constant int* STRIDE_C [[buffer(7)]],\n";
    ss << "                           constant int* STRIDE_A [[buffer(8)]],\n";
    ss << "                           constant int* STRIDE_B [[buffer(9)]],\n";
    ss << "                           uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    ss << "    uint idx = gid;\n";
    ss << "    int off_c = 0; int off_a = 0; int off_b = 0;\n";
    ss << "    for (uint d = RANK; d-- > 0;) {\n";
    ss << "        int coord = idx % OUT_DIMS[d];\n";
    ss << "        idx /= OUT_DIMS[d];\n";
    ss << "        off_c += (STRIDE_C[d] == 0 ? 0 : coord * STRIDE_C[d]);\n";
    ss << "        off_a += (STRIDE_A[d] == 0 ? 0 : coord * STRIDE_A[d]);\n";
    ss << "        off_b += (STRIDE_B[d] == 0 ? 0 : coord * STRIDE_B[d]);\n";
    ss << "    }\n";
    ss << "    O[gid] = (C[off_c] != 0) ? A[off_a] : B[off_b];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
