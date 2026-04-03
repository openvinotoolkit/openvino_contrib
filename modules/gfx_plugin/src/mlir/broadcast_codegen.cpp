// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_broadcast(const BroadcastCodegenDesc& d, mlir::ModuleOp module) {
    std::string scalar_t = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar_t = msl_type_from_mlir(ft.getInput(0));
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

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "kernel void broadcast_kernel(device const scalar_t* A [[buffer(0)]],\n";
    ss << "                            device scalar_t* O [[buffer(1)]],\n";
    ss << "                            constant uint& NUM_ELEMS [[buffer(2)]],\n";
    ss << "                            constant uint& OUT_RANK [[buffer(3)]],\n";
    ss << "                            constant uint& IN_RANK [[buffer(4)]],\n";
    ss << "                            constant int* OUT_DIMS [[buffer(5)]],\n";
    ss << "                            constant int* IN_DIMS [[buffer(6)]],\n";
    ss << "                            constant int* IN_STRIDES [[buffer(7)]],\n";
    ss << "                            constant int* AXES [[buffer(8)]],\n";
    ss << "                            uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    ss << "    uint idx = gid;\n";
    ss << "    int out_coords[8];\n";
    ss << "    for (uint d = OUT_RANK; d-- > 0;) {\n";
    ss << "        out_coords[d] = idx % OUT_DIMS[d];\n";
    ss << "        idx /= OUT_DIMS[d];\n";
    ss << "    }\n";
    ss << "    int in_idx = 0;\n";
    ss << "    for (uint i = 0; i < IN_RANK; ++i) {\n";
    ss << "        int axis = AXES[i];\n";
    ss << "        int coord = out_coords[axis];\n";
    ss << "        if (IN_DIMS[i] == 1) coord = 0;\n";
    ss << "        in_idx += coord * IN_STRIDES[i];\n";
    ss << "    }\n";
    ss << "    O[gid] = A[in_idx];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov

