// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_reverse(const ReverseCodegenDesc& d, mlir::ModuleOp module) {
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

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "struct ReverseParams {\n";
    ss << "    uint rank;\n";
    ss << "    uint total;\n";
    ss << "    uint axes_mask;\n";
    ss << "    uint dims[" << ReverseCodegenDesc::kMaxDims << "];\n";
    ss << "    uint strides[" << ReverseCodegenDesc::kMaxDims << "];\n";
    ss << "};\n";
    ss << "kernel void reverse_kernel(device const scalar_t* I [[buffer(0)]],\n";
    ss << "                         device scalar_t* O [[buffer(1)]],\n";
    ss << "                         constant ReverseParams& P [[buffer(2)]],\n";
    ss << "                         uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= P.total) return;\n";
    ss << "    uint tmp = gid;\n";
    ss << "    uint offset = 0;\n";
    ss << "    for (uint i = 0; i < P.rank; ++i) {\n";
    ss << "        uint stride = P.strides[i];\n";
    ss << "        uint coord = stride > 0 ? (tmp / stride) : 0;\n";
    ss << "        tmp = stride > 0 ? (tmp % stride) : 0;\n";
    ss << "        uint dim = P.dims[i];\n";
    ss << "        if ((P.axes_mask >> i) & 1u) {\n";
    ss << "            coord = (dim - 1u) - coord;\n";
    ss << "        }\n";
    ss << "        offset += coord * stride;\n";
    ss << "    }\n";
    ss << "    O[gid] = I[offset];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
